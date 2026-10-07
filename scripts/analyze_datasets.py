#!/usr/bin/env python3
"""Raw-stage dataset audit (docs/dataset_analysis_plan.md, the [raw] analyses):
per-image quality/noise/frequency metrics, artifact flags, lesion statistics,
DICOM header dumps, near-duplicate + split-leakage detection, outliers, MRI
case-level (3D) statistics, a claimed-vs-verified summary, and review figures.

Outputs per dataset under --out/<dataset>/:
    per_image.parquet, thumbs.npy, dicom_headers.parquet, cases.parquet,
    duplicates.csv, leakage.csv, outliers.csv, outliers_review.csv (manual,
    never overwritten), summary.json, figures/*.png
and, with --cross, cross-modality duplicates and violin plots under
--out/cross_dataset/.

Usage:
    .venv/bin/python scripts/analyze_datasets.py --dataset busi --workers 8
    .venv/bin/python scripts/analyze_datasets.py --dataset all --cross
    .venv/bin/python scripts/analyze_datasets.py --dataset cmmd --limit 200   # smoke test
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from audit import duplicates, report, volumes  # noqa: E402
from audit.analyze import analyze_sample  # noqa: E402
from audit.sources import ADAPTERS, DATASET_ROOTS, MODALITY, case_dirs  # noqa: E402

THUMB_SIZE = {'ultrasound': 128, 'mammography': 128, 'mri': 64}


def _run_parallel(function, items, workers, label):
    results = []
    started = time.time()
    if workers <= 1:
        iterator = map(function, items)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(function, items, chunksize=max(1, min(32, len(items) // (workers * 8) or 1)))
    for k, result in enumerate(iterator, 1):
        results.append(result)
        if k % 500 == 0 or k == len(items):
            print(f'  {label}: {k}/{len(items)} ({time.time() - started:.0f}s)', flush=True)
    if workers > 1:
        executor.shutdown()
    return results


class _SampleJob:
    def __init__(self, thumb_size):
        self.thumb_size = thumb_size

    def __call__(self, sample):
        return analyze_sample(sample, self.thumb_size)


def _case_job(item):
    dataset, path = item
    try:
        if dataset == 'ispy2':
            return volumes.analyze_ispy2_case(path)
        return volumes.analyze_breadm_seg3d_case(path)
    except Exception as error:
        return {'group': Path(path).name, 'load_error': repr(error)}


def _to_parquet(frame, path):
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(lambda v: None if v is None else str(v))
    frame.to_parquet(path, index=False)


def analyze_dataset(name, args):
    out_dir = args.out / name
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f'=== {name}: enumerating samples')
    samples = [] if args.reuse and (out_dir / 'per_image.parquet').exists() else ADAPTERS[name](REPO_ROOT / DATASET_ROOTS[name])
    if args.limit and len(samples) > args.limit:
        rng = np.random.default_rng(42)
        samples = [samples[i] for i in sorted(rng.choice(len(samples), args.limit, replace=False))]
    print(f'  {len(samples)} samples ({sum(s.used for s in samples)} used by the training loader)')

    if args.reuse and (out_dir / 'per_image.parquet').exists():
        # re-run duplicates/outliers/summary/figures on saved per-image results
        frame = pd.read_parquet(out_dir / 'per_image.parquet')
        thumbs = np.load(out_dir / 'thumbs.npy')
        headers = pd.read_parquet(out_dir / 'dicom_headers.parquet') if (out_dir / 'dicom_headers.parquet').exists() else None
        cases = pd.read_parquet(out_dir / 'cases.parquet') if (out_dir / 'cases.parquet').exists() else None
        print(f'  reusing {len(frame)} saved per-image rows')
        return _finish_dataset(name, args, out_dir, frame, thumbs, headers, cases)

    results = _run_parallel(_SampleJob(THUMB_SIZE[MODALITY[name]]), samples, args.workers, 'images')
    frame = pd.DataFrame([row for row, _, _ in results])
    size = THUMB_SIZE[MODALITY[name]]
    thumbs = np.stack([t if t is not None else np.zeros((size, size), np.uint8) for _, _, t in results])
    header_rows = [{'sample_id': row['sample_id'], **header} for row, header, _ in results if header]
    headers = pd.DataFrame(header_rows) if header_rows else None

    _to_parquet(frame, out_dir / 'per_image.parquet')
    np.save(out_dir / 'thumbs.npy', thumbs)
    if headers is not None:
        _to_parquet(headers, out_dir / 'dicom_headers.parquet')

    cases = None
    dirs = [] if args.skip_cases else case_dirs(name)
    if args.limit:
        dirs = dirs[:max(1, args.limit // 20)]
    if dirs:
        print(f'  case-level volumes: {len(dirs)}')
        cases = pd.DataFrame(_run_parallel(_case_job, [(name, d) for d in dirs], args.workers, 'cases'))
        _to_parquet(cases, out_dir / 'cases.parquet')
    return _finish_dataset(name, args, out_dir, frame, thumbs, headers, cases)


def _finish_dataset(name, args, out_dir, frame, thumbs, headers, cases):
    figures_dir = out_dir / 'figures'
    print('  duplicates')
    loaded = frame['load_error'].isna() if 'load_error' in frame else pd.Series(True, index=frame.index)
    pairs, candidates = duplicates.find_duplicates(frame, thumbs, skip_same_group=MODALITY[name] == 'mri')
    print(f'  {candidates} hash candidates checked with SSIM')
    leaks = duplicates.split_leakage(pairs)
    conflicts = duplicates.group_split_conflicts(frame)
    pairs.to_csv(out_dir / 'duplicates.csv', index=False)
    leaks.to_csv(out_dir / 'leakage.csv', index=False)

    outliers = report.flag_outliers(frame[loaded])
    outliers.to_csv(out_dir / 'outliers.csv', index=False)
    report.update_review_file(out_dir / 'outliers_review.csv', outliers)

    summary = report.summarize(name, frame, headers, pairs, leaks, conflicts, candidates, cases)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))

    print('  figures')
    report.metric_distributions(frame, figures_dir / 'metric_distributions.png')
    if MODALITY[name] != 'mri':
        report.mean_std_images(frame, thumbs, figures_dir / 'mean_std_images.png')
    report.lesion_centroids(frame, figures_dir / 'lesion_centroids.png')
    if not args.no_sheets:
        report.review_sheets(frame, outliers, pairs, figures_dir, rng=42)
    d = summary['duplicates']
    print(f"  done: {len(frame)} images, {summary['load_errors']} load errors, "
          f"{d['groups']} duplicate groups, {d['split_leak_pairs']} split-leak pairs, "
          f"{len(outliers)} outliers flagged -> {out_dir}")
    return frame, thumbs


def cross_dataset(args):
    out_dir = args.out / 'cross_dataset'
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name in ADAPTERS:
        path = args.out / name / 'per_image.parquet'
        if path.exists():
            frames[name] = (pd.read_parquet(path), np.load(args.out / name / 'thumbs.npy'))
    if not frames:
        print('no per-dataset results yet; run without --cross-only first')
        return
    report.cross_dataset_violins([f for f, _ in frames.values()], out_dir / 'metric_violins.png')
    for modality in ('ultrasound', 'mammography'):
        names = [n for n in frames if MODALITY[n] == modality]
        if len(names) < 2:
            continue
        frame = pd.concat([frames[n][0] for n in names], ignore_index=True)
        thumbs = np.concatenate([frames[n][1] for n in names])
        pairs, candidates = duplicates.find_duplicates(frame, thumbs)
        print(f'  {modality}: {candidates} hash candidates checked with SSIM')
        if len(pairs):
            pairs = pairs[pairs['dataset_a'] != pairs['dataset_b']]
        pairs.to_csv(out_dir / f'{modality}_cross_duplicates.csv', index=False)
        print(f'  {modality}: {len(pairs)} cross-dataset duplicate pairs among {names}')
        if len(pairs):
            first = pairs.drop_duplicates('group_id').head(18)
            rows, captions = [], []
            for _, pair in first.iterrows():
                rows += [frame.iloc[pair['i']], frame.iloc[pair['j']]]
                captions += [f"{pair['dataset_a']}:{pair['sample_id_a']}", f"{pair['dataset_b']}:{pair['sample_id_b']}"]
            report.contact_sheet(rows, captions, out_dir / f'{modality}_cross_duplicates.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', nargs='+', default=['all'], choices=['all', *ADAPTERS])
    parser.add_argument('--out', type=Path, default=REPO_ROOT / 'analysis')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='Random subset of N images per dataset (smoke test).')
    parser.add_argument('--skip-cases', action='store_true', help='Skip the MRI case-level volume analysis.')
    parser.add_argument('--no-sheets', action='store_true', help='Skip contact sheets (they re-read raw images).')
    parser.add_argument('--reuse', action='store_true',
                        help='Reuse saved per_image.parquet/thumbs.npy and only redo duplicates, outliers, summary and figures.')
    parser.add_argument('--cross', action='store_true', help='Also run cross-dataset duplicates and plots afterwards.')
    parser.add_argument('--cross-only', action='store_true', help='Only run the cross-dataset step on existing results.')
    args = parser.parse_args()

    if not args.cross_only:
        names = list(ADAPTERS) if 'all' in args.dataset else args.dataset
        for name in names:
            analyze_dataset(name, args)
    if args.cross or args.cross_only:
        print('=== cross-dataset')
        cross_dataset(args)


if __name__ == '__main__':
    main()
