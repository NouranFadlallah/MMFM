"""Convert AMBL lesions into BreaDM-compatible img9Se arrays for the MRI branch.

Mirrors scripts/preprocess_ispy2_mri.py so the MRI-branch sources share one
crop rule: per lesion, every axial slice with at least 4 mask pixels becomes a
crop of the mask's bounding box (all fragments) plus a 6-pixel margin, with 9
channels sampled from the DCE phases by linspace (AMBL has 5, so phases repeat
0,0,1,1,2,2,3,3,4 as in ISPY2) and the pre-contrast volume's 1-99th percentile
window. Masks come from the DICOM SEG placed on the original DCE grid
(audit/ambl.py).

Labels come from the audit's side-checked resolution
(analysis/ambl/lesions_resolved.csv, docs/dataset_audit.md): ambiguous lesions
(tumour and benign in one breast) and post-neoadjuvant scans (referral 6) are
left out. Splits are patient-level and stratified on whether the patient has a
malignant lesion: 70 / 15 / 15 with a fixed seed. Writes
<dest>/{train,val,test}/{Benign,Malignant}/<patient>_lesion<k>/p-<z>.npy and
<dest>/lesions.csv.

Usage:
    python3 scripts/preprocess_ambl_mri.py --dest datasets/MRI/AMBL/cls/img9Se
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit import ambl  # noqa: E402

RAW_ROOT = Path('/mnt/data/mmfm_datasets/MRI/ambl/dicom')
RESOLVED = Path('analysis/ambl/lesions_resolved.csv')
NUM_PHASES = 9
BBOX_MARGIN = 6
MIN_LESION_PIXELS = 4
POST_NAT_REFERRAL = 6


def _intensity_range(volume):
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(nonzero, [1, 99])
    return float(low), float(high if high > low else low + 1.0)


def _to_uint8(patch, low, high):
    return (np.clip((patch.astype(np.float32) - low) / (high - low), 0.0, 1.0) * 255.0).astype(np.uint8)


def patient_splits(lesions, seed=42, fractions=(0.70, 0.15, 0.15)):
    """Patient-level train/val/test, stratified on 'patient has a malignant lesion'."""
    rng = np.random.default_rng(seed)
    patients = lesions.groupby('group')['label'].max()
    split_of = {}
    for _, group in patients.groupby(patients):
        ids = group.index.to_numpy(copy=True)
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * fractions[1]))
        n_test = max(1, round(len(ids) * fractions[2]))
        for k, patient in enumerate(ids):
            split_of[patient] = 'val' if k < n_val else 'test' if k < n_val + n_test else 'train'
    return split_of


def lesion_crops(data, mask):
    """Yield (z, (h, w, 9) uint8) for each slice of one lesion."""
    phase_index = np.round(np.linspace(0, len(data['files']) - 1, NUM_PHASES)).astype(int)
    slices = [z for z in np.where(mask.any(axis=(1, 2)))[0] if mask[z].sum() >= MIN_LESION_PIXELS]
    if not slices:
        return
    pre = ambl._slices(data['files'][0], range(len(data['z'])))
    low, high = _intensity_range(pre)
    by_phase = {p: ambl._slices(data['files'][p], slices) for p in sorted(set(phase_index))}
    height, width = mask.shape[1:]
    for k, z in enumerate(slices):
        rows, cols = np.where(mask[z])
        r0, r1 = max(rows.min() - BBOX_MARGIN, 0), min(rows.max() + BBOX_MARGIN + 1, height)
        c0, c1 = max(cols.min() - BBOX_MARGIN, 0), min(cols.max() + BBOX_MARGIN + 1, width)
        yield int(z), np.stack([_to_uint8(by_phase[p][k, r0:r1, c0:c1], low, high) for p in phase_index], axis=-1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dest', type=Path, default=Path('datasets/MRI/AMBL/cls/img9Se'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    resolved = pd.read_csv(RESOLVED)
    keep = resolved[resolved['label'].notna() & (resolved['referral'] != POST_NAT_REFERRAL)].copy()
    keep['label'] = keep['label'].astype(int)
    split_of = patient_splits(keep, args.seed)
    rows = []
    for patient, lesions in keep.groupby('group'):
        data = ambl.load_patient(RAW_ROOT / patient)
        for _, lesion in lesions.iterrows():
            k = int(lesion['segment'])
            split, class_name = split_of[patient], ('Malignant' if lesion['label'] == 1 else 'Benign')
            out_dir = args.dest / split / class_name / f'{patient}_lesion{k}'
            out_dir.mkdir(parents=True, exist_ok=True)
            n = 0
            for z, array in lesion_crops(data, data['masks'][k - 1]):
                np.save(out_dir / f'p-{z:03d}.npy', array)
                n += 1
            rows.append({'patient': patient, 'segment': k, 'label': int(lesion['label']), 'split': split,
                         'label_method': lesion['label_method'], 'slices': n})
        print(f'{patient}: {len(lesions)} lesion(s)', flush=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.dest / 'lesions.csv', index=False)
    print(manifest.groupby(['split', 'label']).agg(lesions=('segment', 'size'), slices=('slices', 'sum'),
                                                  patients=('patient', 'nunique')))


if __name__ == '__main__':
    main()
