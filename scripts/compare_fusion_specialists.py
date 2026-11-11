"""Compare the fold-aligned fusion model against single-dataset specialists on
identical test images (paper Objective 1.2: does one model match the specialists?).

Inputs are the per-sample prediction files written by training/train.py:
  fusion      --dataset combined --combined-fold K  ->  <ckpt>_test_source_metrics.npz
              (arrays probs, labels, source, image)
  specialist  --single-mode --kfold N               ->  <ckpt>_foldK_test_predictions.npz
              (arrays probs, labels, image)

Predictions are joined on the image path, so a specialist file only contributes
images that are also in the fusion's test split for that source. For each source
the script reports both AUCs, their difference with a paired bootstrap 95% CI
(resampling images, or patients if --group-by-parent), and DeLong's paired p-value.

Usage:
  python scripts/compare_fusion_specialists.py \
      --fusion runs/fold/combined_fold{fold}_test_source_metrics.npz \
      --specialist busi=runs/busi_dup/busi_dup_s42_fold{fold}_test_predictions.npz \
      --specialist mias=runs/mias/mias_fold{fold}_test_predictions.npz \
      --folds 1 2 3 4 5 --out runs/fold/fusion_vs_specialists.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def delong_paired(y, p1, p2):
    """Two-sided p-value of DeLong's test for two correlated AUCs (DeLong et al. 1988)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def components(p):
        x, z = p[pos], p[neg]
        psi = (x[:, None] > z[None, :]) + 0.5 * (x[:, None] == z[None, :])
        return psi.mean(axis=1), psi.mean(axis=0), psi.mean()

    v10_1, v01_1, auc1 = components(p1)
    v10_2, v01_2, auc2 = components(p2)
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    if var <= 0:
        return 1.0
    return float(2 * stats.norm.sf(abs(auc1 - auc2) / np.sqrt(var)))


def paired_bootstrap(y, p1, p2, groups, n_boot=2000, seed=0):
    """95% CI of AUC(p1) - AUC(p2), resampling whole groups with replacement."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in unique}
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([index_of[g] for g in rng.choice(unique, len(unique))])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fusion', required=True, help='fusion npz path with {fold}')
    parser.add_argument('--specialist', action='append', default=[], metavar='SOURCE=PATH',
                        help='specialist npz path with {fold}; repeat per source')
    parser.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--group-by-parent', action='store_true',
                        help='bootstrap over image parent folders (patients for per-patient folder layouts)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)

    specialists = dict(item.split('=', 1) for item in args.specialist)
    pooled = {source: {'y': [], 'fusion': [], 'specialist': [], 'group': []} for source in specialists}
    per_fold = []
    for fold in args.folds:
        fusion = np.load(args.fusion.format(fold=fold), allow_pickle=False)
        for source, pattern in specialists.items():
            path = Path(pattern.format(fold=fold))
            if not path.exists():
                print(f'fold {fold} {source}: missing {path}, skipped')
                continue
            spec = np.load(path, allow_pickle=False)
            mask = fusion['source'] == source
            fusion_prob = dict(zip(fusion['image'][mask], fusion['probs'][mask]))
            fusion_label = dict(zip(fusion['image'][mask], fusion['labels'][mask]))
            shared = [i for i, image in enumerate(spec['image']) if image in fusion_prob]
            if len(shared) < len(spec['image']):
                print(f'fold {fold} {source}: {len(spec["image"]) - len(shared)} specialist images not in fusion test')
            images = spec['image'][shared]
            y = spec['labels'][shared]
            assert all(fusion_label[image] == label for image, label in zip(images, y)), f'label mismatch in {source}'
            p_spec, p_fus = spec['probs'][shared], np.array([fusion_prob[image] for image in images])
            per_fold.append({'fold': fold, 'source': source, 'n': int(len(y)),
                             'auc_fusion': float(roc_auc_score(y, p_fus)),
                             'auc_specialist': float(roc_auc_score(y, p_spec))})
            bucket = pooled[source]
            bucket['y'].append(y), bucket['fusion'].append(p_fus), bucket['specialist'].append(p_spec)
            bucket['group'].append(np.array([f'{fold}:{Path(image).parent if args.group_by_parent else image}'
                                             for image in images]))

    summary = {}
    for source, bucket in pooled.items():
        if not bucket['y']:
            continue
        y, p_fus, p_spec, groups = (np.concatenate(bucket[key]) for key in ('y', 'fusion', 'specialist', 'group'))
        folds = [r for r in per_fold if r['source'] == source]
        summary[source] = {
            'n': int(len(y)),
            'auc_fusion_mean_over_folds': float(np.mean([r['auc_fusion'] for r in folds])),
            'auc_specialist_mean_over_folds': float(np.mean([r['auc_specialist'] for r in folds])),
            'delta_pooled': float(roc_auc_score(y, p_fus) - roc_auc_score(y, p_spec)),
            'delta_ci95': paired_bootstrap(y, p_fus, p_spec, groups),
            'delong_p': delong_paired(y, p_fus, p_spec),
        }
        s = summary[source]
        print(f"{source:12s} n={s['n']:5d} fusion={s['auc_fusion_mean_over_folds']:.3f} "
              f"specialist={s['auc_specialist_mean_over_folds']:.3f} delta={s['delta_pooled']:+.3f} "
              f"CI[{s['delta_ci95'][0]:+.3f},{s['delta_ci95'][1]:+.3f}] p={s['delong_p']:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'per_fold': per_fold, 'summary': summary}, indent=2))
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
