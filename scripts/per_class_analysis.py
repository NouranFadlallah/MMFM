"""Per-class (benign/malignant) breakdown and false-positive/false-negative
bias detection for every already-trained checkpoint in runs/, with no
retraining. Reconstructs each fold's exact held-out test set (same seed,
same split helper used during training) and re-runs inference.

Usage: .venv/bin/python3 scripts/per_class_analysis.py
Writes docs/per_class_results.md and prints a summary table to stdout.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import cfg
from data.dataset import BreaDMDataset, BreadmTransform, BusbraTransform, MiasTransform
from training.train import (
    DATASET_KFOLD_SPECS, SingleBackboneClassifier, _busbra_kfold_frame,
    _group_kfold_frame, _make_dataset, collect_predictions,
)
from utils.metrics import classification_metrics, per_class_confusion

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
NUM_FOLDS = 5
RUN1 = Path('runs/run-1')
RUN2 = Path('runs/run-2')
RUN2_EXTRA = Path('runs/run-2/extra-backbones')


def _load_model(checkpoint_path, backbone, input_channels=3):
    model = SingleBackboneClassifier(
        backbone_name=backbone, pretrained=False, embedding_dim=cfg.embedding_dim,
        num_classes=cfg.num_classes, input_channels=input_channels,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    return model


def _eval_checkpoint(checkpoint_path, test_frame, transform, backbone, batch_size=32):
    dataset = _make_dataset(test_frame, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = _load_model(checkpoint_path, backbone)
    probs, targets = collect_predictions(model, loader, DEVICE)
    return classification_metrics(probs, targets), per_class_confusion(probs, targets), len(dataset)


def eval_busbra_folds(backbone, checkpoint_dir_for_fold, contrast_stretch=True):
    root = Path('datasets/ultrasound/busbra/BUSBRA')
    fold_results = []
    for fold in range(1, NUM_FOLDS + 1):
        _, _, test_frame = _busbra_kfold_frame(root, NUM_FOLDS, fold, SEED, holdout_fraction=0.2)
        transform = BusbraTransform(size=224, augment=False, contrast_stretch=contrast_stretch)
        ckpt = checkpoint_dir_for_fold(fold)
        metrics, per_class, n = _eval_checkpoint(ckpt, test_frame, transform, backbone)
        fold_results.append({'fold': fold, 'n': n, 'metrics': metrics, 'per_class': per_class})
    return fold_results


def eval_generic_kfold_folds(dataset_name, backbone, checkpoint_dir_for_fold, contrast_stretch=True):
    spec = DATASET_KFOLD_SPECS[dataset_name]
    metadata = spec['metadata_fn'](spec['default_root'])
    transform_cls = spec['transform_cls']
    supports_contrast = transform_cls is BusbraTransform
    image_size = 128 if dataset_name == 'busc' else 224
    fold_results = []
    for fold in range(1, NUM_FOLDS + 1):
        _, _, test_frame = _group_kfold_frame(
            metadata, spec['group_col'], NUM_FOLDS, fold, SEED, holdout_fraction=0.2
        )
        kwargs = {'contrast_stretch': contrast_stretch} if supports_contrast else {}
        transform = transform_cls(size=image_size, augment=False, **kwargs)
        ckpt = checkpoint_dir_for_fold(fold)
        metrics, per_class, n = _eval_checkpoint(ckpt, test_frame, transform, backbone)
        fold_results.append({'fold': fold, 'n': n, 'metrics': metrics, 'per_class': per_class})
    return fold_results


def eval_breamdm():
    root = Path('datasets/MRI/BreaDM/cls/img9Se')
    transform = BreadmTransform(size=224, augment=False)
    dataset = BreaDMDataset(root, 'val', transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = _load_model(RUN1 / 'breamdm_single_resnet18_best.pth', 'resnet18', input_channels=9)
    probs, targets = collect_predictions(model, loader, DEVICE)
    return classification_metrics(probs, targets), per_class_confusion(probs, targets), len(dataset)


def _aggregate(fold_results):
    """Mean +/- std across folds for classification_metrics, and pooled
    confusion counts (summed, not averaged) for the per-class breakdown."""
    metric_names = fold_results[0]['metrics'].keys()
    agg_metrics = {
        name: {
            'mean': float(np.mean([r['metrics'][name] for r in fold_results])),
            'std': float(np.std([r['metrics'][name] for r in fold_results])),
        }
        for name in metric_names
    }
    pooled_confusion = {k: sum(r['per_class']['confusion'][k] for r in fold_results) for k in ('tn', 'fp', 'fn', 'tp')}
    tn, fp, fn, tp = pooled_confusion['tn'], pooled_confusion['fp'], pooled_confusion['fn'], pooled_confusion['tp']
    fp_rate = fp / (fp + tn) if (fp + tn) else float('nan')
    fn_rate = fn / (fn + tp) if (fn + tp) else float('nan')
    if fp_rate == fn_rate:
        bias = 'balanced'
    elif fp_rate > fn_rate:
        bias = 'false_positive_leaning'
    else:
        bias = 'false_negative_leaning'
    return {
        'metrics': agg_metrics,
        'pooled_confusion': pooled_confusion,
        'pooled_false_positive_rate': fp_rate,
        'pooled_false_negative_rate': fn_rate,
        'bias': bias,
        'per_fold_bias': [r['per_class']['bias'] for r in fold_results],
    }


if __name__ == '__main__':
    results = {}

    print('=== BUS-BRA (ResNet-18) ===')
    fr = eval_busbra_folds('resnet18', lambda f: (RUN1 if f == 1 else RUN2) / f'busbra_single_resnet18_best_fold{f}.pth')
    results['busbra_resnet18'] = _aggregate(fr)

    print('=== BUS-BRA (ResNet-50) ===')
    fr = eval_busbra_folds('resnet50', lambda f: RUN2_EXTRA / f'busbra_single_resnet50_best_fold{f}.pth')
    results['busbra_resnet50'] = _aggregate(fr)

    print('=== BUSI (ResNet-18) ===')
    fr = eval_generic_kfold_folds('busi', 'resnet18', lambda f: RUN2 / f'busi_single_resnet18_best_fold{f}.pth')
    results['busi_resnet18'] = _aggregate(fr)

    print('=== BUSI (EfficientNet-B0) ===')
    fr = eval_generic_kfold_folds('busi', 'efficientnet_b0', lambda f: RUN2_EXTRA / f'busi_single_efficientnet_b0_best_fold{f}.pth')
    results['busi_efficientnet_b0'] = _aggregate(fr)

    print('=== BUSC (ResNet-18) ===')
    fr = eval_generic_kfold_folds('busc', 'resnet18', lambda f: RUN2 / f'busc_single_resnet18_best_fold{f}.pth', contrast_stretch=False)
    results['busc_resnet18'] = _aggregate(fr)

    print('=== BrEaST-Lesions USG (ResNet-18) ===')
    fr = eval_generic_kfold_folds('breast', 'resnet18', lambda f: RUN2 / f'breast_single_resnet18_best_fold{f}.pth')
    results['breast_resnet18'] = _aggregate(fr)

    print('=== BrEaST-Lesions USG (EfficientNet-B0) ===')
    fr = eval_generic_kfold_folds('breast', 'efficientnet_b0', lambda f: RUN2_EXTRA / f'breast_single_efficientnet_b0_best_fold{f}.pth')
    results['breast_efficientnet_b0'] = _aggregate(fr)

    print('=== mini-MIAS (ResNet-18) ===')
    fr = eval_generic_kfold_folds('mias', 'resnet18', lambda f: RUN2 / f'mias_single_resnet18_best_fold{f}.pth', contrast_stretch=False)
    results['mias_resnet18'] = _aggregate(fr)

    print('=== BreastDM (ResNet-18) ===')
    metrics, per_class, n = eval_breamdm()
    results['breamdm_resnet18'] = {
        'metrics': {k: {'mean': v, 'std': 0.0} for k, v in metrics.items()},
        'pooled_confusion': per_class['confusion'],
        'pooled_false_positive_rate': per_class['false_positive_rate'],
        'pooled_false_negative_rate': per_class['false_negative_rate'],
        'bias': per_class['bias'],
        'per_fold_bias': [per_class['bias']],
        'per_class_single_run': per_class['per_class'],
    }

    out_path = Path('docs/per_class_results_raw.json')
    out_path.write_text(json.dumps(results, indent=2))
    print(f'\nsaved {out_path}')

    print('\n=== SUMMARY: bias verdict per model ===')
    for name, r in results.items():
        print(f"{name}: {r['bias']} (pooled FPR={r['pooled_false_positive_rate']:.3f}, "
              f"FNR={r['pooled_false_negative_rate']:.3f}, confusion={r['pooled_confusion']})")
