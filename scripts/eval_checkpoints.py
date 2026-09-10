"""Re-run the full metric set (accuracy/sensitivity/specificity/precision/F1/AUC)
against each dataset's already-trained single-mode ResNet-18 checkpoint, without
retraining. Reconstructs each validation split with the same defaults the original
training runs used (seed=42, validation_fraction=0.1, per-dataset image size).
"""
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import cfg
from data.dataset import BreaDMDataset, BreadmTransform, BusbraTransform, MiasTransform
from training.train import (
    SingleBackboneClassifier, _breast_frame, _busc_frame, _busi_frame,
    _make_dataset, _mias_frame, evaluate_full_metrics,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
VALIDATION_FRACTION = 0.1  # matches the --validation-fraction argparse default used in every run


def _load_model(checkpoint_path, input_channels=3):
    model = SingleBackboneClassifier(
        backbone_name='resnet18', pretrained=False, embedding_dim=cfg.embedding_dim,
        num_classes=cfg.num_classes, input_channels=input_channels,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    return model


def _eval_frame_dataset(validation_frame, transform, checkpoint_path, batch_size=16):
    dataset = _make_dataset(validation_frame, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = _load_model(checkpoint_path)
    return evaluate_full_metrics(model, loader, DEVICE), len(dataset)


def eval_mias():
    root = Path('datasets/mammography/mias/all-mias')
    _, validation_frame = _mias_frame(root, SEED, VALIDATION_FRACTION)
    transform = MiasTransform(size=224, augment=False)
    return _eval_frame_dataset(validation_frame, transform, 'mias_single_resnet18_best.pth')


def eval_busi():
    root = Path('datasets/ultrasound/BUS/BUS')
    _, validation_frame = _busi_frame(root, SEED, VALIDATION_FRACTION)
    transform = BusbraTransform(size=224, augment=False)
    return _eval_frame_dataset(validation_frame, transform, 'busi_single_resnet18_best.pth')


def eval_busc():
    root = Path('datasets/ultrasound/us-dataset')
    _, validation_frame = _busc_frame(root, SEED, VALIDATION_FRACTION)
    transform = MiasTransform(size=128, augment=False)
    return _eval_frame_dataset(validation_frame, transform, 'busc_single_resnet18_best.pth')


def eval_breast():
    root = Path('datasets/ultrasound/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023')
    _, validation_frame = _breast_frame(root, SEED, VALIDATION_FRACTION)
    transform = BusbraTransform(size=224, augment=False)
    return _eval_frame_dataset(validation_frame, transform, 'breast_single_resnet18_best.pth')


def eval_breamdm():
    root = Path('datasets/MRI/BreaDM/cls/img9Se')
    transform = BreadmTransform(size=224, augment=False)
    dataset = BreaDMDataset(root, 'val', transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = _load_model('breamdm_single_resnet18_best.pth', input_channels=9)
    return evaluate_full_metrics(model, loader, DEVICE), len(dataset)


if __name__ == '__main__':
    results = {}
    for name, fn in (
        ('mias', eval_mias), ('busi', eval_busi), ('busc', eval_busc),
        ('breast', eval_breast), ('breamdm', eval_breamdm),
    ):
        metrics, n = fn()
        results[name] = {'metrics': metrics, 'validation_samples': n}
        print(f'{name} (n={n}): ' + ' '.join(f'{k}={v:.4f}' for k, v in metrics.items()))

    out_path = Path('checkpoint_eval_results.json')
    out_path.write_text(json.dumps(results, indent=2))
    print(f'saved {out_path}')
