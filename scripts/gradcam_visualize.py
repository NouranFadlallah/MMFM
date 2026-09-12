"""Grad-CAM visualization for a trained single-modality checkpoint: saves
heatmap-overlay PNGs showing what the model actually looked at for a
handful of benign/malignant examples (correct and incorrect predictions),
using the same test-fold reconstruction as scripts/per_class_analysis.py.

Default target is BUSC fold 1, the case docs/latex/main.tex's preprocessing
discussion flagged for a saliency check (least preprocessing of any dataset
here, and the best result -- worth checking whether the model is keying on
the lesion or on background/acquisition artifacts).

Usage:
  .venv/bin/python3 scripts/gradcam_visualize.py
  .venv/bin/python3 scripts/gradcam_visualize.py --dataset busi --backbone efficientnet_b0 \
      --checkpoint runs/run-2/extra-backbones/busi_single_efficientnet_b0_best_fold1.pth --fold 1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import cfg
from data.dataset import BreaDMDataset, BreadmTransform, BusbraTransform, MiasTransform
from training.train import (
    DATASET_KFOLD_SPECS, SingleBackboneClassifier, _busbra_kfold_frame, _group_kfold_frame,
)
from utils.gradcam import GradCAM, overlay_heatmap, target_layer_for_backbone

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42


def _load_model(checkpoint_path, backbone, input_channels=3):
    model = SingleBackboneClassifier(
        backbone_name=backbone, pretrained=False, embedding_dim=cfg.embedding_dim,
        num_classes=cfg.num_classes, input_channels=input_channels,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def _test_frame(dataset, fold, num_folds, contrast_stretch):
    if dataset == 'busbra':
        root = Path('datasets/ultrasound/busbra/BUSBRA')
        _, _, frame = _busbra_kfold_frame(root, num_folds, fold, SEED, holdout_fraction=0.2)
        image_size = 224
        transform = BusbraTransform(size=image_size, augment=False, contrast_stretch=contrast_stretch)
        return frame, transform
    spec = DATASET_KFOLD_SPECS[dataset]
    metadata = spec['metadata_fn'](spec['default_root'])
    _, _, frame = _group_kfold_frame(metadata, spec['group_col'], num_folds, fold, SEED, holdout_fraction=0.2)
    image_size = 128 if dataset == 'busc' else 224
    transform_cls = spec['transform_cls']
    kwargs = {'contrast_stretch': contrast_stretch} if transform_cls is BusbraTransform else {}
    transform = transform_cls(size=image_size, augment=False, **kwargs)
    return frame, transform


def _load_one_image(row, transform, dataset_name):
    image = Image.open(row['image'])
    if 'mask' in row and str(row.get('mask', '')).strip():
        mask_path = row['mask']
        mask = Image.open(mask_path) if Path(mask_path).exists() else None
        tensor = transform(image, mask) if mask is not None else transform(image)
    elif 'crop_box' in row and row.get('crop_box') is not None:
        tensor = transform(image, row['crop_box'])
    else:
        tensor = transform(image)
    return tensor


CLASS_NAMES = ('benign', 'malignant')


def visualize_fold(dataset, backbone, checkpoint, fold, num_folds=5, contrast_stretch=False,
                    out_dir=None, num_per_class_correct=4, max_errors=None, model=None):
    """Reconstruct one test fold, find misclassified examples, and save a
    Grad-CAM heatmap overlay for each (capped at max_errors if given), plus
    up to num_per_class_correct correctly classified examples per class.
    Returns the list of (row, pred_class, true_label, probs) actually saved.
    Pass a pre-loaded `model` to avoid reloading the checkpoint per call.
    """
    out_dir = out_dir or Path('docs/gradcam') / f'{dataset}_{backbone}_fold{fold}'
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, transform = _test_frame(dataset, fold, num_folds, contrast_stretch)
    if model is None:
        model = _load_model(checkpoint, backbone)
    target_layer = target_layer_for_backbone(model, backbone)
    cam = GradCAM(model, target_layer)

    all_rows = frame.to_dict('records')
    with torch.no_grad():
        preds = []
        for row in all_rows:
            tensor = _load_one_image(row, transform, dataset)
            logits, _, _ = model(tensor.unsqueeze(0).to(DEVICE))
            preds.append(int(logits.argmax(dim=1).item()))
    errors = [row for row, pred in zip(all_rows, preds) if pred != row['label']]
    correct = [row for row, pred in zip(all_rows, preds) if pred == row['label']]

    if max_errors is not None:
        errors = errors[:max_errors]
    picked = list(errors)
    for label in (0, 1):
        rows = [r for r in correct if r['label'] == label]
        picked.extend(rows[:num_per_class_correct])

    print(f'{dataset}/{backbone} fold {fold}: {len(errors)} misclassified (of '
          f'{sum(1 for p, r in zip(preds, all_rows) if p != r["label"])} total) visualized '
          f'of {len(all_rows)} test examples -> {out_dir}')
    saved = []
    for row in picked:
        tensor = _load_one_image(row, transform, dataset)
        x1 = tensor.unsqueeze(0).to(DEVICE).requires_grad_(False)
        cam_map, pred_class, probs = cam(x1)
        cam_map, pred_class, probs = cam_map[0], int(pred_class[0]), probs[0]

        true_label = int(row['label'])
        overlay = overlay_heatmap(tensor.numpy(), cam_map)
        outcome = 'correct' if pred_class == true_label else 'WRONG'
        fname = (
            f'fold{fold}_{CLASS_NAMES[true_label]}_pred-{CLASS_NAMES[pred_class]}_{outcome}_'
            f'p{probs[1]:.2f}_{Path(str(row["image"])).stem}.png'
        )
        Image.fromarray(overlay).save(out_dir / fname)
        saved.append((row, pred_class, true_label, probs))
        print(f'  {fname}')

    print(f'saved {len(saved)} images to {out_dir}')
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='busc', choices=list(DATASET_KFOLD_SPECS.keys()) + ['busbra'])
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/run-2/busc_single_resnet18_best_fold1.pth'))
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--contrast-stretch', action='store_true', default=False)
    parser.add_argument('--num-per-class', type=int, default=4)
    parser.add_argument('--out-dir', type=Path, default=None)
    args = parser.parse_args()

    visualize_fold(
        args.dataset, args.backbone, args.checkpoint, args.fold, args.num_folds,
        args.contrast_stretch, args.out_dir, num_per_class_correct=args.num_per_class,
    )


if __name__ == '__main__':
    main()
