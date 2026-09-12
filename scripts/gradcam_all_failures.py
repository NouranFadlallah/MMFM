"""Grad-CAM on multiple misclassified examples from every trained model in
runs/ (all backbones, all cross-validation folds) -- no correctly classified
examples this time, just failures, to look for a shared visual pattern in
what each model gets wrong.

Caps at MAX_ERRORS_PER_FOLD saved images per fold (folds with fewer errors
than that just save all of them) to keep the total output bounded.

Usage: .venv/bin/python3 scripts/gradcam_all_failures.py
Writes docs/gradcam/<dataset>_<backbone>/fold{N}_*.png and a JSON manifest
at docs/gradcam_failures_manifest.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import BreaDMDataset, BreadmTransform
from scripts.gradcam_visualize import _load_model, visualize_fold
from utils.gradcam import GradCAM, overlay_heatmap, target_layer_for_backbone

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_ERRORS_PER_FOLD = 3
NUM_FOLDS = 5
RUN1 = Path('runs/run-1')
RUN2 = Path('runs/run-2')
RUN2_EXTRA = Path('runs/run-2/extra-backbones')

# (dataset, backbone, checkpoint_for_fold, num_folds, contrast_stretch)
CONFIGS = [
    ('busbra', 'resnet18', lambda f: (RUN1 if f == 1 else RUN2) / f'busbra_single_resnet18_best_fold{f}.pth', NUM_FOLDS, True),
    ('busbra', 'resnet50', lambda f: RUN2_EXTRA / f'busbra_single_resnet50_best_fold{f}.pth', NUM_FOLDS, True),
    ('busi', 'resnet18', lambda f: RUN2 / f'busi_single_resnet18_best_fold{f}.pth', NUM_FOLDS, True),
    ('busi', 'efficientnet_b0', lambda f: RUN2_EXTRA / f'busi_single_efficientnet_b0_best_fold{f}.pth', NUM_FOLDS, True),
    ('busc', 'resnet18', lambda f: RUN2 / f'busc_single_resnet18_best_fold{f}.pth', NUM_FOLDS, False),
    ('breast', 'resnet18', lambda f: RUN2 / f'breast_single_resnet18_best_fold{f}.pth', NUM_FOLDS, True),
    ('breast', 'efficientnet_b0', lambda f: RUN2_EXTRA / f'breast_single_efficientnet_b0_best_fold{f}.pth', NUM_FOLDS, True),
    ('mias', 'resnet18', lambda f: RUN2 / f'mias_single_resnet18_best_fold{f}.pth', NUM_FOLDS, False),
]


def breamdm_failures(max_errors=MAX_ERRORS_PER_FOLD):
    """BreastDM uses BreaDMDataset/9-channel input, not the row-based k-fold
    path the other datasets share, so it needs its own loop. For display,
    the 9 DCE slice channels are averaged into one grayscale image."""
    class_names = ('benign', 'malignant')
    out_dir = Path('docs/gradcam/breamdm_resnet18')
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path('datasets/MRI/BreaDM/cls/img9Se')
    transform = BreadmTransform(size=224, augment=False)
    dataset = BreaDMDataset(root, 'val', transform)
    model = _load_model(RUN1 / 'breamdm_single_resnet18_best.pth', 'resnet18', input_channels=9)
    target_layer = target_layer_for_backbone(model, 'resnet18')
    cam = GradCAM(model, target_layer)

    errors = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            x1, _, _, label, _ = dataset[idx]
            logits, _, _ = model(x1.unsqueeze(0).to(DEVICE))
            pred = int(logits.argmax(dim=1).item())
            if pred != label:
                errors.append(idx)
    errors = errors[:max_errors]

    print(f'breamdm/resnet18: {len(errors)} misclassified examples visualized -> {out_dir}')
    saved = []
    for idx in errors:
        x1, _, _, true_label, _ = dataset[idx]
        x1_batched = x1.unsqueeze(0).to(DEVICE)
        cam_map, pred_class, probs = cam(x1_batched)
        cam_map, pred_class, probs = cam_map[0], int(pred_class[0]), probs[0]

        display = x1.numpy().mean(axis=0, keepdims=True)  # [1,H,W] average over 9 slices
        overlay = overlay_heatmap(display, cam_map)
        outcome = 'correct' if pred_class == true_label else 'WRONG'
        image_path, _ = dataset.samples[idx]
        fname = f'{class_names[true_label]}_pred-{class_names[pred_class]}_{outcome}_p{probs[1]:.2f}_{Path(str(image_path)).stem}.png'
        Image.fromarray(overlay).save(out_dir / fname)
        saved.append({'image': str(image_path), 'true_label': class_names[true_label],
                       'pred_label': class_names[pred_class], 'p_malignant': float(probs[1])})
        print(f'  {fname}')
    print(f'saved {len(saved)} images to {out_dir}')
    return saved


def _summarize(row, pred_class, true_label, probs):
    class_names = ('benign', 'malignant')
    return {
        'image': str(row['image']),
        'true_label': class_names[true_label],
        'pred_label': class_names[pred_class],
        'p_malignant': float(probs[1]),
    }


if __name__ == '__main__':
    manifest = {}
    for dataset, backbone, ckpt_for_fold, num_folds, contrast_stretch in CONFIGS:
        key = f'{dataset}_{backbone}'
        manifest[key] = []
        model_cache = {}
        for fold in range(1, num_folds + 1):
            ckpt = ckpt_for_fold(fold)
            if ckpt not in model_cache:
                model_cache[ckpt] = _load_model(ckpt, backbone)
            saved = visualize_fold(
                dataset, backbone, ckpt, fold, num_folds, contrast_stretch,
                out_dir=Path('docs/gradcam') / key,
                num_per_class_correct=0, max_errors=MAX_ERRORS_PER_FOLD,
                model=model_cache[ckpt],
            )
            manifest[key].extend(_summarize(*s) for s in saved)

    manifest['breamdm_resnet18'] = breamdm_failures()

    out_path = Path('docs/gradcam_failures_manifest.json')
    out_path.write_text(json.dumps(manifest, indent=2))
    total = sum(len(v) for v in manifest.values())
    print(f'\nsaved {total} failure heatmaps total across {len(CONFIGS)} model configs')
    print(f'manifest: {out_path}')
