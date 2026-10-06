"""Grad-CAM restricted to false positives (true label benign, predicted
malignant) across every trained model in runs/ plus the root-level cdd_cesm
and cmmd checkpoints -- to look for a shared visual pattern in what each
model mistakes for malignancy.

Unlike scripts/gradcam_all_failures.py (which caps at 3 mixed FP/FN errors
per fold), this saves every false positive found, since false positives are
the class of error this sweep exists to inspect.

Usage: .venv/bin/python3 scripts/gradcam_false_positives.py
Writes docs/gradcam_fp/<dataset>_<backbone>/fold{N}_*.png (or single-split
equivalents for breamdm/cdd_cesm/cmmd) and a JSON manifest at
docs/gradcam_fp_manifest.json.
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import BreaDMDataset, BreadmTransform, MiasTransform
from scripts.gradcam_visualize import _load_model, _load_one_image, _test_frame, CLASS_NAMES
from training.train import _cdd_cesm_frame, _cmmd_frame, _dbt_frame
from utils.gradcam import GradCAM, overlay_heatmap, target_layer_for_backbone

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
NUM_FOLDS = 5
RUN1 = Path('runs/run-1')
RUN2 = Path('runs/run-2')
RUN2_EXTRA = Path('runs/run-2/extra-backbones')
OUT_ROOT = Path('docs/gradcam_fp')

# (dataset, backbone, checkpoint_for_fold, num_folds, contrast_stretch)
KFOLD_CONFIGS = [
    ('busbra', 'resnet18', lambda f: (RUN1 if f == 1 else RUN2) / f'busbra_single_resnet18_best_fold{f}.pth', NUM_FOLDS, True),
    ('busbra', 'resnet50', lambda f: RUN2_EXTRA / f'busbra_single_resnet50_best_fold{f}.pth', NUM_FOLDS, True),
    ('busi', 'resnet18', lambda f: RUN2 / f'busi_single_resnet18_best_fold{f}.pth', NUM_FOLDS, True),
    ('busi', 'efficientnet_b0', lambda f: RUN2_EXTRA / f'busi_single_efficientnet_b0_best_fold{f}.pth', NUM_FOLDS, True),
    ('busc', 'resnet18', lambda f: RUN2 / f'busc_single_resnet18_best_fold{f}.pth', NUM_FOLDS, False),
    ('breast', 'resnet18', lambda f: RUN2 / f'breast_single_resnet18_best_fold{f}.pth', NUM_FOLDS, True),
    ('breast', 'efficientnet_b0', lambda f: RUN2_EXTRA / f'breast_single_efficientnet_b0_best_fold{f}.pth', NUM_FOLDS, True),
    ('mias', 'resnet18', lambda f: RUN2 / f'mias_single_resnet18_best_fold{f}.pth', NUM_FOLDS, False),
]


def _save_fp(cam, tensor, row_image, true_label, out_dir, tag):
    x1 = tensor.unsqueeze(0).to(DEVICE).requires_grad_(False)
    cam_map, pred_class, probs = cam(x1)
    cam_map, pred_class, probs = cam_map[0], int(pred_class[0]), probs[0]
    if not (true_label == 0 and pred_class == 1):
        return None  # correct, or a false negative -- not what this sweep collects

    overlay = overlay_heatmap(tensor.numpy(), cam_map)
    fname = f'{tag}_FP_p{probs[1]:.2f}_{Path(str(row_image)).stem}.png'
    Image.fromarray(overlay).save(out_dir / fname)
    return {'image': str(row_image), 'true_label': CLASS_NAMES[true_label],
            'pred_label': CLASS_NAMES[pred_class], 'p_malignant': float(probs[1]), 'file': fname}


def kfold_false_positives(dataset, backbone, ckpt_for_fold, num_folds, contrast_stretch):
    key = f'{dataset}_{backbone}'
    out_dir = OUT_ROOT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    found = []
    model_cache = {}
    for fold in range(1, num_folds + 1):
        ckpt = ckpt_for_fold(fold)
        if ckpt not in model_cache:
            model_cache[ckpt] = _load_model(ckpt, backbone)
        model = model_cache[ckpt]
        target_layer = target_layer_for_backbone(model, backbone)
        cam = GradCAM(model, target_layer)

        frame, transform = _test_frame(dataset, fold, num_folds, contrast_stretch)
        for row in frame.to_dict('records'):
            if int(row['label']) != 0:
                continue  # false positives only start from a benign ground truth
            tensor = _load_one_image(row, transform, dataset)
            result = _save_fp(cam, tensor, row['image'], 0, out_dir, tag=f'fold{fold}')
            if result is not None:
                found.append(result)
    print(f'{key}: {len(found)} false positives -> {out_dir}')
    return found


def breamdm_false_positives():
    out_dir = OUT_ROOT / 'breamdm_resnet18'
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path('datasets/MRI/BreaDM/cls/img9Se')
    transform = BreadmTransform(size=224, augment=False)
    dataset = BreaDMDataset(root, 'val', transform)
    model = _load_model(RUN1 / 'breamdm_single_resnet18_best.pth', 'resnet18', input_channels=9)
    target_layer = target_layer_for_backbone(model, 'resnet18')
    cam = GradCAM(model, target_layer)

    found = []
    for idx in range(len(dataset)):
        x1, _, _, label, _ = dataset[idx]
        if int(label) != 0:
            continue
        image_path, _ = dataset.samples[idx]
        x1_batched = x1.unsqueeze(0).to(DEVICE)
        cam_map, pred_class, probs = cam(x1_batched)
        cam_map, pred_class, probs = cam_map[0], int(pred_class[0]), probs[0]
        if pred_class != 1:
            continue
        display = x1.numpy().mean(axis=0, keepdims=True)
        overlay = overlay_heatmap(display, cam_map)
        fname = f'FP_p{probs[1]:.2f}_{Path(str(image_path)).stem}.png'
        Image.fromarray(overlay).save(out_dir / fname)
        found.append({'image': str(image_path), 'true_label': 'benign', 'pred_label': 'malignant',
                      'p_malignant': float(probs[1]), 'file': fname})
    print(f'breamdm_resnet18: {len(found)} false positives -> {out_dir}')
    return found


def _single_split_false_positives(key, checkpoint, validation_frame):
    out_dir = OUT_ROOT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(checkpoint, 'resnet18')
    target_layer = target_layer_for_backbone(model, 'resnet18')
    cam = GradCAM(model, target_layer)
    transform = MiasTransform(size=224, augment=False)

    found = []
    for row in validation_frame.to_dict('records'):
        if int(row['label']) != 0:
            continue
        tensor = _load_one_image(row, transform, key)
        result = _save_fp(cam, tensor, row['image'], 0, out_dir, tag='val')
        if result is not None:
            found.append(result)
    print(f'{key}: {len(found)} false positives -> {out_dir}')
    return found


def cdd_cesm_false_positives():
    root = Path('datasets/mammography/cdd_cesm')
    annotation_path = Path('datasets/mammography/cdd_cesm_annotations.xlsx')
    _, validation_frame = _cdd_cesm_frame(root, annotation_path, SEED, 0.1)
    return _single_split_false_positives('cdd_cesm_resnet18', Path('cdd_cesm_single_resnet18_best.pth'), validation_frame)


def cmmd_false_positives():
    dicom_root = Path('datasets/mammography/cmmd_dicom')
    clinical_data_path = Path('datasets/mammography/cmmd_clinicaldata.xlsx')
    cache_dir = Path('datasets/mammography/cmmd_png')
    _, validation_frame = _cmmd_frame(dicom_root, clinical_data_path, cache_dir, SEED, 0.1)
    return _single_split_false_positives('cmmd_resnet18', Path('cmmd_single_resnet18_best.pth'), validation_frame)


def bcsdbt_false_positives():
    _, validation_frame = _dbt_frame(
        Path('datasets/mammography/bcsdbt_dicom'),
        Path('datasets/mammography/bcsdbt_labels.csv'),
        Path('datasets/mammography/bcsdbt_file_paths.csv'),
        Path('datasets/mammography/bcsdbt_boxes.csv'),
        Path('datasets/mammography/bcsdbt_png'),
        SEED, 0.2,
    )
    return _single_split_false_positives('bcsdbt_resnet18', Path('bcsdbt_single_resnet18_best.pth'), validation_frame)


if __name__ == '__main__':
    manifest = {}
    for dataset, backbone, ckpt_for_fold, num_folds, contrast_stretch in KFOLD_CONFIGS:
        manifest[f'{dataset}_{backbone}'] = kfold_false_positives(
            dataset, backbone, ckpt_for_fold, num_folds, contrast_stretch
        )

    manifest['breamdm_resnet18'] = breamdm_false_positives()

    if Path('cdd_cesm_single_resnet18_best.pth').exists():
        manifest['cdd_cesm_resnet18'] = cdd_cesm_false_positives()
    if Path('cmmd_single_resnet18_best.pth').exists():
        manifest['cmmd_resnet18'] = cmmd_false_positives()
    if Path('bcsdbt_single_resnet18_best.pth').exists():
        manifest['bcsdbt_resnet18'] = bcsdbt_false_positives()

    out_path = Path('docs/gradcam_fp_manifest.json')
    out_path.write_text(json.dumps(manifest, indent=2))
    total = sum(len(v) for v in manifest.values())
    print(f'\nsaved {total} false-positive heatmaps total across {len(manifest)} model configs')
    print(f'manifest: {out_path}')
