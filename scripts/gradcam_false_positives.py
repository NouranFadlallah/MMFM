"""Grad-CAM restricted to one error type -- false positives (true label
benign, predicted malignant; the default) or false negatives (true label
malignant, predicted benign; --error fn) -- across every trained model in
runs/ plus the root-level cdd_cesm/cmmd/bcsdbt checkpoints, to look for a
shared visual pattern in what each model gets wrong.

Unlike scripts/gradcam_all_failures.py (which caps at 3 mixed FP/FN errors
per fold), this saves every error of the chosen type found.

Usage: .venv/bin/python3 scripts/gradcam_false_positives.py [--error fn] [--export-figures]
Writes docs/gradcam_<error>/<dataset>_<backbone>/fold{N}_*.png (or
single-split equivalents for breamdm/cdd_cesm/cmmd/bcsdbt) and a JSON
manifest at docs/gradcam_<error>_manifest.json. --export-figures also writes
the most confident error per model as an (original, overlay) pair to
docs/latex/gradcam/<dataset>_<backbone>_<error>{_original,}.png.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
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
FIGURE_DIR = Path('docs/latex/gradcam')

# Set from the CLI in __main__: which error type the sweep collects.
ERROR = 'fp'
OUT_ROOT = Path('docs/gradcam_fp')
# key -> (confidence in the wrong class, display image [C,H,W], overlay) of
# the most confident error seen so far, for --export-figures.
_most_confident = {}


def _target_label():
    """Ground-truth label an error of the current type starts from."""
    return 0 if ERROR == 'fp' else 1


def _record(key, out_dir, fname, display, cam_map, probs):
    overlay = overlay_heatmap(display, cam_map)
    Image.fromarray(overlay).save(out_dir / fname)
    wrong_confidence = float(probs[1 - _target_label()])
    if key not in _most_confident or wrong_confidence > _most_confident[key][0]:
        _most_confident[key] = (wrong_confidence, np.asarray(display), overlay)

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


def _save_error(cam, tensor, row_image, true_label, out_dir, tag, key):
    x1 = tensor.unsqueeze(0).to(DEVICE).requires_grad_(False)
    cam_map, pred_class, probs = cam(x1)
    cam_map, pred_class, probs = cam_map[0], int(pred_class[0]), probs[0]
    if not (true_label == _target_label() and pred_class == 1 - true_label):
        return None  # correct, or the other error type -- not what this sweep collects

    fname = f'{tag}_{ERROR.upper()}_p{probs[1]:.2f}_{Path(str(row_image)).stem}.png'
    _record(key, out_dir, fname, tensor.numpy(), cam_map, probs)
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
            if int(row['label']) != _target_label():
                continue
            tensor = _load_one_image(row, transform, dataset)
            result = _save_error(cam, tensor, row['image'], _target_label(), out_dir, tag=f'fold{fold}', key=key)
            if result is not None:
                found.append(result)
    print(f'{key}: {len(found)} {ERROR.upper()} -> {out_dir}')
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
        if int(label) != _target_label():
            continue
        image_path, _ = dataset.samples[idx]
        x1_batched = x1.unsqueeze(0).to(DEVICE)
        cam_map, pred_class, probs = cam(x1_batched)
        cam_map, pred_class, probs = cam_map[0], int(pred_class[0]), probs[0]
        if pred_class != 1 - _target_label():
            continue
        display = x1.numpy().mean(axis=0, keepdims=True)
        fname = f'{ERROR.upper()}_p{probs[1]:.2f}_{Path(str(image_path)).stem}.png'
        _record('breamdm_resnet18', out_dir, fname, display, cam_map, probs)
        found.append({'image': str(image_path), 'true_label': CLASS_NAMES[_target_label()],
                      'pred_label': CLASS_NAMES[pred_class], 'p_malignant': float(probs[1]), 'file': fname})
    print(f'breamdm_resnet18: {len(found)} {ERROR.upper()} -> {out_dir}')
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
        if int(row['label']) != _target_label():
            continue
        tensor = _load_one_image(row, transform, key)
        result = _save_error(cam, tensor, row['image'], _target_label(), out_dir, tag='val', key=key)
        if result is not None:
            found.append(result)
    print(f'{key}: {len(found)} {ERROR.upper()} -> {out_dir}')
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


def _export_figures():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for key, (confidence, display, overlay) in _most_confident.items():
        image = np.clip(display, 0.0, 1.0)
        image = np.repeat(image, 3, axis=0) if image.shape[0] == 1 else image
        Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8)).save(
            FIGURE_DIR / f'{key}_{ERROR}_original.png')
        Image.fromarray(overlay).save(FIGURE_DIR / f'{key}_{ERROR}.png')
        print(f'figure: {key} (wrong-class confidence {confidence:.2f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--error', choices=['fp', 'fn'], default='fp')
    parser.add_argument('--export-figures', action='store_true')
    args = parser.parse_args()
    ERROR = args.error
    OUT_ROOT = Path(f'docs/gradcam_{ERROR}')

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

    out_path = Path(f'docs/gradcam_{ERROR}_manifest.json')
    out_path.write_text(json.dumps(manifest, indent=2))
    if args.export_figures:
        _export_figures()
    total = sum(len(v) for v in manifest.values())
    print(f'\nsaved {total} {ERROR.upper()} heatmaps total across {len(manifest)} model configs')
    print(f'manifest: {out_path}')
