"""Grad-CAM on the FusionLateModel checkpoints -- the mixed-backbone study
(Baseline/A/B/C, docs/mixed_backbone_results.md, Section "Mixed-Backbone
Multimodal Fusion" of the LaTeX report) and the current combined run (E,
+I-SPY2+BCS-DBT, Section "Combined Fusion with I-SPY2 MRI and BCS-DBT").

utils/gradcam.py's GradCAM.__call__ already accepts (x1, x2, x3,
presence_mask) matching FusionLateModel.forward exactly, but no script has
ever exercised it on a fusion checkpoint -- every prior Grad-CAM sweep in
this repo only covers SingleBackboneClassifier models. This closes that gap.

Each combined-dataset sample has exactly one real branch
(SingleModalityBranchDataset zero-fills the other two and marks them absent
in presence_mask), but a fused forward pass still runs all three backbones
regardless, so FusionBranchGradCAM hooks all three branches' last conv layer
at once and picks out whichever branch a given sample actually populates.

Two different validation-set compositions are needed: Baseline/A/B/C share
the original 427-sample set (mini-MIAS + BUS-BRA/BUSI/BUSC/BrEaST + BreastDM,
no I-SPY2/BCS-DBT, since those didn't exist yet when those checkpoints were
trained); Run E uses the current, larger set that training/train.py's own
_build_combined_sources now produces. Rebuilding the old composition by hand
(_build_old_combined_val_sources) avoids the gated-on-existence I-SPY2/BCS-DBT
blocks in the current code, which would otherwise silently evaluate those
older checkpoints against a validation set they were never trained for.

Usage: .venv/bin/python3 scripts/gradcam_fusion_failures.py
Writes docs/gradcam_fusion/<run>/<source>/*.png and a JSON manifest at
docs/gradcam_fusion_manifest.json.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import BreaDMDataset, BreadmTransform, MiasTransform  # noqa: E402
from models.fusion_model import FusionLateModel  # noqa: E402
from training.train import (  # noqa: E402
    BRANCH_CHANNELS, MAMMOGRAPHY_BRANCH, MRI_BRANCH, ULTRASOUND_BRANCH,
    SingleModalityBranchDataset, _build_combined_sources, _make_dataset,
    _mias_frame, _ultrasound_frame,
)
from utils.gradcam import overlay_heatmap, target_layer_for_raw_backbone  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
VALIDATION_FRACTION = 0.1
MAX_ERRORS_PER_SOURCE = 5
CLASS_NAMES = ('benign', 'malignant')
BRANCH_NAMES = ('mammography', 'ultrasound', 'mri')
OUT_ROOT = Path('docs/gradcam_fusion')

ARGS = type('Args', (), {
    'seed': SEED, 'validation_fraction': VALIDATION_FRACTION, 'image_size': 224,
    'ultrasound_datasets': ['busbra', 'busi', 'busc', 'breast'], 'max_samples': None,
})()

RUNS = [
    # (run_name, checkpoint, (mammography_backbone, ultrasound_backbone, mri_backbone), val_set)
    ('baseline', Path('runs/run-1/combined_fusion_resnet18_best.pth'), ('resnet18', 'resnet18', 'resnet18'), 'old'),
    ('run_a_mixed', Path('runs/run-3/combined_fusion_mixed_best.pth'), ('resnet18', 'resnet50', 'resnet18'), 'old'),
    ('run_b_resnet18_control', Path('runs/run-3/combined_fusion_resnet18_control_best.pth'), ('resnet18', 'resnet18', 'resnet18'), 'old'),
    ('run_c_radimagenet', Path('runs/run-3/combined_fusion_radimagenet_best.pth'), ('resnet18', 'resnet50', 'resnet18'), 'old'),
    ('run_e_ispy2_bcsdbt', Path('combined_fusion_resnet18_best.pth'), ('resnet18', 'resnet18', 'resnet18'), 'new'),
]


class FusionBranchGradCAM:
    """Grad-CAM for a 3-branch FusionLateModel: hooks all three branches'
    last conv layer simultaneously (a fused forward always runs all three
    backbones, even on a zero-filled absent branch), then computes the
    standard Grad-CAM weighted-channel-sum for whichever branch_index the
    caller asks for after one shared forward+backward."""

    def __init__(self, model, target_layers):
        self.model = model
        self.activations = [None, None, None]
        self.gradients = [None, None, None]
        for i, layer in enumerate(target_layers):
            layer.register_forward_hook(self._activation_hook(i))
            layer.register_full_backward_hook(self._gradient_hook(i))

    def _activation_hook(self, i):
        def hook(module, inputs, output):
            self.activations[i] = output
        return hook

    def _gradient_hook(self, i):
        def hook(module, grad_input, grad_output):
            self.gradients[i] = grad_output[0]
        return hook

    def __call__(self, x1, x2, x3, presence_mask, branch_index, target_class=None):
        self.model.zero_grad(set_to_none=True)
        logits, _, _ = self.model(x1, x2, x3, presence_mask)
        probs = torch.softmax(logits, dim=1).detach()
        if target_class is None:
            target_class = logits.argmax(dim=1)
        score = logits.gather(1, target_class.view(-1, 1)).squeeze(1)
        score.sum().backward()

        activation = self.activations[branch_index]
        gradient = self.gradients[branch_index]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1))
        image_hw = (x1, x2, x3)[branch_index].shape[-2:]
        cam = F.interpolate(cam.unsqueeze(1), size=image_hw, mode='bilinear', align_corners=False).squeeze(1)
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach().cpu().numpy(), target_class.detach().cpu().numpy(), probs.cpu().numpy()


def _build_old_combined_val_sources():
    """The pre-I-SPY2/BCS-DBT combined validation composition: mini-MIAS,
    BUS-BRA+BUSI+BUSC+BrEaST, BreastDM -- 10/300/117 samples, matching
    docs/mixed_backbone_results.md exactly. Returns a list of
    (source_name, branch_index, val_dataset)."""
    sources = []

    mias_root = Path('datasets/mammography/mias/all-mias')
    train_frame, validation_frame = _mias_frame(mias_root, ARGS.seed, ARGS.validation_fraction)
    from data.dataset import MiasTransform as _MiasTransform
    validation_transform = _MiasTransform(size=ARGS.image_size, augment=False)
    val_ds = _make_dataset(validation_frame, validation_transform)
    sources.append(('mias', MAMMOGRAPHY_BRANCH, SingleModalityBranchDataset(val_ds, MAMMOGRAPHY_BRANCH, BRANCH_CHANNELS)))

    for name in ARGS.ultrasound_datasets:
        _, validation_frame, _, validation_transform = _ultrasound_frame(name, ARGS)
        val_ds = _make_dataset(validation_frame, validation_transform)
        sources.append((name, ULTRASOUND_BRANCH, SingleModalityBranchDataset(val_ds, ULTRASOUND_BRANCH, BRANCH_CHANNELS)))

    breamdm_root = Path('datasets/MRI/BreaDM/cls/img9Se')
    validation_transform = BreadmTransform(size=ARGS.image_size, augment=False)
    val_ds = BreaDMDataset(breamdm_root, 'val', validation_transform)
    sources.append(('breamdm', MRI_BRANCH, SingleModalityBranchDataset(val_ds, MRI_BRANCH, BRANCH_CHANNELS)))

    return sources


def _build_new_combined_val_sources():
    """The current combined validation composition (training/train.py's own
    _build_combined_sources, which now also includes BCS-DBT and I-SPY2 since
    those directories exist locally). Returns a list of
    (source_name, branch_index, val_dataset), in the same fixed append order
    _build_combined_sources itself uses."""
    sources_by_branch = _build_combined_sources(ARGS)
    names_by_branch = {
        MAMMOGRAPHY_BRANCH: ['mias', 'bcsdbt'],
        ULTRASOUND_BRANCH: list(ARGS.ultrasound_datasets),
        MRI_BRANCH: ['breamdm', 'ispy2'],
    }
    sources = []
    for branch_index, entries in sources_by_branch.items():
        names = names_by_branch[branch_index]
        for name, entry in zip(names, entries):
            sources.append((name, branch_index, entry['val']))
    return sources


def _load_fusion_model(checkpoint_path, backbone_names):
    model = FusionLateModel(
        backbone_names=backbone_names, pretrained=False, input_channels=BRANCH_CHANNELS,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def _display_array(real_image, branch_index):
    """real_image must be the (x1, x2, x3)[branch_index] tensor -- the other
    two are zero-filled by SingleModalityBranchDataset and would display as
    solid black."""
    array = real_image.numpy()
    if branch_index == MRI_BRANCH:
        return array.mean(axis=0, keepdims=True)  # 9 DCE channels -> 1 grayscale
    return array


def run_one_checkpoint(run_name, checkpoint, backbone_names, val_set):
    if not checkpoint.exists():
        print(f'{run_name}: checkpoint {checkpoint} not found, skipping')
        return {}
    print(f'\n=== {run_name} ({checkpoint}) ===')
    model = _load_fusion_model(checkpoint, backbone_names)
    target_layers = [
        target_layer_for_raw_backbone(model.backbones[i], backbone_names[i]) for i in range(3)
    ]
    cam = FusionBranchGradCAM(model, target_layers)

    sources = _build_old_combined_val_sources() if val_set == 'old' else _build_new_combined_val_sources()

    run_manifest = {}
    for source_name, branch_index, dataset in sources:
        out_dir = OUT_ROOT / run_name / source_name
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for idx in range(len(dataset)):
            x1, x2, x3, true_label, presence = dataset[idx]
            batch = (x1.unsqueeze(0).to(DEVICE), x2.unsqueeze(0).to(DEVICE), x3.unsqueeze(0).to(DEVICE), presence.unsqueeze(0).to(DEVICE))
            cam_map, pred_class, probs = cam(*batch, branch_index=branch_index)
            cam_map, pred_class, probs = cam_map[0], int(pred_class[0]), probs[0]
            if pred_class == true_label:
                continue

            real_image = (x1, x2, x3)[branch_index]
            display = _display_array(real_image, branch_index)
            overlay = overlay_heatmap(display, cam_map)
            fname = (f'{BRANCH_NAMES[branch_index]}_{CLASS_NAMES[true_label]}_'
                     f'pred-{CLASS_NAMES[pred_class]}_p{probs[1]:.2f}_idx{idx}.png')
            Image.fromarray(overlay).save(out_dir / fname)
            saved.append({'source': source_name, 'branch': BRANCH_NAMES[branch_index],
                          'true_label': CLASS_NAMES[true_label], 'pred_label': CLASS_NAMES[pred_class],
                          'p_malignant': float(probs[1]), 'file': str(out_dir / fname)})
            if len(saved) >= MAX_ERRORS_PER_SOURCE:
                break
        print(f'  {source_name} ({BRANCH_NAMES[branch_index]}): {len(saved)} wrong cases -> {out_dir}')
        run_manifest[source_name] = saved
    return run_manifest


if __name__ == '__main__':
    manifest = {}
    for run_name, checkpoint, backbone_names, val_set in RUNS:
        manifest[run_name] = run_one_checkpoint(run_name, checkpoint, backbone_names, val_set)

    out_path = Path('docs/gradcam_fusion_manifest.json')
    out_path.write_text(json.dumps(manifest, indent=2))
    total = sum(len(v) for run in manifest.values() for v in run.values())
    print(f'\nsaved {total} wrong-case heatmaps total across {len(RUNS)} fusion checkpoints')
    print(f'manifest: {out_path}')
