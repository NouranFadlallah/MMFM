---
license: other
license_name: per-model
tags:
  - medical-imaging
  - breast-cancer
  - image-classification
  - pytorch
pipeline_tag: image-classification
---

# MMFM checkpoints (private, pre-publication)

Weights for the MMFM thesis project (modality-agnostic breast-cancer classification
over ultrasound, mammography/tomosynthesis and DCE-MRI). Code, training commands and
the full write-up live in the GitHub repository `NouranFadlallah/MMFM`; this repo holds
weights only. Benign vs. malignant throughout, except ISPY2 pCR (not included).

**Research use only. Not a medical device; not validated for clinical use.**

## Licences

Each checkpoint inherits the terms of the data it was trained on. Licences below are
those of the training datasets as recorded in the report's dataset table.

| Folder | Trained on | Data licence | Commercial use |
|---|---|---|---|
| `reproductions/bus_cot/` | BUS-CoT BUS-Lesion (includes BUSI copies, licence not stated) | CC BY 4.0 + BUSI not stated | unclear |
| `reproductions/busbra/` | BUS-BRA | CC BY 4.0 | yes, with attribution |
| `reproductions/cbis_ddsm/` | CBIS-DDSM | CC BY 3.0 | yes, with attribution |
| `reproductions/bcsdbt/` | BCS-DBT | CC BY-NC 4.0 | **no** |
| `reproductions/odelia_mst/` | ODELIA | CC BY-NC 4.0 | **no** |
| `multisource/` | 14 datasets (all except CDD-CESM and CMMD) incl. BCS-DBT and ODELIA (NC), BUSI/BUSC/BreastDM (not stated) | mixed, includes NC | **no** |
| `leakage_controlled/` | BUSI (not stated), BUSC (not stated) | not stated | unclear |
| `vlm/biomedclip/` | 10 sources incl. BCS-DBT (NC), BUSI/BUSC/BreastDM (not stated) | mixed, includes NC | **no** |
| `vlm/medgemma_lora/` | same manifest; adapter on `google/medgemma-4b-it` | mixed incl. NC, **plus Health AI Developer Foundations terms** | **no** |
| `preprocessing/` | BrEaST (CC BY 4.0) ↔ BUS-BRA (CC BY 4.0) | CC BY 4.0 | yes, with attribution |

## Files

| File | Model | Evaluation (from the report) | Caveats |
|---|---|---|---|
| `reproductions/bus_cot/bus_cot_resnet50_fold{1..5}.pth` | ResNet-50, Yu et al. 2025 recipe | test AUC 0.849 ± 0.005 on the official 825-image test split (paper 0.809) | BUS-Lesion trainval contains 824 BUS-BRA and 439 BUSI crops: **do not use as a warm-start or evaluate on BUSI/BUS-BRA** |
| `reproductions/busbra/busbra_resnet18_fold{1..5}.pth` | ResNet-18, `--paper-match`, official K5B folds | test AUC 0.886 / 0.896 / 0.926 / 0.923 / 0.886 | fold k never saw fold k's test images; used as fold-k ultrasound warm-start |
| `reproductions/cbis_ddsm/cbis_ddsm_resnet50_valacc.pth` | ResNet-50, Shen et al. patch-classifier recipe | ROI-crop test AUC 0.752 (250 crops), best-validation-accuracy selection | no comparable paper number for crops |
| `reproductions/bcsdbt/bcsdbt_full_resnet18.pth` | ResNet-18 on box-cropped slices, 396 views | validation AUC 0.725 on 42 views | too few validation views for a stable estimate |
| `reproductions/odelia_mst/odelia_mst_fold{0..4}.pth` | Medical Slice Transformer (DINOv3-S + transformer) | in-distribution macro AUC 79.0 ± 3.8 (paper 79.0) | Valencia (RSH) held out; centre shortcut documented in the audit |
| `vlm/biomedclip/biomedclip_vitb16_combined.pth` | BiomedCLIP ViT-B/16 image tower + linear head | validation accuracy 0.943 overall, **0.705 excluding malignant-only ISPY2** | single brief run; epoch-1 checkpoint |
| `vlm/medgemma_lora/` | LoRA (r=16, α=32) on MedGemma-4B-it, 250 steps | ≈0.70 aggregate accuracy on 40-sample subsets | not a tuned comparison; Grad-CAM maps show a grid artifact |
| `preprocessing/cyclegan_breast2busbra.pth` | CycleGAN BrEaST → BUS-BRA style | — | **learns to draw BUS-BRA calipers and text onto translated images**; do not use as a preprocessing step without removing annotations first |

| `leakage_controlled/busi_dupgroups/busi_resnet18_dupgroups_fold{1..5}.pth` | ResNet-18, `--paper-match`, 5-fold CV with SSIM duplicate groups kept in one fold | AUC 0.945 (seed 42; 0.938 ± 0.007 over 3 seeds) | BUSI has no patient IDs; splits are image-level otherwise |
| `leakage_controlled/busc_frameblocks/busc_resnet18_frameblocks10_fold{1..5}.pth` | ResNet-18, `--paper-match`, 128 px, 5-fold CV grouped by blocks of 10 consecutive frames | AUC 0.949 (seed 42; 0.935 ± 0.011 over 3 seeds), accuracy 0.787 | image-level splits leak consecutive frames (accuracy 0.992) |
| `multisource/multisource_resnet18_fold{1..5}.pth` | three-branch late-fusion model (`FusionLateModel`, masked scalar weights), one ResNet-18 per modality, all layers trained, fold-aligned splits (BUSC in 10-frame blocks) | mean per-branch test AUC 0.806 ± 0.015 over 13 sources | every training sample has one modality, so this is per-modality multi-source training, not cross-modality fusion; fold k never saw fold k's test images | Earlier combined-fusion checkpoints are
deliberately not included: their ultrasound branch was warm-started from the BUS-CoT
model above, and their BatchNorm statistics were contaminated by zero-filled absent
branches (fixed in the code on 2026-11-11).

## Loading

```python
import torch
from training.train import SingleBackboneClassifier  # from the MMFM repository
model = SingleBackboneClassifier('resnet18', pretrained=False, embedding_dim=512, num_classes=2)
model.load_state_dict(torch.load('reproductions/busbra/busbra_resnet18_fold1.pth', map_location='cpu'))
```

ODELIA MST checkpoints load with `scripts/reproduce_odelia.py`; the CycleGAN with
`scripts/train_cyclegan.py`'s generator; the MedGemma adapter with
`peft.PeftModel.from_pretrained(base, 'vlm/medgemma_lora')`.
