---
license: cc-by-4.0
tags:
  - medical-imaging
  - breast-cancer
  - classification
  - pytorch
  - resnet
pipeline_tag: image-classification
---

# MMFM Breast-Imaging Baselines — Single-Modality ResNet-18 Checkpoints

ResNet-18 binary (benign vs. malignant) classification checkpoints, trained
per-dataset on eight public breast-imaging sources spanning ultrasound,
mammography, and DCE-MRI. Code: [MMFM repo](https://github.com/NouranFadlallah/MMFM).
Full write-up, methodology, and comparison to each source paper's own
numbers: `docs/latex/main.tex` in the repo.

**These are single-modality baselines, not the 3-branch fusion model.**
Each checkpoint is `models.SingleBackboneClassifier` (one ResNet-18 backbone,
ImageNet-pretrained, first conv adapted for non-RGB inputs where applicable)
— see `models/backbone.py` / `training/train.py` in the repo for the
loading code.

## Files

| File | Dataset | Modality | Evaluation | Test accuracy | Notes |
|---|---|---|---|---|---|
| `busbra_resnet18_fold1.pth` | BUS-BRA | Ultrasound | 1 of 5 official CV folds | 0.828 ± 0.023 (5-fold mean) | Paper-matched recipe; see Table 2 in the report |
| `busi_resnet18_fold1.pth` | BUSI | Ultrasound | 1 of 5 stratified CV folds | 0.884 ± 0.038 (5-fold mean) | Full 780-image release |
| `busc_resnet18_fold1.pth` | BUSC | Ultrasound | 1 of 5 stratified CV folds | 0.992 ± 0.010 (5-fold mean) | **Near-perfect result flagged as unverified** — Grad-CAM analysis in the report suggests possible shortcut learning (no ROI crop in this dataset's preprocessing); do not treat as a working classifier without further checks |
| `breast_lesions_usg_resnet18_fold1.pth` | BrEaST-Lesions USG | Ultrasound | 1 of 5 case-level CV folds | 0.726 ± 0.042 (5-fold mean) | High fold-to-fold variance, ~51 test cases/fold |
| `mias_resnet18_fold1.pth` | mini-MIAS | Mammography | 1 of 5 patient-level CV folds | 0.599 ± 0.096 (5-fold mean) | Weak result; misses ~51% of malignant cases (see per-class bias table in the report) |
| `breastdm_resnet18.pth` | BreastDM | DCE-MRI (9-channel `img9Se`) | Official train/val split | 0.880 | Within 0.002 accuracy of the paper's own LG-CAFN fusion model; specificity is weak (0.542) |
| `cdd_cesm_resnet18.pth` | CDD-CESM | Mammography (contrast-enhanced) | Single patient-level split, smoke test | 0.614 | **Not a tuned reproduction** — no augmentation search or class weighting yet |
| `cmmd_resnet18.pth` | CMMD | Mammography | Single patient-level split, smoke test | 0.758 | Class-weighted CE; **not a tuned reproduction** — no CV, no multi-view aggregation |

For the 5-fold datasets, this is **one fold's checkpoint, not an ensemble or
the averaged model** — reported accuracy is the 5-fold mean from the full
report for context, not this specific file's number. See the repo's
`docs/results_report.md` and `docs/per_class_results.md` for per-fold and
per-class (sensitivity/specificity/FP-vs-FN bias) detail.

## Known limitations (see the full report for detail)

- BUSC's near-perfect accuracy is flagged, not confirmed — Grad-CAM shows
  diffuse, non-lesion-localized activation, consistent with the dataset's
  lack of an ROI crop step.
- mini-MIAS, CDD-CESM, and CMMD are weak/preliminary results, not
  submission-ready baselines.
- Most datasets use image- or row-level splits, not patient-level, where the
  source release doesn't ship a patient identifier (BUSI, BUSC) — possible
  near-duplicate leakage across folds.

## Source data attribution

- BUS-BRA: Gómez-Flores et al., 2024, *Medical Physics*, [10.1002/mp.16812](https://doi.org/10.1002/mp.16812)
- BUSI: Al-Dhabyani et al., 2020, *Data in Brief*, [10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863)
- BUSC: [Mendeley Data](https://data.mendeley.com/datasets/vckdnhtw26/1)
- BrEaST-Lesions USG: Pawlowska et al., 2024, *Scientific Data*, [10.1038/s41597-024-02984-z](https://doi.org/10.1038/s41597-024-02984-z), CC BY 4.0
- mini-MIAS: Suckling et al., 1994
- BreastDM: Zhao et al., 2023, *Computers in Biology and Medicine*, [10.1016/j.compbiomed.2023.107255](https://doi.org/10.1016/j.compbiomed.2023.107255)
- CDD-CESM: Khaled et al., 2022, *Scientific Data*, [10.1038/s41597-022-01238-0](https://doi.org/10.1038/s41597-022-01238-0), CC BY 4.0, via [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611)
- CMMD: Cui et al., 2021, The Cancer Imaging Archive, [10.7937/tcia.eqde-4b16](https://doi.org/10.7937/tcia.eqde-4b16), CC BY 4.0, via [TCIA](https://www.cancerimagingarchive.net/collection/cmmd/)

## Intended use

Research and educational use for breast-imaging CAD baselines and
reproduction studies. **Not validated for clinical use.** Several of these
checkpoints (BUSC, mini-MIAS, CDD-CESM, CMMD) have known weaknesses
documented above and in the linked report — read those sections before
using any of these checkpoints for anything beyond a baseline comparison.
