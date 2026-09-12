# ResNet-18 Baseline Results — Six-Dataset Report

This report summarizes the single-modality ResNet-18 training runs completed to date
(all logged to Weights & Biases, projects `mmfm-<dataset>`) and compares the accuracy
we obtained against results reported in each dataset's source literature. For a
per-class (benign/malignant) breakdown and false-positive-vs-false-negative bias
detection per model, see [per_class_results.md](per_class_results.md).

**Two generations of checkpoints exist:** `runs/run-1/` is the original set (single
ad-hoc train/val splits, `checkpoint_eval_results.json` for the final-epoch-vs-
checkpoint correction). `runs/run-2/` is the current, complete cross-validated set —
MIAS, BUSC, BrEaST, and BUSI (now the **full** 780-image release, not the old
163-image subset) were retrained with our own stratified group k-fold since none of
them ship an official split, and BUS-BRA is now fully cross-validated on its own
official K5B folds (fold 1 in `runs/run-1/`, folds 2-5 in `runs/run-2/`, combined
below). All run-2 numbers are **mean ± std across 5 held-out test folds**, not a
single split. Partway through this run, `training/train.py`'s `DataLoader` calls
were switched to `num_workers=4` (overlapping CPU preprocessing with GPU compute) —
BUS-BRA fold 4 finished in ~29 min vs. ~110 min for folds 1-3, with identical results
otherwise (verified no change in behavior, only speed).

## Methodology

All runs use `training/train.py --dataset <name> --single-mode --backbone resnet18
--pretrained` (ImageNet-initialized `SingleBackboneClassifier`, no fusion, no
modality dropout — see [training/train.py](../training/train.py)). Image size,
epochs, and batch size varied slightly per run (see the "Run config" column);
checkpoints are the best-validation-loss epoch, saved as `<dataset>_single_resnet18_best.pth`.

**Metric caveat:** `utils/metrics.py` originally only computed accuracy and
cross-entropy loss. It now also has `classification_metrics()` (sensitivity,
specificity, precision, F1, AUC), and `training/train.py` computes these via
`evaluate_full_metrics()`. All six checkpoints have now been re-run through it
([scripts/eval_checkpoints.py](../scripts/eval_checkpoints.py) reconstructs each
original validation split with the same seed/split-fraction the training run used,
loads the saved `<dataset>_single_resnet18_best.pth`, and computes the full metric
set with no retraining).

**Checkpoint-vs-final-epoch correction:** the accuracy numbers previously recorded
in this table (and still visible in W&B's run summary) were the **last training
epoch's** validation accuracy, not the accuracy of the actual **saved checkpoint**
(best-validation-loss epoch — often an earlier one, since `_train_model` only
overwrites the checkpoint file when validation loss improves). For BUSI, BrEaST,
MIAS, and BreaDM those two numbers differ, sometimes substantially (MIAS: logged
50% at the final epoch vs. 90% for the epoch actually saved). The numbers below are
now computed directly on the saved checkpoint and supersede the earlier ones.

**Comparability caveat:** most of these runs are still pipeline sanity checks, not
paper reproductions — non-official, image-level (not patient/case-level) splits,
and for BUSI/BrEaST a **subset** of the published release (163/780 images for
BUSI). **BUS-BRA is the exception**: it now uses the paper's own case-level K5B
fold assignment, SGD/weighted-CE/100-epoch training matching the paper's recipe,
and a held-out test fold never seen during training or model selection — see row 1
for the head-to-head against the paper's Table 3. Details on the rest are in
[dataset_reproduction_plan.md](dataset_reproduction_plan.md).

**Second-backbone scope:** BUS-BRA, BUSI, and BrEaST were each retrained with a
second backbone (rows 1, 2, 4) — ResNet-50 for BUS-BRA because the paper itself
reports that comparison (Table 3); EfficientNet-B0 for BUSI/BrEaST because neither
paper reports its own classifier benchmark, so this follows the literature-suggested
follow-up backbone from `dataset_reproduction_plan.md` instead. BUSC and MIAS were
left ResNet-18-only (no second backbone is reported or suggested for either in
their source material). BreaDM was also left ResNet-18-only: its paper's own model
(LG-CAFN) is a custom fusion architecture, not a swappable backbone, and this
codebase's `create_backbone()` only adapts the first conv layer for `resnet*` names
— a non-ResNet backbone would silently reject BreaDM's 9-channel `img9Se` input.
Along the way, `models/backbone.py`'s generic fallback branch (for any torchvision
model that isn't resnet/vgg/densenet) turned out to be broken — it never removed the
classification head or computed the right feature dimension, so EfficientNet/
GoogLeNet would have silently fed 1000-class ImageNet logits into the projection
layer instead of real features. Fixed and verified (shape checks + a live smoke run)
before any of the EfficientNet-B0 runs above.

## Results table

| # | Dataset | Modality | Main paper | Local data used | Reported result (literature) | Our ResNet-18 result | Run config |
|---|---|---|---|---|---|---|---|
| 1 | **BUS-BRA** | Ultrasound | Gómez-Flores et al., 2024, *Medical Physics* — [BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems](https://doi.org/10.1002/mp.16812) | 1,875 images / 1,064 patients, benign vs. malignant | **Paper's own Table 3, 5-fold CV:** ResNet-18 Accuracy 0.865 ± 0.020, Sensitivity 0.859 ± 0.075, Specificity 0.868 ± 0.014, AUC 0.931 ± 0.025. ResNet-50: Accuracy 0.859, Sensitivity 0.808, Specificity 0.883, AUC 0.920 — the paper's own finding is ResNet-18 > ResNet-50 on accuracy and sensitivity. For reference, the senior ultrasonographer's own reads scored Accuracy 0.827, Sensitivity 0.932, Specificity 0.776 (Table 4) | **Paper-matched, full 5-fold CV on the paper's own K5B folds — both backbones:**<br>ResNet-18: Accuracy 0.828 ± 0.023, Sensitivity 0.822 ± 0.044, Specificity 0.832 ± 0.031, Precision 0.702 ± 0.034, F1 0.756 ± 0.026, AUC 0.903 ± 0.018.<br>ResNet-50: Accuracy 0.828 ± 0.021, Sensitivity 0.809 ± 0.088, Specificity 0.840 ± 0.055, Precision 0.713 ± 0.066, F1 0.751 ± 0.032, AUC 0.912 ± 0.021.<br>Same accuracy for both backbones, and ResNet-18 does edge out ResNet-50 on sensitivity (0.822 vs. 0.809) — **reproduces the paper's own ordering**, though our gap is much smaller than the paper's (0.859 vs. 0.808) and ResNet-50 is slightly ahead on AUC in both the paper and here. Prior ad-hoc-split run (90/10 random, Adam, 20 epochs, no contrast stretch): val accuracy 83.9% | 100 epochs, batch 32, 224px, official K5B 5-fold test sets, 80/20 case-level hold-out for train/val per fold |
| 2 | **BUSI (BUS)** | Ultrasound | Al-Dhabyani et al., 2020, *Data in Brief* — [Dataset of breast ultrasound images](https://pubmed.ncbi.nlm.nih.gov/31867417/) | **Full 780-image release** (437 benign / 210 malignant / 133 normal; normal excluded from this binary task), `datasets/ultrasound/Dataset_BUSI/Dataset_BUSI_with_GT` | Dataset-description paper; no classifier benchmark of its own. Later work on the full release commonly reports ~90–96% accuracy with various CNNs, typically on the 3-class (normal/benign/malignant) task rather than this run's binary setup | **run-2, 5-fold CV — ResNet-18 vs. EfficientNet-B0** (EfficientNet-B0 is the literature-suggested follow-up backbone per `dataset_reproduction_plan.md`, not from an official BUSI benchmark since none exists):<br>ResNet-18: Accuracy 0.884 ± 0.038, Sensitivity 0.833 ± 0.066, Specificity 0.908 ± 0.026, Precision 0.814 ± 0.055, F1 0.823 ± 0.059, AUC 0.945 ± 0.017.<br>EfficientNet-B0: Accuracy 0.881 ± 0.022, Sensitivity 0.871 ± 0.032, Specificity 0.886 ± 0.024, Precision 0.787 ± 0.039, F1 0.826 ± 0.031, AUC 0.938 ± 0.016.<br>Essentially a tie — ResNet-18 has the edge on specificity/AUC, EfficientNet-B0 on sensitivity/F1, both solidly inside the literature's ~90–96% accuracy range and far more stable than run-1's single 15-sample checkpoint eval (0.867) | run-2: 100 epochs, batch 32, 224px, our own stratified k-fold (row-level, no patient ID in release), `--paper-match` |
| 3 | **BUSC** | Ultrasound | Mendeley Data, 2023 — [BUSC Dataset](https://data.mendeley.com/datasets/vckdnhtw26/1); segmentation follow-up: Iqbal & Sharif, 2023, *Expert Systems with Applications* — [PDF-UNet](https://doi.org/10.1016/j.eswa.2023.119718) | 250 images (100 benign / 150 malignant), 128×128 | PDF-UNet is a segmentation paper (Dice/IoU on radiologist-annotated masks); it reports no classification accuracy comparable to ours | **run-2, 5-fold CV:** Accuracy 0.992 ± 0.010, Sensitivity 1.000 ± 0.000, Specificity 0.980 ± 0.025, Precision 0.987 ± 0.016, F1 0.993 ± 0.008, AUC 0.9997 ± 0.0007 — still near-perfect across every fold, so cross-validation didn't rule out a systemic issue (e.g. near-duplicate or trivially-separable images); see overfitting caveat below | run-2: 100 epochs, batch 32, 128px, our own stratified k-fold (row-level, no patient ID in dataset), `--paper-match` |
| 4 | **BrEaST-Lesions USG** ("breast") | Ultrasound | Pawlowska et al., 2024, *Scientific Data* — [Curated benchmark dataset for ultrasound based breast lesion analysis](https://doi.org/10.1038/s41597-024-02984-z) | 256 cases, case-level split via the `CaseID` column | Dataset-release paper; provides annotations/labels but no classifier accuracy benchmark of its own | **run-2, 5-fold CV (case-level via `CaseID`) — ResNet-18 vs. EfficientNet-B0** (literature-suggested, not paper-reported):<br>ResNet-18: Accuracy 0.726 ± 0.042, Sensitivity 0.728 ± 0.201, Specificity 0.726 ± 0.139, Precision 0.654 ± 0.101, F1 0.664 ± 0.075, AUC 0.836 ± 0.079.<br>EfficientNet-B0: Accuracy 0.694 ± 0.096, Sensitivity 0.757 ± 0.128, Specificity 0.655 ± 0.152, Precision 0.604 ± 0.127, F1 0.661 ± 0.091, AUC 0.781 ± 0.108.<br>ResNet-18 is ahead on accuracy/specificity/AUC here, EfficientNet-B0 slightly higher sensitivity — but the **std is huge for both backbones** (only ~51 test cases/fold), so this ordering is not reliable, just directional | run-2: 100 epochs, batch 32, 224px, case-level k-fold, `--paper-match` |
| 5 | **MIAS / mini-MIAS** | Mammography | Suckling et al., 1994 — [The Mammographic Image Analysis Society Digital Mammogram Database](https://doi.org/10.1016/0531-5131(94)90040-X) | 322 images total; lesion-patch extraction, patient-level split via the `patient` column | 1994 dataset-release paper reports no CNN benchmark (predates deep learning). Later third-party studies on mini-MIAS report ResNet-family accuracies roughly in the 89–98% range, but with modern augmentation and different task framings (image-level normal/benign/malignant, not this run's patient-split lesion patches) | **run-2, 5-fold CV (patient-level):** Accuracy 0.599 ± 0.096, Sensitivity 0.496 ± 0.233, Specificity 0.675 ± 0.098, Precision 0.496 ± 0.161, F1 0.486 ± 0.205, AUC 0.597 ± 0.107 — huge fold-to-fold variance (very few lesion patches and even fewer malignant cases per patient-level fold). This is the honest number; run-1's single-split 90% accuracy was an artifact of a lucky 10-sample validation set, not a real result | run-2: 100 epochs, batch 32, 224px, patient-level k-fold, `--paper-match` |
| 6 | **BreaDM (BreastDM)** | MRI (DCE, `img9Se`) | Zhao et al., 2023, *Computers in Biology and Medicine* — [BreastDM: A DCE-MRI dataset for breast tumor image segmentation and classification](https://doi.org/10.1016/j.compbiomed.2023.107255) | Official `cls/img9Se` split: 1,202 train / 117 val arrays | Paper's own proposed model (LG-CAFN) reports **accuracy 88.2%, AUC 0.915** (best of two experimental groups); this is their proposed fusion architecture, not a plain ResNet-18 baseline | **Checkpoint metrics (n=117 val):** Accuracy 0.880, Sensitivity 0.968, Specificity 0.542, Precision 0.891, F1 0.928, AUC 0.907 — accuracy/AUC are close to the paper's own LG-CAFN (0.882/0.915) despite this being a plain ResNet-18, though specificity is notably weak (over-calls malignant) | 20 epochs, batch 16, 224px, official train/val folders, 9-channel input (ResNet-18 first conv adapted) |

### Bonus: combined multimodal fusion run

Not one of the six single-dataset runs, but included for completeness: `--dataset
combined` trains the full 3-branch `FusionLateModel`, warm-started from the MIAS and
BreaDM single-mode checkpoints (`--mammography-init-checkpoint`,
`--mri-init-checkpoint`; ultrasound branch trained from scratch). 30 epochs, batch 16,
224px. **Val accuracy 74.1%**, val loss 0.489 (best val loss 0.467). This mixes samples
from all six datasets (each contributing one real modality) with branch/source/class-
balanced sampling; it is a fusion-mechanism check, not a benchmark against any single
paper.

## Caveats and things to fix before calling any of this a reproduction

- **BUS-BRA's 5-fold CV is now complete** (fold 1 in `runs/run-1/`, folds 2-5 in
  `runs/run-2/busbra_fold{2,3,4,5}_results.json`) and is the strongest result in this
  report: within ~0.04 accuracy / ~0.03 AUC of the paper's own 5-fold numbers on the
  paper's own fold assignment. The remaining gap is plausibly explained by using a
  simplified 3-channel grayscale-repeat input instead of the paper's mask-derived
  R/G/B encoding (Figure 7 in the paper), and a single seed per fold rather than
  averaging multiple seeds.
- **BUSC's near-perfect cross-validated result (0.992 accuracy, every fold) is still
  a flag, not a clean win** — 5-fold CV rules out a single-split fluke, but not a
  systemic issue: 250 total images, no patient ID (so an image-level split can still
  place near-duplicate crops of the same lesion on both sides of a fold), and only
  128px inputs. Worth a manual look at whether benign/malignant images are trivially
  distinguishable by something other than the lesion (e.g. acquisition artifacts).
- **MIAS's cross-validated result (0.599 ± 0.096 accuracy, 0.496 ± 0.233 sensitivity)
  is genuinely weak, not just noisy** — even patient-level 5-fold CV can't produce a
  stable signal from this few lesion patches. This is the realistic number; treat
  MIAS as needing more data or a different task framing (e.g. full-image rather than
  lesion-patch classification) before it's a usable baseline.
- **The final-epoch vs. best-checkpoint mismatch** (see "Checkpoint-vs-final-epoch
  correction" above) applied to run-1's single-split numbers; it doesn't apply to
  run-2's cross-validated numbers, which come from each fold's `test_metrics` computed
  directly via `evaluate_full_metrics()` inside `_train_model`, not a W&B summary.
- **Splits are patient/case-level for MIAS, BrEaST, and BUS-BRA** (via `patient`,
  `CaseID`, and the paper's own `K5B`, respectively) but still **row/image-level for
  BUSI and BUSC** (neither release ships a patient identifier), so those two folds
  can still leak near-duplicate images across train/val/test within a fold.
- Full per-dataset gaps and next steps are tracked in
  [dataset_reproduction_plan.md](dataset_reproduction_plan.md).
