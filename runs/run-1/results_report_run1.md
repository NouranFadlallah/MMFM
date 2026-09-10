# ResNet-18 Baseline Results — Six-Dataset Report

This report summarizes the single-modality ResNet-18 training runs completed to date
(all logged to Weights & Biases, projects `mmfm-<dataset>`) and compares the accuracy
we obtained against results reported in each dataset's source literature.

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

## Results table

| # | Dataset | Modality | Main paper | Local data used | Reported result (literature) | Our ResNet-18 result | Run config |
|---|---|---|---|---|---|---|---|
| 1 | **BUS-BRA** | Ultrasound | Gómez-Flores et al., 2024, *Medical Physics* — [BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems](https://doi.org/10.1002/mp.16812) | 1,875 images / 1,064 patients, benign vs. malignant | **ResNet-18, 5-fold CV, lesion-ROI crops (paper's own Table 3):** Accuracy 0.865 ± 0.020, Sensitivity 0.859 ± 0.075, Specificity 0.868 ± 0.014, AUC 0.931 ± 0.025. (ResNet50: Acc 0.859, Sens 0.808, Spec 0.883, AUC 0.920 — ResNet18 was the slightly better choice due to higher sensitivity.) For reference, the senior ultrasonographer's own reads scored Accuracy 0.827, Sensitivity 0.932, Specificity 0.776 (Table 4) | **Paper-matched, official fold 1/5 (K5B, `--paper-match`):** Accuracy 0.807, Sensitivity 0.837, Specificity 0.792, Precision 0.671, F1 0.745, AUC 0.886 — held-out test fold, SGD(momentum 0.9, lr 1e-3), weighted CE, 100 epochs no early stopping, contrast-stretched inputs. *Single fold only, not yet averaged over all 5 like the paper's Table 3.* Prior ad-hoc-split run (90/10 random, Adam, 20 epochs, no contrast stretch): val accuracy 83.9% | Paper-matched: 100 epochs, batch 32, 224px, official K5B fold 1 test set, 80/20 case-level hold-out for train/val. Ad-hoc run: 20 epochs, batch 16, 224px, image-level 90/10 split |
| 2 | **BUSI (BUS)** | Ultrasound | Al-Dhabyani et al., 2020, *Data in Brief* — [Dataset of breast ultrasound images](https://pubmed.ncbi.nlm.nih.gov/31867417/) | **163/780** images (local checkout is an incomplete subset) | Dataset-description paper; no classifier benchmark of its own. Later work on the full 780-image release commonly reports ~90–96% accuracy with various CNNs, but on 3-class (normal/benign/malignant) splits, not this run's local subset/binary setup | **Checkpoint metrics (n=15 val):** Accuracy 0.867, Sensitivity 0.800, Specificity 0.900, Precision 0.800, F1 0.800, AUC 0.940. (Previously reported 73.3% was the final-epoch number, not this checkpoint's — see correction note above.) | 20 epochs, batch 16, 224px, stratified split on the 163-image local subset |
| 3 | **BUSC** | Ultrasound | Mendeley Data, 2023 — [BUSC Dataset](https://data.mendeley.com/datasets/vckdnhtw26/1); segmentation follow-up: Iqbal & Sharif, 2023, *Expert Systems with Applications* — [PDF-UNet](https://doi.org/10.1016/j.eswa.2023.119718) | 250 images (100 benign / 150 malignant), 128×128 | PDF-UNet is a segmentation paper (Dice/IoU on radiologist-annotated masks); it reports no classification accuracy comparable to ours | **Checkpoint metrics (n=25 val):** Accuracy 1.000, Sensitivity 1.000, Specificity 1.000, Precision 1.000, F1 1.000, AUC 1.000 — see overfitting caveat below | 20 epochs, batch 16, 128px, image-level split, no patient ID in dataset |
| 4 | **BrEaST-Lesions USG** ("breast") | Ultrasound | Pawlowska et al., 2024, *Scientific Data* — [Curated benchmark dataset for ultrasound based breast lesion analysis](https://doi.org/10.1038/s41597-024-02984-z) | 256 cases (227 train / 25 val, local split) | Dataset-release paper; provides annotations/labels but no classifier accuracy benchmark of its own | **Checkpoint metrics (n=25 val):** Accuracy 0.840, Sensitivity 1.000, Specificity 0.800, Precision 0.556, F1 0.714, AUC 0.890. (Previously reported 60.0% was the final-epoch number — see correction note above.) Sensitivity 1.0 / precision 0.56 means it never misses a malignant case but over-calls benign ones as malignant | 20 epochs, batch 16, 224px |
| 5 | **MIAS / mini-MIAS** | Mammography | Suckling et al., 1994 — [The Mammographic Image Analysis Society Digital Mammogram Database](https://doi.org/10.1016/0531-5131(94)90040-X) | 322 images total; local lesion-patch extraction yielded 109 train / 10 val patches | 1994 dataset-release paper reports no CNN benchmark (predates deep learning). Later third-party studies on mini-MIAS report ResNet-family accuracies roughly in the 89–98% range, but with modern augmentation and different task framings (image-level normal/benign/malignant, not this run's patient-split lesion patches) | **Checkpoint metrics (n=10 val, 2 malignant / 8 benign):** Accuracy 0.900, Sensitivity 0.500, Specificity 1.000, Precision 1.000, F1 0.667, AUC 1.000. (Previously reported 50.0% was the final-epoch number — the actual checkpoint gets 9/10 right, missing only 1 of the 2 malignant cases; still not meaningful at n=10, see caveat below) | 20 epochs, batch 16, 224px, patient-level split, only 10 val samples |
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

- **BUS-BRA fold 1/5 is done; folds 2-5 still need to run** (`training/train.py
  --dataset busbra --single-mode --busbra-kfold 5 --paper-match ...`, omitting
  `--busbra-test-fold` runs all 5 sequentially and writes the mean±std summary).
  At ~68s/epoch × 100 epochs, one fold takes ~1.9 hours, so all 5 is a ~9.5-hour
  unattended run. Fold 1 alone (0.807 accuracy / 0.886 AUC) is already close to the
  paper's 5-fold mean (0.865 / 0.931) but a single fold isn't a fair comparison to
  an averaged number — treat it as directional, not final.
- **BUSC's 100% across every metric is a red flag, not a win** — 250 total images,
  no patient ID, an image-level (not patient-level) split, and only 20 epochs at
  128px is a classic small-dataset overfitting setup. Don't report this number
  without a stratified k-fold check.
- **MIAS's checkpoint metrics are on only 10 samples (2 malignant)** — getting 1 of
  2 malignant cases right or wrong swings sensitivity by 50 points. Not statistically
  meaningful either way, regardless of which epoch's checkpoint you look at.
- **The final-epoch vs. best-checkpoint mismatch found above** (see "Checkpoint-vs-
  final-epoch correction") means every number anyone quotes from a W&B run summary
  for these single-mode runs should be treated as suspect unless it was re-derived
  from the actual saved `.pth` file, the way [scripts/eval_checkpoints.py](../scripts/eval_checkpoints.py)
  does. Consider changing `_train_model` to log the checkpoint-epoch's metrics into
  `run.summary` directly, so this doesn't have to be done as a manual follow-up.
- **BUSI and BrEaST are trained on local subsets**, not the full released datasets,
  so literature numbers for those datasets are not apples-to-apples even qualitatively.
- **Splits are image/patch-level, not patient/case-level**, for every run except MIAS
  and BreaDM (which do use official/patient splits) — this can leak information
  between train and validation for BUS-BRA, BUSI, BUSC, and BrEaST.
- Full per-dataset gaps and next steps are tracked in
  [dataset_reproduction_plan.md](dataset_reproduction_plan.md).
