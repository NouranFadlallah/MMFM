# Training Plan — Five Newly Audited Datasets (2026-10-16)

Covers CBIS-DDSM, BCS-DBT extra (train + test releases), BUS-CoT, ODELIA and
BreastDCEDL-ISPY2. All five are downloaded and audited (`analysis/<name>/`,
[dataset_audit.md](dataset_audit.md)), and none has been trained single-mode
yet. Four of them are not even `--dataset` choices in `training/train.py`.

Two phases:

1. **Paper reproduction.** For each dataset, copy the source paper's
   preprocessing and training recipe and aim for its reported number.
2. **Combined foundation model.** Add the reproduced datasets as new sources
   in `--dataset combined`.

Hardware: one RTX 3060 (12 GB VRAM), 15 GB RAM. Paper recipes that assume
bigger GPUs get gradient accumulation instead of larger batches.

---

## Decisions needed before Phase 1

| # | Decision | Recommended |
|---|---|---|
| D1 | CBIS-DDSM: the paper number we would reproduce (Shen et al., whole-image AUC 0.88) needs all 2,478 full mammograms. We have 187 locally, and the full set does not fit in the 66 GB free on `/mnt/data`. | Reproduce Shen's recipe on the ROI crops. Report it as a recipe replication without a paper target. Revisit the full download if disk frees up. |
| D2 | BreastDCEDL-ISPY2: its paper task is **pCR prediction**, not benign vs. malignant, and the pCR labels and official split are a separate small CSV we don't have. | Download the metadata CSV from the BreastDCEDL GitHub/Zenodo (a few MB), then reproduce pCR. ISPY2 stays a malignant-only source in `combined`. |
| D3 | ODELIA has no lesion masks, so we cannot cut BreaDM-style 9-channel lesion crops for the MRI branch. | Give the MRI branch a 3-channel whole-breast input (Pre / Sub_1 / T2 middle slices, as in the ODELIA paper) for ODELIA only. Alternative: a separate 3D ODELIA encoder. |
| D4 | ODELIA's MST model uses DINOv3. Its weights on Hugging Face sit behind a licence you must accept yourself. | Accept it on huggingface.co (`facebook/dinov3-*`) before step R5 runs. |

---

## Phase 0 — CPU work, can start now (GPU driver still down)

| Step | What | Time |
|---|---|---|
| P0.1 | Add a `PAPER_RECIPES` dict in `training/train.py`, keyed by dataset, holding optimizer, lr, schedule, weight decay, epochs, batch size, input size, padding and augmentation. `--paper-match` reads it instead of the BUS-BRA-only branch at `train.py:1223`. BUS-BRA's current recipe becomes the `busbra` entry, so its behaviour does not change. | 45 min |
| P0.2 | Add `--dataset` choices `cbis_ddsm`, `bcsdbt_full` and `bus_cot` with frame builders. Each one reuses its `audit/sources.py` adapter so labels and splits match the audit. | 1.5 h |
| P0.3 | Cache 16-bit DICOM crops (CBIS-DDSM) and annotated DBT slices (BCS-DBT train/test) to PNG, as already done for `bcsdbt_png` and `cmmd_png`. | 30 min work + about 1 h unattended |
| P0.4 | Write `scripts/reproduce_odelia.py`, a thin wrapper around the official `mueller-franzes/odelia_breast_mri` code that points it at our `/mnt/data/.../odelia/unilateral` parquet shards. | 1 h |
| P0.5 | Write `scripts/reproduce_breastdcedl_pcr.py`: RGB-fused tumour slices plus a ViT. Blocked on D2 (metadata CSV). | 1 h |
| P0.6 | CPU smoke test of every new path with `--device cpu --max-samples 32 --epochs 1`. Add a split/leakage unit test per new dataset to `tests/`. | 45 min |

### Phase 0 status (2026-10-16)

| Step | Status |
|---|---|
| P0.1 | Done. `PAPER_RECIPES` covers `default` (BUS-BRA SGD, unchanged), `bus_cot` (AdamW + cosine) and `cbis_ddsm` (Shen 3-stage unfreeze). Tests: `tests/test_paper_recipes.py`. |
| P0.2 | Done. New `--dataset` choices: `bus_cot` (official test held fixed; `--kfold` rotates validation over trainval), `bcsdbt_full` (396 views / 201 patients) and `cbis_ddsm` (Shen's 85/15 patient split with 10% validation; `--cbis-include-bwc`). New transforms: `PadToSquareTransform`, `ShenTransform`. |
| P0.3 | Done for CBIS-DDSM (2,006 crops) and BCS-DBT (396 slices). BCS-DBT now decodes only the annotated frame: 5 min instead of an estimated 10 h, with pixel-identical output. ODELIA `.npz` cache was still building at the time of writing. |
| P0.4 | Written: `scripts/reproduce_odelia.py` (cache / eval-released / train). Blocked on D4: the gated `ODELIA-AI/MST` and `facebook/dinov3-vits16-pretrain-lvd1689m` licences, plus `pip install x_transformers einops`. |
| P0.5 | Written: `scripts/reproduce_breastdcedl_pcr.py`. Blocked on D2: `BreastDCEDL_metadata.csv` from the authors' GitHub. |
| P0.6 | Done. GPU smoke runs of all three new `--dataset` paths, zero patient overlap across splits, 56/56 tests pass. |

---

## Phase 1 — Paper reproductions (GPU)

Rule for every run: keep modality dropout off, log to W&B project
`mmfm-<dataset>`, and save the checkpoint as `<dataset>_single_<backbone>_best.pth`.
A result **counts as reproduced** if it lands inside the paper's confidence
interval or within ±0.03 AUC when the paper gives no interval.

### R1. BUS-CoT (ultrasound), easiest, run first

- **Paper:** Yu et al. 2025, [arXiv 2509.17046](https://arxiv.org/abs/2509.17046), Table 2.
- **Target:** ResNet-50, **AUROC 0.8085 (95% CI 0.781–0.836), ACC 0.738**. Swin-L is the best model in the paper at AUROC 0.846; that can be a second run.
- **Data:** the whole `BUS-Lesion` set (`trainval` / `test`, patient-level 8:2), all sources included, so the comparison is like for like. Drop the 349 "Others" lesions (not benign or malignant).
- **Recipe:** keep the aspect ratio and zero-pad to 224×224. ImageNet ResNet-50 with global average pooling and an FC layer. AdamW (β 0.9/0.999), lr 5e-5 with cosine decay, weight decay 0.1, batch size 16, **5 epochs**, 5-fold cross-validation inside `trainval`, then the held-out `test`.
- **Local gap:** none.
- **GPU time:** about 1 h (5 folds × about 10 min).

### R2. CBIS-DDSM (mammography)

- **Paper:** the dataset paper (Lee et al., *Sci Data* 2017) reports segmentation Dice only, with no classifier. The standard reproduction target is Shen et al. 2019, [arXiv 1708.09427](https://arxiv.org/abs/1708.09427).
- **Target:** whole-image AUC 0.88 for a single model, 0.91 for an ensemble. That needs the full mammograms (see D1). On crops we have **no comparable paper number**, so we report the result on its own.
- **Data:** 2,006 ROI crops. Headline set is BENIGN vs. MALIGNANT, pathology-confirmed only. A second run adds BENIGN_WITHOUT_CALLBACK, following audit finding 3. Split by **patient**, not the official folders, because 16 patients cross the official split (audit finding 2). Patient-level 85/15, with 10% of train held out for validation, as in Shen.
- **Recipe (Shen's patch stage):** ImageNet ResNet-50 at 224×224. Subtract the training-set mean gray level. Adam, batch size 32, class-balanced sample weights, 3-stage unfreeze: lr 1e-3 last layer only for 3 epochs, then 1e-4 top 46 layers for 10 epochs, then 1e-5 all layers until epoch 200 (the S1 schedule). Augmentation: horizontal and vertical flips, rotation ±25°, shear ±0.2, zoom 0.8–1.2, channel shift ±20.
- **GPU time:** about 1.5 h.

### R3. BCS-DBT, full benign/cancer set (mammography, DBT)

**Decision (2026-10-17): skipped, BCS-DBT stays classification-only.** The
authors never released trained weights: `mateuszbuda/duke-dbt-detection` has
no releases, and its README only refers to a local `./yolo.pt`. So R3a cannot
run, and R3b (retraining their detector: about 3 h of porting, about 50 GB of
slice export, 4-6 h of GPU) was judged not worth it for a detection result the
fusion model doesn't use. The original plan is kept below for reference.

- **Paper:** Buda et al. 2021, *JAMA Netw Open*, [arXiv 2011.07995](https://arxiv.org/abs/2011.07995). This is a **detection** paper.
- **Target:** **60% sensitivity at 2 false positives per DBT volume on the validation set.** We hold the whole validation partition locally (280 patients, normal views included), so this number is reproducible. The test number (42% / volume, 65% / breast) is not, because the test partition's normal views are not local.
- **Step R3a, about 30 min:** run the authors' released model (`mateuszbuda/duke-dbt-detection`) on our validation partition and check we get about 60%. This proves our data loading and FROC evaluation are correct before we train anything.
- **Step R3b, retrain:** the paper trains only on positive slices from biopsied volumes, which we have (200 train views). Window-level from the DICOM header. Downscale 2× with a 2×2 mean filter. Erode skin with a 5 px radius. Keep the largest connected component. DenseNet single-stage detector with 96 px grid cells and 256 px anchors. Random 1056×672 crops containing the box. Adam, lr 1e-3, batch size 16, 100 epochs, early stopping patience 25, focal loss. Select the model on validation sensitivity at 2 FP per volume.
- **Also, for `combined`:** the box-cropped benign/cancer slice classifier over all 396 views (validation + train + test, 201 patients), split by patient with ResNet-18. It has no paper target, but it is what the mammography branch actually consumes.
- **GPU time:** about 30 min for R3a. Up to about 4 h for R3b at 100 epochs, likely less with early stopping. About 20 min for the classifier.

### R4. BreastDCEDL-ISPY2 (MRI), pCR task

- **Paper:** Fridman et al. 2025, [arXiv 2506.12190](https://arxiv.org/abs/2506.12190), Table 2.
- **Target:** I-SPY2 test subset (n = 99): **image-only AUC 0.78**, accuracy 0.73. With clinical features added via a random forest: AUC 0.85.
- **Data:** the official train/val/test split and pCR labels come from the metadata CSV (D2). We have I-SPY2 only. The paper also trained on Duke and I-SPY1 (1,099 training patients total), so expect us to land somewhat below 0.78 for that reason alone.
- **Recipe:** tumour-containing axial slices. RGB fusion of SER timepoints 0, 2 and min(last, 6) as R, G, B. Min-max to 8-bit. ViT (ImageNet). The paper does not state ViT hyperparameters, so read them from the authors' GitHub (`naomifridman/BreastDCEDL`) during P0.5.
- **GPU time:** about 1.5 h.

### R5. ODELIA (MRI), heaviest, run overnight

- **Paper:** Müller-Franzes et al. 2025, [arXiv 2506.00474](https://arxiv.org/abs/2506.00474), Table 2.
- **Target:** MST, 3-class (no lesion / benign / malignant). **In-distribution macro AUC 79.0 ± 1.3**, micro AUC 87.4. Out-of-distribution on RSH: macro AUC 63.1.
- **Data:** the unilateral 256×256×32 configuration, which is what we downloaded. Use the official `split_unilateral.csv` 5 folds (64/16/20). Train on every centre except RSH. Test in-distribution on the test split, then on RSH.
- **Recipe:** MST, i.e. DINOv3 per-slice features plus a transformer over slices. Inputs Pre, Sub_1 and T2. Augmentation: random rotation, horizontal and vertical flips, Gaussian noise, random 224×224×32 crop. AdamW, lr 1e-5, batch size 8 (on the 3060: batch size 2 × 4 accumulation steps), cross-entropy, early stopping. Use the official code (P0.4).
- **Quick check first:** score the released `ODELIA-AI/MST` weights on fold 0's test split, about 15 min.
- **Watch for:** the centre shortcut (audit finding 1). Report per-centre AUC as well.
- **GPU time:** the paper reports about 1 h per run on an L40S. Expect 2–3 h per fold on the 3060, so **10–15 h** for all 5 folds.

---

## Phase 2 — Add them to `--dataset combined`

**Status (2026-10-17): wired in.** `--dataset combined` now adds CBIS-DDSM
crops and full BCS-DBT (mammography), BUS-CoT's de-duplicated clinical subset
(ultrasound: 2,036 images from TFUSL, TUHD, BUDIAT) and ODELIA (MRI). For D3,
ODELIA uses a 9-channel stack: the central slice ±1 for Pre, Sub_1 and T2. The
ResNet-18 warm-starts and the combined run are queued in `runs/queue/jobs.txt`.

**First Phase 2 run (2026-10-17) and the pending rerun.** The combined run
(`combined_fusion_resnet18_phase2_best.pth`) stopped after 7 epochs and kept
epoch 2 by lowest validation loss (mammography accuracy 0.611, ultrasound
0.731, MRI 0.985). By epochs 6-7 mammography had reached 0.665 and ultrasound
0.803. Overall accuracy is inflated by ISPY2's malignant-only MRI validation
slices.

**Rerun with per-source AUC and mean-branch-AUC selection (2026-10-19).**
`combined_fusion_resnet18_phase2_auc_best.pth`, same warm-starts. Branch AUCs
pool only sources with both classes, so ISPY2 (malignant-only) is excluded
from the MRI AUC; pooling it had inflated MRI to 0.97.

| Epoch | Mammography AUC | Ultrasound AUC | MRI AUC | Mean | Train loss |
|---|---|---|---|---|---|
| **1 (kept)** | 0.679 | **0.888** | 0.782 | **0.783** | 0.48 |
| 2 | 0.699 | 0.823 | 0.814 | 0.779 | 0.32 |
| 6 (early stop) | 0.663 | 0.853 | 0.790 | 0.769 | 0.12 |

Per source at epoch 1: MIAS 0.375 (n = 10), BCS-DBT 0.680, CBIS-DDSM 0.677,
BUS-BRA 0.908, BUSI 0.922, BUSC 1.000, BrEaST 0.960, BUS-CoT 0.791,
BreaDM 0.847, ODELIA 0.699, ISPY2 accuracy 0.959 (no AUC: malignant only).
Full table: `runs/run-4/combined_fusion_resnet18_phase2_auc_best_source_metrics.json`.

**Finding: joint training does not improve on the warm-started branches.**
Training loss falls from 0.48 to 0.12 while validation AUC drifts down
(ultrasound 0.888 -> 0.82-0.85), so the best checkpoint is epoch 1, which is
essentially the warm-starts. The likely cause is lr 1e-4 on all parameters
wiping pretrained features. The next experiment should be a lower lr (1e-5)
or frozen backbones with only the projections, heads and fusion weights
trained.

**Follow-up runs (2026-10-19).** Same sources and warm-starts:

| Variant | Kept epoch | Mammography | Ultrasound | MRI | Mean |
|---|---|---|---|---|---|
| lr 1e-4, all trained | 1 | 0.679 | 0.888 | 0.782 | 0.783 |
| lr 1e-5, all trained (`--lr 1e-5`) | 9 of 10 | 0.717 | 0.868 | 0.841 | 0.809 |
| Frozen backbones (`--freeze-backbones`) | 1 (stopped at 6) | 0.747 | 0.881 | 0.824 | 0.817 |

Frozen backbones score highest but peak at epoch 1, then plateau at 0.80-0.815.
lr 1e-5 was still improving at its 10-epoch cap. Source metrics are in
`runs/run-4/combined_fusion_resnet18_phase2_{lr1e-5,frozen}_best_source_metrics.json`.

**Adding AMBL as a 4th MRI source (2026-10-20).** Same frozen-backbone setup:

| | With AMBL (kept epoch 4) | Without (kept epoch 1) |
|---|---|---|
| Mean branch AUC | 0.808 | 0.817 |
| MRI branch (sources with both classes) | 0.816 | 0.824 |
| BreaDM | 0.885 | 0.904 |
| ODELIA | 0.693 | 0.647 |
| AMBL (validation slices) | 0.806 | - |

Roughly neutral: ODELIA gains and BreaDM loses by amounts within the noise of
their small validation sets. The mammography and ultrasound sources, which
AMBL cannot affect directly, also shift, because a different epoch was kept.
This shows how much single-run numbers move here. AMBL stays in the combined
model. Its single-dataset reference remains lesion size (lesion AUC 0.854;
`dataset_reproduction_plan.md` section 16).

The paper recipes use ResNet-50, ViT and DINOv3. `combined` warm-starts branches
from single-mode checkpoints with a matching architecture (`_warm_start_branch`).
So for each new dataset, **also** run one ResNet-18 single-mode training with the
same preprocessing (about 15–30 min each) and use that checkpoint for warm-start.

| Branch | New source | Rules (from the audit) |
|---|---|---|
| Mammography | CBIS-DDSM crops | BENIGN + MALIGNANT only; split by patient |
| Mammography | BCS-DBT full set | Replaces the current validation-only `bcsdbt` source (75 → 396 views); split by patient |
| Ultrasound | BUS-CoT, **new clinical subset only** (TFUSL, TUHD, UDIAT, STU-Hospital, Birads-ultrasound: 3,599 images) | Never the BUSI or BUS-BRA copies (2,655 images identical to ours). De-duplicate TUHD first (36% duplicates, 88 label conflicts). Add `bus_cot` to `ULTRASOUND_DATASET_CHOICES`. |
| MRI | ODELIA benign + malignant breasts (504) | Drop no-lesion breasts. Hold out RSH as an OOD test centre. Representation per D3. Use centre-balanced sampling. |
| MRI | ISPY2 | Already in `combined` (malignant-only). No change. |

Steps:

1. Write the source builders, gated on the data path existing, like BCS-DBT at `train.py:716`. About 1.5 h.
2. CPU smoke test of `combined` with every source on. About 20 min.
3. GPU run: same settings as `combined_resnet18_ispy2_bcsdbt`, with warm-starts from Phase 1's ResNet-18 checkpoints. About 2–3 h.
4. Report per-branch and **per-source** validation metrics. Check that no single source dominates its branch's accuracy, which is the ISPY2 lesson in [dataset_reproduction_plan.md](dataset_reproduction_plan.md).

---

## Order and total time

| Order | Item | Needs | Approx. time |
|---|---|---|---|
| 1 | Phase 0 (P0.1–P0.6) | CPU only | about 6 h of work |
| 2 | R1 BUS-CoT | GPU | 1 h |
| 3 | ~~R3a BCS-DBT released-model check~~ (skipped: no released weights) | – | – |
| 4 | R2 CBIS-DDSM | GPU | 1.5 h |
| 5 | R4 ISPY2 pCR | GPU, D2 | 1.5 h |
| 6 | R5 ODELIA (overnight) | GPU, D4 | 10–15 h |
| 7 | ~~R3b BCS-DBT retrain~~ (skipped) | – | – |
| 8 | ResNet-18 warm-start runs (5 datasets) | GPU | 2 h |
| 9 | Phase 2 combined | GPU | 4–5 h |

Total GPU time is about 25–30 h, so plan 2 nights plus 2 days. When each run
finishes, record the result in [dataset_reproduction_plan.md](dataset_reproduction_plan.md)
as a new numbered section (12–16), in the same format as sections 8–11.
