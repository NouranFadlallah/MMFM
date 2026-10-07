# Dataset Analysis Plan (pre-deletion audit)

Goal: a systematic, reproducible audit of every dataset the project trains on.
The parts that need the original data run **before any raw data is deleted**
([storage_management_plan.md](storage_management_plan.md) §6 gates deletion on
them). Everything else can run later on the prepared data.

Starting point: the dataset-audit method in Siddique, Rada, Yap et al., *Deep
Learning for Breast Cancer Analysis: Classification and Segmentation on
Ultrasound Datasets* (2026 draft, §2). It covers image-quality metrics,
aHash/SSIM duplicate detection, z-score outliers, Doppler and caliper detection,
and a "claimed vs. verified" table per dataset. This plan extends that to
mammography, DBT and DCE-MRI. It also adds the other analyses the literature
commonly uses (see [References](#references)), grouped as:

- **Part 1 — Inventory and pixel-level audit:** A–G. The paper's method, extended.
- **Part 2 — Signal, texture and annotation analysis:** H–K.
- **Part 3 — Representation-level analysis:** L–N. Embeddings, dataset shift, source identifiability.
- **Part 4 — Label quality and shortcut risk:** O–R. Model-based checks.
- **Part 5 — Documentation:** S.

**Status: the raw stage is implemented** (`audit/` package +
`scripts/analyze_datasets.py`, see [Implementation status](#implementation-status));
the [prepared] stages (I, J, L–S) are not yet.

## Why part of this must run before deletion

The prepared data loses information the audit needs:

| Lost in preparation | Affects |
|---|---|
| DICOM headers (manufacturer, model, pixel spacing, kVp, VOI LUT, bits stored, laterality, view, acquisition date, patient age) | device and subgroup tables (E, Q), dataset shift (M), leakage checks (C) |
| Native bit depth (12–16 bit → uint8, min–max normalized per image) | dynamic range and bit depth (B), noise/SNR (H), uint8 vs uint16 decision for prepared data |
| Full frame outside the crop (BCS-DBT keeps only lesion boxes; ISPY2 keeps only tumor-bbox slices) | burnt-in markers (E), breast-region statistics, background-only shortcut test (R) |
| Excluded samples (BCS-DBT Normal/Actionable; CDD-CESM Normal and low-energy DM views; BUSI normal; non-tumor slices and unused DCE phases of ISPY2/BreaDM) | claimed-vs-actual counts (A), future tasks |
| Unused annotations (BreaDM `seg`/`seg3D` masks) | lesion and annotation statistics (F, K) |
| Original compression state (JPEG vs lossless) | compression-artifact analysis (H) |

So the audit extracts all of this into **small, permanent tables** that survive
deletion. Each analysis below is tagged:

- **[raw]** must run before the dataset's raw data is deleted.
- **[prepared]** can run any time on the prepared data and the saved tables.

## Scope

| Modality | Datasets (as wired in `training/train.py`) |
|---|---|
| Ultrasound | BUS-BRA, BUSI, BUSC (`us-dataset`), BrEaST-Lesions, plus `ultrasound/BUS` and `raw/oasbud` (present but not wired in — audit to decide) |
| Mammography | MIAS, CMMD, CDD-CESM (subtracted + low-energy), BCS-DBT (validation release) |
| MRI | BreaDM (`cls/img9Se` + `seg`/`seg3D`), BreastDCEDL-ISPY2 |

Histopathology (BreCaHAD) and the non-imaging sets in `datasets/raw/` are out of scope.

---

# Part 1 — Inventory and pixel-level audit (paper method, extended)

### A. Claimed vs. verified audit table [raw] (paper Tables 3–11)

Per dataset, three columns:
- **reported:** from the dataset paper / TCIA page; seed from [breast_cancer_datasets.md](breast_cancer_datasets.md).
- **on disk:** the raw data.
- **used by the loader:** after this repo's filtering.

Rows:
- total images; patients; benign / malignant / normal (or other)
- images per patient (min / median / max)
- masks or boxes available
- devices / vendors; radiologists
- histopathological confirmation; BI-RADS available; license
- duplicate groups (from C)
- images with markers / Doppler / text (from E)
- multi-lesion candidates; unique image sizes

Give each discrepancy a one-line explanation (e.g. "5 annotated CDD-CESM images
missing on disk").

### B. Basic image-quality metrics [raw] (paper §2.1), one row per image

- **Metrics:**
  - mean brightness, contrast (std), sharpness (variance of Laplacian)
  - entropy (256-bin histogram), dynamic range (max−min)
  - bit depth (stored bits, not dtype)
  - width, height, aspect ratio, file format / compression
- **Compute on native pixels and on the prepared image.** Native means the DICOM after VOI LUT, before 8-bit conversion. Comparing the two quantifies the conversion loss. If the uint8 conversion routinely clips the native dynamic range, store prepared mammography as uint16 PNG.
- **Mammography:** compute within a breast mask (Otsu + largest component), so the black background doesn't dominate every metric.
- **MRI:** per phase and per slice; also voxel spacing, slice thickness, number of phases, matrix size, field strength.
- **Report** per-dataset distributions as violin plots coloured by dataset.

### C. Duplicate and near-duplicate detection [raw for full frames; prepared otherwise]

- **Hashes:** aHash as in the paper, plus pHash and dHash. aHash alone gives many false positives on low-texture US images.
- **Embeddings:** cosine similarity of pretrained-model embeddings (see L). Embeddings outperform hashes for medical near-duplicates [A1], especially for re-cropped, rescaled or re-encoded copies, which is how public US sets re-host each other's images.
- **Candidate pairs:** Hamming ≤ 5 on any hash, or embedding cosine above a calibrated threshold. Confirm with SSIM > 0.95 at a common size.
- **Within dataset:** duplicate groups (the paper reports 81 groups in BUSI).
- **Across datasets:** especially ultrasound.
- **Across our splits:** run the actual split functions (`_*_frame`, `_group_kfold_frame`, k-fold) with the default seed. Report duplicate pairs that straddle train/val/test. This is the most project-relevant output, because leakage inflates the reported numbers [A2].
- **Label conflicts:** duplicate groups that contain both benign and malignant.
- **Output:** `duplicates.csv` (group id, members, distances, SSIM, labels, splits) and a contact sheet per group for manual review.

### D. Outlier detection [prepared, using B's tables]

- **Univariate (paper):** z-score each metric in B within its dataset; flag |z| > 3.
- **Multivariate:** isolation forest on all B + H metrics, and kNN distance in embedding space (L). These catch images that are unusual in combination but not on any single metric — wrong modality, scout views, screenshots, corrupted files.
- **Review:** render a contact sheet of flagged images. Tag each by hand as *keep / exclude / fix* in `outliers_review.csv` (tracked).

### E. Modality-specific artifact detection [raw]

**Ultrasound (paper §2.1):**
- **Color Doppler.** The paper's rule (fraction of R>150 + B>150 pixels > 2%) also fires on bright grayscale pixels. Use HSV saturation instead: the fraction of pixels with S > 0.3 and V > 0.2. Calibrate the threshold on a handful of hand-labeled images.
- **Measurement markers / calipers.** Canny + probabilistic Hough (> 10 strong lines, as in the paper), plus template matching for the "+" and "×" caliper glyphs.
- **Burnt-in text / annotations.** Colored overlays via HSV (red/yellow/green). White text via thresholded small connected components outside the scan cone.
- **Scan-cone geometry.** Detect the sector/linear acquisition region. Its shape and position identify the scanner and preset, which makes it a likely shortcut.
- **Split-screen / dual-panel images, and axillary images.** Flag manually from the contact sheets.
- **Multi-lesion candidates.** Connected components of the GT mask, where masks exist.

**Mammography / DBT:**
- view (CC/MLO) and laterality balance per class
- burnt-in laterality/view labels; pectoral muscle presence; implants
- film artifacts and scanner labels (MIAS); MONOCHROME1 vs 2
- vendor/model mix (CMMD, BCS-DBT headers)
- CDD-CESM: whether every low-energy image has its subtracted pair

**MRI:**
- DCE phase count and timing per patient; how the 9 phases were sampled (`preprocess_ispy2_mri.py`)
- slices per patient; tumor bbox size distribution; vendor / field strength
- motion and subtraction misregistration (bright rims in subtraction images)

### F. Lesion and annotation statistics [raw for BreaDM masks and BCS-DBT full frames]

Applies where masks or boxes exist: BUS-BRA, BUSI, BrEaST, MIAS `crop_box`, BCS-DBT boxes, BreaDM/ISPY2 masks.
- **Statistics:** lesion area as a fraction of the image, bbox size, lesions per image.
- **Center bias:** a heatmap of lesion centroid position per dataset × class.
- **Grad-CAM link:** cross-reference with the existing Grad-CAM false-positive outputs (`docs/gradcam_fp_manifest.json`). Do false positives cluster on markers, text, or edges?

### G. Label and patient structure [prepared + headers]

- class balance at image level vs. patient level
- images per patient by class; patients with mixed labels
- confirm the grouping key each split function uses (MIAS patient, BrEaST `CaseID`, CMMD `PatientID`, …)
- check whether any dataset is currently split at image level when it should be split at patient level

---

# Part 2 — Signal, texture and annotation analysis

### H. Noise, contrast and frequency characterization [raw]

- **SNR and CNR with masks.** Measure lesion vs. a surrounding ring of tissue: CNR = |μ_lesion − μ_ring| / σ_ring. This shows how visible lesions are per dataset and per class. Datasets where malignant lesions are systematically higher-contrast are easier and won't transfer.
- **MRI quality metrics (MRIQC-style).** SNR, CNR, entropy-focus criterion (EFC), foreground-background energy ratio (FBER), and smoothness (FWHM), adapted to breast DCE. Literature reference ranges come mostly from brain MRI [B1], so use them comparatively across datasets rather than as absolute pass/fail.
- **Ultrasound speckle statistics.** Speckle SNR (μ/σ in homogeneous regions; ≈1.91 for fully developed speckle) and Nakagami/Rayleigh fits. Deviations reveal post-processing (speckle reduction, compounding) that differs between scanners and datasets.
- **Radial power spectrum.** Exposes resampling and interpolation history (missing high frequencies), denoising, and sharpening.
- **JPEG blockiness.** 8×8-grid discontinuity measure. CDD-CESM and parts of BreaDM ship as JPEG. If compression differs by class or source, it's a shortcut.
- **Mean and std images per dataset × class.** The average of all resized images, and the per-pixel std. Simple but very revealing: fixed-position text, markers, cone edges, or padding show up immediately, and so does any difference between the benign and malignant mean images that isn't the lesion.
- **Intensity histograms per dataset × class,** overlaid. This shows the cross-dataset intensity shift directly.

### I. No-reference perceptual image quality [prepared]

- **Metrics:** BRISQUE and NIQE, via `pyiqa`.
- **Caveat:** these are trained on natural images and correlate only partially with radiologist judgments on medical images [B2, B3]. Use them for ranking and outlier-finding within a modality, not as absolute quality scores.

### J. Texture and radiomics features [prepared, needs masks/boxes]

- **Features:** IBSI-compliant first-order, shape, GLCM, GLRLM and GLSZM features on the lesion ROI, via PyRadiomics [B4].
- **Benign vs. malignant within each dataset:** effect sizes (Cohen's d, AUC per feature). Shows which handcrafted cues separate the classes, and whether that agrees across datasets.
- **Same class across datasets:** shows the scanner/site effect. Fit ComBat harmonization [B5] on the radiomics features and see how much of the cross-dataset variance it removes. A large reduction means the datasets differ mostly by acquisition, not by population.
- **Baseline:** a cheap handcrafted baseline (logistic regression on radiomics) to put the CNN numbers in context.

### K. Annotation quality [raw for BreaDM masks; prepared otherwise]

- **Mask style:**
  - tight vs. loose
  - polygonal vs. freehand (boundary smoothness, convexity)
  - mask area relative to the box
  - mask leaking outside the scan cone or breast
  - mask–image size mismatches
- **Inter-dataset annotation conventions.** E.g. BUS-BRA vs. BUSI vs. BrEaST masks. This matters because `BusbraTransform` crops to the mask: a looser mask convention means more context and a different input distribution.
- **Missing or empty masks;** masks on "normal" images.

---

# Part 3 — Representation-level analysis

### L. Embedding extraction [prepared]

Extract embeddings once per image and reuse them for C, D, M, N, O:
- an ImageNet backbone (ResNet-50 / DINOv2)
- a medical vision encoder already used in this repo (BiomedCLIP; optionally the MedGemma vision tower)
- the project's own trained ResNet-18 checkpoints

Cache them to `analysis/<dataset>/embeddings_<model>.npy`.

### M. Dataset shift and distance between datasets [prepared]

- **Dataset distances:** Fréchet distance (FID-style, on the embeddings above) and MMD between every pair of datasets within a modality, and between train and test splits of each dataset. This gives a distance matrix that predicts which cross-dataset transfers will fail.
- **Shift typing.** Separate prevalence shift (class balance) from acquisition shift (image appearance) and annotation shift, following the shift-identification framework of [C1]. Each type has a different remedy: reweighting, harmonization or augmentation, relabeling.
- **Visualization:** UMAP / t-SNE of embeddings coloured by dataset, by label, and by vendor.

### N. Source identifiability ("Name That Dataset") [prepared]

- **Test:** train a linear probe on the embeddings to predict *which dataset* an image came from [C2].
- **Why it matters:** high accuracy means a combined model can tell sources apart. That only becomes a shortcut if class balance differs by source, which it does here (e.g. ISPY2 is malignant-only). A cross-dataset mammography study recently showed exactly this "dataset-origin signature" shortcut [C3].
- **Generalization:** apply the same probe to any metadata attribute (vendor, view, age bucket). Rank attributes by how *detectable* they are and how *predictive of the label* they are; attributes high on both are the shortcut risks. This is the G-AUDIT framing [C4].

---

# Part 4 — Label quality and shortcut risk

### O. Label-error detection [prepared]

- **Confident learning** [D1] (`cleanlab`) on *out-of-fold* predicted probabilities. The k-fold machinery already exists (`_run_busbra_kfold`, `_run_generic_kfold`), so every image gets a prediction from a model that didn't train on it. This flags likely mislabeled images and estimates the noise rate per dataset.
- **kNN label consistency in embedding space.** Images whose nearest neighbours mostly carry the other label.
- **Dataset cartography** [D2]. Log each training sample's predicted probability every epoch. Plot confidence vs. variability to split the data into easy / ambiguous / hard-to-learn regions. Hard-to-learn samples are disproportionately label errors. Ambiguous ones are useful for curriculum learning or for review.
- **Review:** cross-reference flags with duplicates (C), artifacts (E) and outliers (D), then review by hand. Medical label noise is mostly instance-dependent and observer-driven [D3], so treat flags as candidates, never as auto-corrections.

### P. Hidden stratification and subgroup performance [prepared + headers]

- **Report performance per subgroup:**
  - vendor / scanner
  - view and laterality
  - lesion-size bin (from F)
  - breast density and age, where available (CMMD, CDD-CESM)
  - artifact present / absent (E)
  - mass vs. calcification (MIAS, CMMD)
- **Unlabelled subgroups:** cluster the embeddings within each class to find clinically meaningful subgroups that aren't labelled [D4]. High average AUC can hide a failing subgroup.

### Q. Learning curves (data sufficiency) [prepared]

- **Method:** train each single-dataset baseline on 10 / 25 / 50 / 100% of its training data, repeated over seeds.
- **What it tells you:** whether performance has plateaued. That directly answers whether downloading more data for a modality is worth the disk space, and which modality benefits most.

### R. Shortcut-risk probes [raw for full-frame variants]

These are cheap tests of whether labels can be predicted from things other than the lesion [E1, E2]:
- **Metadata-only model.** Logistic regression / gradient boosting on the B + E + H features (size, aspect, brightness, markers, Doppler, vendor, compression) predicting the label. AUC well above 0.5 means shortcut risk.
- **Background-only model.** Train on images with the lesion region masked out (mask or box filled with local mean/noise). If AUC is well above 0.5, the label leaks through context: markers, scanner, crop size, patient population.
- **Lesion-only model.** Train on a tight lesion crop. Compared with the full-image model, this shows how much the model relies on context.
- **Artifact counterfactuals.** Inpaint markers/text and compare predictions. This is a cheap version of the paper's clean/messy protocol.
- **Background Grad-CAM energy.** Quantify the fraction of Grad-CAM energy outside the lesion mask across all test images, rather than inspecting only failures.

### R′. Follow-up experiment (not a deletion blocker)

The paper's clean/messy protocol: train-clean/test-messy, train-messy/test-clean, and clean/clean. Cleaning uses an HSV mask, Telea inpainting and a median filter. The audit only needs to **record which images have overlays (E)** for this to be run later on prepared data. However, if inpainting will ever be wanted on full uncropped frames, generate those frames before raw deletion for datasets whose prepared form is cropped.

---

# Part 5 — Documentation

### S. Datasheet per dataset [prepared]

- **Write** a datasheet [F1] per dataset in `docs/dataset_audit.md`: motivation, composition, collection process, preprocessing, uses, distribution, maintenance.
- **Fill it from:** tables A–R plus the registry entry (storage plan §4).
- **Cross-check** against the issues catalogued in the medical-imaging datasets living review [F2]: label quality, shortcuts, missing metadata, duplicate releases.
- **LaTeX:** audit tables go into the report (`docs/latex/main.tex`), one per dataset, in the paper's format.

---

## Implementation status

Implemented (raw stage, runs on every dataset in `training/train.py`):

| Section | What's computed | Where |
|---|---|---|
| A | reported / on-disk / used-by-loader counts, by label, split, group; devices; DICOM vendors | `audit/report.py:summarize`, `audit/reported.py` |
| B | brightness, contrast, Laplacian-variance sharpness, entropy, dynamic range (native units and as fraction of stored bit range), bit depth, size/aspect, clipping — on native pixels, and on the prepared image where training reads a different file (CMMD, BCS-DBT: `prep_*` columns) | `audit/metrics.py:quality_metrics` |
| C | aHash/dHash/pHash candidates → SSIM ≥ 0.95 confirmation → duplicate groups; label conflicts; hold-out and k-fold leakage using the real split functions; cross-dataset duplicates per modality (`--cross`) | `audit/duplicates.py` |
| D | per-metric \|z\| > 3 and isolation forest; manual `outliers_review.csv` that later runs never overwrite | `audit/report.py:flag_outliers` |
| E | US: HSV-saturation Doppler (plus the paper's rule for comparison), paired '+' calipers, dotted measurement lines, burnt-in text lines, scan-region solidity, split-screen. Mammo: breast mask, burnt-in labels, implant suspicion, header laterality, photometric. MRI: per-channel means / enhancement ratio | `audit/artifacts.py` |
| F, K | lesion area, bbox, centroid, components, compactness, fill, mask-outside-scan-region, mask/image size mismatch, lesion-vs-ring CNR | `audit/metrics.py:lesion_metrics` |
| G | groups, images per group, groups split across hold-out splits | `audit/report.py`, `audit/duplicates.py` |
| H (part) | Immerkær noise σ, SNR, radial-spectrum high-frequency fraction and slope, JPEG blockiness, US speckle SNR; MRI case-level tumor volume, kinetic curve, enhancement ratio, washout, lesion CNR (ISPY2 NIfTI, BreaDM seg3D) | `audit/metrics.py`, `audit/volumes.py` |

Not yet: MRIQC-style EFC/FBER and Nakagami fits (H), and everything tagged
[prepared] (I, J, L–S).

Calibration notes from the first BUSI run (contact sheets reviewed by eye):
- The paper's Doppler rule (R>150 + B>150 > 2%) fires on 758/780 BUSI images
  and its marker rule (> 10 Hough lines) on 732/780 — both are dominated by
  ordinary bright tissue. They are kept as `paper_*` columns for comparison only.
- `marker_flag` (paired calipers or dotted measurement line) and `text_flag`
  were 24/24 correct on their review sheets; recall is not yet measured (one
  random sheet of 12 showed one missed two-letter label).

Run:

```bash
.venv/bin/python scripts/analyze_datasets.py --dataset all --cross --workers 10
```

Outputs land in `analysis/<dataset>/` (CSV/JSON tracked in git; parquet,
thumbnails and figures gitignored). Tests: `tests/test_audit.py`.

## Implementation outline

- **Raw script:** `scripts/analyze_datasets.py --dataset <name> --stage raw|prepared`.
  - One adapter per dataset yields `(sample_id, native_array, prepared_array, metadata dict)`.
  - Adapters reuse the existing metadata functions in `training/train.py`, so "used by the loader" is exactly what training sees.
  - `--stage raw` runs everything tagged [raw]. That is the deletion gate.
- **Model-based script:** `scripts/analyze_models.py` for O–R, which need training runs. It reuses `_train_model` and the k-fold runners, with a per-sample probability logging hook.
- **Code structure:** shared metric/hash/artifact/embedding modules; modality-specific checks in small plugins.
- **New dependencies** (add to `requirements.txt`): `opencv-python-headless`, `ImageHash`, `scikit-image`, `scikit-learn`, `pyiqa`, `pyradiomics`, `neuroCombat`, `cleanlab`, `umap-learn`. `pydicom` and `nibabel` are already in `.venv`.
- **Run order:**
  1. Smallest datasets first (ultrasound, MIAS), to tune thresholds by eye.
  2. Then CMMD, CDD-CESM, BCS-DBT, ISPY2, BreaDM.
  3. Within each: [raw] stage → deletion gate → [prepared] stages at leisure.

### Outputs (small, survive deletion)

```
analysis/<dataset>/
    per_image.parquet       # B + E + H + hashes + header fields, one row per raw image
    dicom_headers.parquet   # full non-pixel header dump (DICOM sets only)
    radiomics.parquet       # J
    embeddings_<model>.npy  # L
    duplicates.csv          # C
    outliers_review.csv     # D, manual keep/exclude/fix decisions (tracked)
    label_issues.csv        # O, flagged + reviewed
    summary.json            # table A values
    figures/                # distributions, contact sheets, mean images, UMAPs, centroid heatmaps
analysis/cross_dataset/     # M, N distance matrices, source-probe results
```

Git tracks the small CSV/JSON files under `analysis/`. Parquet files, embeddings and large figures are backed up with the prepared data (storage plan §5).

## Deletion-gate checklist per dataset ([raw] items only)

- [ ] A: table filled, discrepancies explained
- [ ] B, E, H: `per_image.parquet` (native + prepared metrics, artifact flags) and header dump saved
- [ ] C: full-frame hashes saved; duplicates reviewed; cross-split leakage reported (and fixed in the split code if found)
- [ ] F, K: mask/box statistics saved (BreaDM masks, BCS-DBT full frames)
- [ ] R: full-frame variants for background-only / counterfactual probes saved, if they'll be wanted
- [ ] Decision on uint8 vs uint16 for the prepared form
- [ ] Outputs backed up off-machine

## References

Source method:
- Siddique, Rada, Yap, Alam, Yuca, Arıbal. *Deep Learning for Breast Cancer Analysis: Classification and Segmentation on Ultrasound Datasets*. 2026 draft (`~/Desktop/Breast_Cancer_prediction_using_UltraSound.pdf`).

Duplicates and leakage:
- [A1] [Benchmarking Pretrained Vision Embeddings for Near- and Duplicate Detection in Medical Images](https://www.alphaxiv.org/abs/2312.07273), 2024.
- [A2] [Scores That Hold, Benchmarks That Leak: Measuring Dataset Contamination in Public Brain-Tumor MRI Classification](https://www.alphaxiv.org/abs/2610.00421), 2026.

Image quality, texture, harmonization:
- [B1] [Development of reference Image Quality Metrics for quantitative MRI research using MRIQC](https://consensus.app/papers/details/e816a5289ea650a385db0b6e651edcee/), Joshi et al., 2025. MRIQC itself: Esteban et al., *PLOS ONE* 2017.
- [B2] [Image Quality Assessment for Magnetic Resonance Imaging](https://consensus.app/papers/details/87d20ce554ad5b85a1dca525fb08f589/), Kastryulin et al., *IEEE Access* 2022.
- [B3] [A Brief Survey on No-Reference Image Quality Assessment Methods for Magnetic Resonance Images](https://consensus.app/papers/details/6df15085a39654c1a8f560531aa2654a/), Stępień et al., *J. Imaging* 2022.
- [B4] Zwanenburg et al., *The Image Biomarker Standardization Initiative* (IBSI), *Radiology* 2020 (arXiv:1612.07003).
- [B5] Fortin et al., *Harmonization of cortical thickness measurements across scanners and sites* (ComBat), *NeuroImage* 2018.

Dataset shift and source identifiability:
- [C1] [Automatic dataset shift identification to support safe deployment of medical imaging AI](https://www.alphaxiv.org/abs/2411.07940), 2024.
- [C2] Torralba & Efros, *Unbiased Look at Dataset Bias*, CVPR 2011.
- [C3] [Dataset-Origin Signatures and Shortcut Learning in Screening Mammography AI: A Cross-Dataset Case Study](https://www.alphaxiv.org/abs/2607.15416), 2026.
- [C4] [Detecting Dataset Bias in Medical AI: A Generalized and Modality-Agnostic Auditing Framework](https://www.alphaxiv.org/abs/2503.09969) (G-AUDIT), 2025.

Label quality and hidden stratification:
- [D1] [Confident Learning: Estimating Uncertainty in Dataset Labels](https://consensus.app/papers/details/68bd0974e8ca5a398f44e9eb25fa439c/), Northcutt et al., *JAIR* 2021.
- [D2] Swayamdipta et al., *Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics*, EMNLP 2020 (arXiv:2009.10795).
- [D3] [A survey of label-noise deep learning for medical image analysis](https://consensus.app/papers/details/2af6241e5ada59eda574b0c12acb9959/), Shi et al., *MedIA* 2024.
- [D4] Oakden-Rayner et al., *Hidden Stratification Causes Clinically Meaningful Failures in Machine Learning for Medical Imaging*, 2020 (arXiv:1909.12475).

Shortcut learning:
- [E1] [AI for radiographic COVID-19 detection selects shortcuts over signal](https://consensus.app/papers/details/0b362fd3f04850f58f4db42cb0fbc4be/), DeGrave et al., 2021.
- [E2] [Shortcut learning in medical AI hinders generalization: method for estimating AI model generalization without external data](https://consensus.app/papers/details/ef11742cd1875d7989c76d3cc5063b37/), Unnikrishnan et al., *npj Digit. Med.* 2024.

Documentation:
- [F1] Gebru et al., *Datasheets for Datasets*, CACM 2021 (arXiv:1803.09010).
- [F2] [In the Picture: Medical Imaging Datasets, Artifacts, and their Living Review](https://www.alphaxiv.org/abs/2501.10727), 2025.
