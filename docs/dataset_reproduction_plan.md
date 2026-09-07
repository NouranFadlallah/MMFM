# Dataset Reproduction Plan

This document records the paper-aligned preprocessing and training plans for the
locally available datasets. Dataset counts and paths refer to the current local
checkout. Exact paper metrics should only be compared after matching the paper's
split, target definition, preprocessing, and evaluation unit.

## Status and Scope

- `datasets/ultrasound/busbra`: BUS-BRA, 1,875 images from 1,064 patients, with image masks and `bus_data.csv`.
- `datasets/ultrasound/BUS`: likely the BUSI / Breast Ultrasound Images Dataset release, with `original/`, `GT/`, and `DatasetB.xlsx`. The local copy has 163 original PNGs, so completeness and the exact dataset variant must be verified before claiming BUSI reproduction.
- `datasets/ultrasound/us-dataset`: BUSC, 250 BMP images arranged as 100 benign and 150 malignant samples. The Mendeley release describes original 64 x 64 images and 128 x 128 preprocessing in the associated PDF-UNet work; it does not provide ground-truth masks.
- `datasets/ultrasound/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023`: BrEaST-Lesions USG, 256 clinical cases in the release paper; the local folder contains image/mask PNGs plus some `other` images.
- `datasets/mammography/mias`: mini-MIAS, 322 PGM images at 1024 x 1024 pixels, with labels and lesion coordinates in `Info.txt`.
- `datasets/MRI/BreaDM`: BreastDM, a DCE-MRI dataset. The local copy contains derived classification arrays (`cls/img9Se`, `cls/img17Se`, `cls/GLCM`, `cls/LBP`) and 3D segmentation arrays (`seg3D`), with official train/val/test structure.

The current training code is being used first as a single-dataset preprocessing
and paper-reproduction harness. Segmentation metrics and 3D-volume results require
separate dataset adapters and model heads. The multimodal fusion model is deferred
until the individual dataset pipelines have been verified.

## 1. BUS-BRA

### Source papers

- Dataset: [BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems](https://doi.org/10.1002/mp.16812)
- Classification recipe used for comparison: [Role of inter- and extra-lesion tissue, transfer learning, and fine-tuning in the robust classification of breast lesions](https://www.nature.com/articles/s41598-024-74316-5)

### Reproduction plan

1. Read `bus_data.csv`; map `Pathology=benign` to 0 and `malignant` to 1.
2. Pair `Images/bus_<id>.png` with `Masks/mask_<id>.png`.
3. Keep all views from the same `Case` in one split. Prefer the paper's official split if it is supplied; otherwise use a fixed patient/case-level split and report it.
4. Convert to grayscale and normalize intensities to `[0, 1]`.
5. Apply a 3 x 3 median filter for speckle reduction.
6. Use the mask to derive lesion-centered ROI crops. The paper also discusses erosion/dilation to form inter-lesion and extra-lesion tissue; implement those as separate ablations or input channels rather than silently treating them as the same image.
7. Resize to 224 x 224.
8. Training augmentation: horizontal/vertical flips, rotations up to +/-10 degrees, and zoom up to 10 percent. Do not augment validation/test data.
9. Train ImageNet-transfer CNNs, at minimum ResNet-18, EfficientNet-B0, and GoogLeNet. Use Adam at 1e-4, categorical cross-entropy, 20-50 epochs, and early stopping on validation loss.
10. Report accuracy, sensitivity/recall for malignant lesions, specificity, precision, F1, ROC-AUC, confusion matrix, and the split unit. Add Grad-CAM only after the classifier protocol is fixed.

### Local implementation status

The current code implements steps 1-8 and supports a true single-backbone mode.
Single-mode runs do not apply modality dropout. The local BUS-BRA run used a 90/10
case-level train/validation split, not a paper official split, so its result is a
preprocessing and training-pipeline check rather than a paper reproduction.

## 2. BUS / BUSI

### Source paper

- [Dataset of breast ultrasound images](https://pubmed.ncbi.nlm.nih.gov/31867417/), Al-Dhabyani et al., 2020.

### Reproduction plan

1. Confirm whether `datasets/ultrasound/BUS` is the complete BUSI release or a subset. Reconcile the spreadsheet labels with the image and `GT` filenames and verify the expected normal/benign/malignant class counts.
2. Preserve the three-class target (normal, benign, malignant) for the primary reproduction. Do not collapse normal into benign without a separate experiment.
3. Pair each original image with its ground-truth mask where available. Keep patient or lesion identity together across splits; never split augmented/mask variants independently.
4. Convert PNGs to grayscale, normalize to `[0, 1]`, and resize to the model input size. Use masks for a lesion-ROI ablation, but also report the full-image baseline because the original dataset paper is a dataset description rather than one canonical classifier recipe.
5. Compare a pretrained ResNet-18 baseline with the paper-style CNN candidates used in later BUSI studies: EfficientNet-B0 and GoogLeNet. Use class-weighted cross-entropy or a documented balanced sampler if the three classes are imbalanced.
6. Use fixed train/validation/test partitions or stratified patient-level cross-validation. Record the exact image count after filtering and whether multiple masks per image become multiple samples.
7. Report macro-F1, balanced accuracy, per-class recall, and one-vs-rest ROC-AUC in addition to accuracy.

### Blocking check

The local folder currently contains 163 original images, which does not match the
commonly cited 780-image BUSI release. Resolve this before interpreting results.

### Local implementation status

The runner supports `--dataset busi --single-mode`. It reads `DatasetB.xlsx`,
matches numbered files in `original/` and `GT/`, maps the local Benign/Malignant
labels, applies the mask-ROI preprocessing, and uses a stratified split for this
local subset. Modality dropout remains disabled. The local run is therefore a
pipeline check and cannot be compared directly with results reported on the full
BUSI release.

## 3. BUSC

### Source dataset and paper

- [BUSC Dataset](https://data.mendeley.com/datasets/vckdnhtw26/1), Mendeley Data, 2023.
- [PDF-UNet: A semi-supervised method for segmentation of breast tumor images](https://doi.org/10.1016/j.eswa.2023.119718), Iqbal and Sharif, 2023.

### Reproduction plan

1. Read the `originals/benign` and `originals/malignant` directory labels.
2. Use all 250 BMP images for binary classification: 100 benign and 150 malignant.
3. Preserve the original `64 x 64` resolution for the dataset baseline, and run a separate `128 x 128` experiment to match the PDF-UNet release description.
4. Convert to grayscale float tensors and normalize intensities using training-set statistics or the paper's exact normalization once recovered.
5. Use only classification augmentation for a classifier. Do not invent segmentation masks: the dataset release does not include ground-truth masks; masks in the associated work were annotated separately by a radiologist.
6. Use a stratified train/validation/test split with a fixed seed and report the exact image counts. There is no patient identifier in the local files, so treat image-level splitting as a limitation.
7. Train a single pretrained ResNet-18 baseline first, then reproduce PDF-UNet separately with its semi-supervised segmentation protocol if the radiologist-annotated masks are obtained.
8. Report accuracy, balanced accuracy, sensitivity, specificity, F1, and ROC-AUC for classification; report Dice/IoU only for the separately acquired segmentation annotations.

### Local status

The local folder is a complete 250-image BUSC classification set with explicit
folder labels. It is distinct from BUSI and has no masks, so its classification
baseline can be run now, while PDF-UNet segmentation reproduction is blocked on
the missing annotations and the original paper training details.

## 4. BrEaST-Lesions USG

### Source paper

- [Curated benchmark dataset for ultrasound based breast lesion analysis](https://www.nature.com/articles/s41597-024-02984-z), Pawlowska et al., 2024.

### Reproduction plan

1. Build a manifest from each `caseNNN.png` and its `caseNNN_tumor.png` mask. Treat `caseNNN_other*.png` as additional images only after confirming their role in the release metadata.
2. Preserve the dataset's lesion-level benign/malignant labels and exclude cases whose target is unavailable. Keep all images from one case in the same fold.
3. Establish two tasks: segmentation and classification. For segmentation, resize image and mask together with nearest-neighbor interpolation for masks. For classification, use the full-image baseline and a mask-derived lesion ROI baseline.
4. Normalize grayscale intensities to `[0, 1]`, resize to the selected CNN input size, and apply only train-time flips, small rotations, and scale/zoom transformations. Apply identical geometric transforms to image and mask.
5. Train U-Net or an attention U-Net for segmentation with Dice plus binary cross-entropy (or the exact loss from the selected paper), and report Dice, IoU, sensitivity, and specificity.
6. Train a pretrained ResNet-18 or EfficientNet-B0 classifier on full images and ROI crops. Use patient/case-level cross-validation because the dataset is small.
7. Report confidence intervals across folds; a single random split is too unstable for a small benchmark.

### Local implementation status

The local release contains 256 base case images, 252 `_tumor.png` segmentation
masks, and a small number of `_other`/`_other1` additional image views. These
suffixes describe image/mask roles, not benign/malignant diagnosis labels. The
official clinical workbook now supplies `Classification` labels and identifies
the tumor masks. The runner supports a metadata-backed single-classifier baseline
using the tumor ROI, while `_other*` masks remain reserved for segmentation work.
The local split currently resolves to 227 training and 25 validation cases; it is
not an official paper split.

## 5. MIAS / mini-MIAS

### Source paper

- Dataset: [The Mammographic Image Analysis Society Digital Mammogram Database](https://doi.org/10.1016/0531-5131(94)90040-X), Suckling et al., 1994.
- Local metadata: `datasets/mammography/mias/all-mias/Info.txt`.

### Reproduction plan

1. Parse `Info.txt`; use `NORM` as normal and lesion severity `B/M` for benign/malignant classification. Keep abnormality type (`CIRC`, `SPIC`, `CALC`, `ASYM`, etc.) as metadata.
2. Treat paired left/right films as belonging to the same patient. Split by patient, not by filename. The local data has 322 images and 1024 x 1024 pixels at the distributed 200-micron resolution.
3. For lesion classification, crop a square patch around `(X,Y)` using the provided radius. Convert the documented bottom-left coordinate origin to the image-array coordinate convention before cropping. Keep a full-image baseline as an ablation.
4. For normal images with no lesion coordinate, use a reproducible full-breast crop or fixed central/padded crop. Document this choice because it materially changes the task.
5. Convert PGM to float grayscale, remove constant borders/padding if used, normalize per image or using training-set statistics, and resize lesion patches to 224 x 224. Do not use bicubic/nearest interpolation on labels or coordinates without adjusting them.
6. Use train-only augmentation appropriate for mammograms: horizontal flip only if laterality semantics are not being modeled, small rotations, translations, and modest scale changes. Avoid vertical flips.
7. Start with pretrained ResNet-18 and compare against a patch-based CNN. Use weighted cross-entropy or balanced sampling because malignant examples are rare.
8. Evaluate sensitivity, specificity, ROC-AUC, PR-AUC, balanced accuracy, and patient-level bootstrap confidence intervals. Do not report only image-level accuracy.

### Reproduction caveat

MIAS is small and the metadata has multiple abnormalities for some images. Decide
whether the unit is image, lesion, or patient before creating the manifest; the
published result cannot be reproduced until that unit is fixed.

### Local implementation status

The runner now supports `--dataset mias --single-mode`. It parses the local
`Info.txt`, converts the documented bottom-left lesion coordinates, extracts
coordinate-centered patches with a 1.5x radius crop, splits paired films by
patient, and disables modality dropout. The first GPU smoke run produced 109
training lesion patches and 10 validation samples.

## 6. BreastDM / BreaDM MRI

### Source paper

- [BreastDM: A DCE-MRI dataset for breast tumor image segmentation and classification](https://doi.org/10.1016/j.compbiomed.2023.107255), Zhao et al., 2023.

### Reproduction plan

1. Use the release's official train/validation/test folders rather than making a new random split. Keep all slices from one patient/volume together.
2. First reproduce classification using one official representation at a time: `img9Se` or `img17Se`. Treat `GLCM` and `LBP` as engineered-feature baselines, not interchangeable raw-image inputs.
3. Inspect each `.npy` shape, dtype, intensity range, and slice ordering. Confirm whether the arrays are raw slices, registered/selected slices, or feature maps from the release preprocessing.
4. For raw DCE inputs, reproduce the paper's sequence handling: register/subtract or otherwise derive the documented subtraction images, normalize intensities, and select the reported number of slices/phases. Do not regenerate these steps from the derived arrays unless the paper and release scripts agree.
5. For a 2D classifier, use the same slice aggregation rule as the paper (for example, per-slice logits followed by patient-level aggregation). A slice-level random split would leak patient information.
6. For segmentation, use the `seg3D` image/label arrays as 3D volumes. Match voxel spacing/orientation, foreground cropping, intensity normalization, patch/volume size, and augmentation from the paper; evaluate Dice/IoU at the volume or lesion level.
7. Begin with the paper's baseline architecture if available, then compare a 2D ResNet/EfficientNet classifier and a 3D U-Net-style segmenter only as controlled baselines. Use Adam and the paper's learning rate/schedule, batch size, epochs, and stopping rule after extracting them from the article or released code.
8. Report both slice-level and patient-level metrics only when both are defined, plus Dice/IoU for segmentation. Include the exact representation (`img9Se`, `img17Se`, GLCM, or LBP) in every run name.

### Blocking check

The local MRI folder contains derived arrays and official split directories but no
paper training script was found in the checkout. Before claiming reproduction,
recover the exact paper preprocessing and aggregation protocol from the authors'
release/code and verify that local arrays correspond to the published version.

### Local implementation status

The runner now supports `--dataset breamdm --single-mode` using the official
`cls/img9Se/train` and `cls/img9Se/val` directories. It preserves the nine slice
channels, resizes spatial dimensions to the model input, normalizes uint8 values
to `[0, 1]`, adapts ResNet-18's first convolution to nine channels, and disables
modality dropout. The local smoke run resolved 1,202 training and 117 validation
arrays. This is a 2D derived-array baseline; it is not yet the paper's full 3D
segmentation reproduction.

## Execution Order

1. Resolve BUSI completeness and recover BrEaST labels/metadata.
2. Add dataset-specific manifest builders and patient-level split tests.
3. Reproduce BUS-BRA single-backbone ResNet-18 first, because its masks and labels are already wired.
4. Add BUSI and BUSC classification baselines, then BrEaST classification and segmentation tasks.
5. Add MIAS lesion-patch classification with patient-level evaluation.
6. Add BreaDM 2D classification from the official derived arrays, then the 3D segmentation path.
7. Log every run to W&B with dataset, representation, split seed, preprocessing version, backbone, and checkpoint path. Keep modality dropout disabled for all single-dataset reproduction runs; introduce it only when the multimodal fusion experiment begins.
