# Public Dataset Intake and Reproduction Plan

## Purpose

This plan is a research and intake checklist for datasets that are not yet in
the local checkout. The objective is to download each approved dataset later,
reproduce the source study or an explicitly selected canonical benchmark, and
understand its labels, preprocessing, and leakage risks before including it in
the multimodal study. A dataset-description paper is not automatically a model
benchmark; where no canonical training recipe exists, this plan says so.

## Dataset Inventory

| Modality | Dataset | Intake status |
| --- | --- | --- |
| Mammography | VinDr-Mammo | Resolved public restricted-access release |
| Mammography | CSAW-CC | Resolved public catalogue record; access terms to confirm |
| MRI | I-SPY1 / ACRIN 6657 | Resolved TCIA collection; use separate expert-SEG release for Dice work |
| MRI | I-SPY2 | Resolved TCIA collection; cohort definition is critical |
| MRI | Breast-MRI-NACT-Pilot | Resolved TCIA collection |
| Ultrasound | UDIAT / Dataset B | Resolved public research release |
| Ultrasound | BUSIS | Resolved segmentation benchmark; download artifact must be confirmed |
| Ultrasound | Breast Ultrasound Videos | Unresolved dataset identity |
| Ultrasound | Medical Image Database (eight diagnoses) | Unresolved dataset identity |
| Ultrasound | Breast Ultrasound Lesions Dataset (three diagnoses) | Ambiguous alias; do not count separately |
| Ultrasound | BUSI / Breast Ultrasound Dataset | Resolved public release |
| Ultrasound | BUID / Breast Ultrasound Images Database | Resolved public release |

## Shared Intake Rules

1. Record the source URL, version, DOI, access terms, manifest/checksum, date,
   files, labels, and image-to-patient linkage before any preprocessing.
2. Preserve source files read-only. Create a versioned manifest that records the
   source identifier, patient/exam/case identifier, image/mask/box paths,
   target, split, and exclusion reason.
3. Split by patient or exam before augmentation and before extracting video
   frames. When identifiers are unavailable, report the image-level split as a
   leakage limitation rather than implying patient-level validity.
4. Reproduce the source paper's cohort, target, preprocessing, model, and
   evaluation unit before comparing a metric. Run our ResNet-18/U-Net baselines
   as separate, clearly labelled experiments.
5. Do not combine datasets until overlap, label semantics, acquisition units,
   and licences have been checked.

## Mammography

### VinDr-Mammo

- **Download and access:** [PhysioNet VinDr-Mammo v1.0.0](https://physionet.org/content/vindr-mammo/1.0.0/), DOI [10.13026/br2v-7517](https://doi.org/10.13026/br2v-7517). Create a PhysioNet account and sign the [data-use agreement](https://physionet.org/sign-dua/vindr-mammo/1.0.0/) before downloading. Its restricted health-data licence permits research/education use but restricts redistribution and re-identification.
- **Primary paper:** Nguyen et al., [VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography](https://doi.org/10.1038/s41597-023-02100-7), *Scientific Data*, 2023.
- **Data and intended tasks:** 5,000 full-field digital mammography exams (20,000 DICOM images; four CC/MLO views per exam), with breast BI-RADS 1-5, density A-D, and finding boxes/categories. The fixed release split is 4,000 exams for training and 1,000 for testing. Labels are radiologist consensus rather than pathology-confirmed outcomes. BI-RADS 2 findings are not lesion-box annotated and cross-view lesions are not linked.
- **Mostly used for:** exam/multi-view BI-RADS or density classification, multi-label finding classification, and finding-box detection. The paper recommends a screening framing of normal BI-RADS 1, benign BI-RADS 2, and recall BI-RADS 3-5.
- **Source preprocessing, model, and result:** this is a dataset-description paper, not a canonical training benchmark. It removes PHI/burned-in corner text and publishes DICOM/CSV labels, but specifies no image conversion, normalization, crop, augmentation, model, optimizer, split beyond the release partition, or model result. Do not attribute later detector or classifier scores to the release paper.
- **Reproduction sequence:** validate DICOM decoding and photometric interpretation (the release warns some images are not strictly DICOM compliant); join image, breast, and finding CSVs; retain the published exam split; choose exactly one target; apply any breast crop/windowing and intensity normalization only after documenting it; train per-view and view-aggregation baselines separately; report exam-level AUROC, recall/sensitivity, specificity, and calibration for classification, or lesion sensitivity/FROC for boxes.
- **Decision gate:** first confirm that the DUA and DICOM reader work locally. Do not call a BI-RADS classifier a cancer detector.

### CSAW-CC

- **Download and access:** [CSAW-CC (mammography) dataset record](https://doi.org/10.5878/45vm-t798). This DOI is the authoritative starting point; inspect its current Swedish National Data Service/Karolinska access workflow and licence before requesting data. Do not use unverified mirrors.
- **Primary source:** no standalone, peer-reviewed CSAW-CC release paper was confirmed. Cite the dataset record and the parent cohort paper: Dembrower, Lindholm, and Strand, [A multi-million mammography image dataset and population-based screening cohort for the training and evaluation of deep neural networks: the cohort of screen-aged women (CSAW)](https://doi.org/10.1007/s10278-019-00278-0), *Journal of Digital Imaging*, 2020.
- **Data and intended tasks:** Swedish screening case-control cohort described as 1,303 cancer cases plus 10,000 controls; the VinDr-Mammo comparison table reports 24,694 exams and 98,788 FFDM images. Visible cancer signs have pixel-level contours. It does not establish BI-RADS/density labels or exhaustive benign-finding annotations. Counts in later studies may differ because they define filtered subsets.
- **Mostly used for:** cancer detection, external validation, risk prediction, and reader/model evaluation, generally as a case-control cohort rather than a screening-prevalence dataset.
- **Source preprocessing, model, and result:** neither the catalogue record nor parent cohort paper supplies a canonical image preprocessing pipeline, official CSAW-CC split, architecture, or benchmark score. A task-specific later paper must be selected before a result can be reproduced; do not transfer a risk-prediction metric into image-level cancer detection.
- **Reproduction sequence:** obtain the licensed manifest and label dictionary; determine image, screening round, prior, and patient links; freeze a cancer-detection target and patient/date-safe split; inspect DICOM pixel convention and contour geometry; reproduce the selected later study's cohort and preprocessing exactly; then compare our whole-image/breast-crop model at patient or exam level.
- **Decision gate:** access terms, actual manifest counts, contour semantics, and longitudinal grouping must be verified before implementation.

## MRI

### I-SPY1 / ACRIN 6657

- **Download and access:** [TCIA I-SPY1 collection](https://www.cancerimagingarchive.net/collection/ispy1/), DOI [10.7937/K9/TCIA.2016.HdHpgJLK). Download DICOM through TCIA Data Retriever and follow the [TCIA data-use policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/). The collection is CC BY 3.0. For expert tumor masks, also download [I-SPY1-Tumor-SEG-Radiomics](https://www.cancerimagingarchive.net/analysis-result/ispy1-tumor-seg-radiomics/), DOI [10.7937/TCIA.XC7A-QT20].
- **Primary sources:** Hylton et al., [Locally Advanced Breast Cancer: MR Imaging for Prediction of Response to Neoadjuvant Chemotherapy--Results from ACRIN 6657/I-SPY TRIAL](https://doi.org/10.1148/radiol.12110748), *Radiology*, 2012; expert-segmentation release paper Chitalia et al., [I-SPY1 Tumor Segmentation and Radiomics Data](https://doi.org/10.1038/s41597-022-01555-4), *Scientific Data*, 2022.
- **Data and intended tasks:** TCIA has 222 subjects/847 studies of longitudinal, unilateral sagittal DCE-MRI plus T2 and clinical/outcome data. It contains response, pCR-related, and survival variables. The base `SEG` objects are rectangular VOI/thresholded functional-tumor-volume (FTV) analysis masks, not expert structural tumor contours. The separate analysis-result release supplies one expert structural tumor-volume mask for each of 163 baseline patients.
- **Mostly used for:** neoadjuvant response/recurrence analyses using DCE-derived FTV, and, with the separate release, 3D tumor segmentation/radiomics.
- **Source preprocessing, model, and result:** the original trial papers perform SER/FTV measurements and statistical outcome modeling, not a released deep-learning train/test benchmark. Chitalia et al.'s later expert-SEG benchmark converts DICOM to NIfTI, bias-field corrects, resamples to 1 mm isotropic voxels, instance-z-scores three DCE phases, and trains a residual 3D U-Net with multiclass Dice loss, ghosting/blur/noise augmentation, SGD, a triangular learning-rate schedule, and nested k-fold CV. Its median holdout Dice is 0.74.
- **Reproduction sequence:** select FTV replication or expert-STV segmentation; retain subject-level grouping across visits; inventory DICOM phase order/scaling/geometry; parse `SEG` with a DICOM-aware reader; use nearest-neighbour resampling for masks; reproduce the cited pipeline and nested CV before evaluating any local 3D U-Net.
- **Decision gate:** never describe FTV/VOI analysis masks as manual whole-tumor ground truth.

### I-SPY2

- **Download and access:** [TCIA I-SPY2 collection](https://www.cancerimagingarchive.net/collection/ispy2/), DOI [10.7937/TCIA.D8Z0-9T85]. It contains 719 patients/2,688 studies (about 1.75 TB) and uses TCIA Data Retriever under CC BY 4.0 and TCIA policy. For Imaging Cohort 1 comparisons, use the collection page's official 985-case combined manifest (719 I-SPY2 plus 266 ACRIN-6698 DWI cases), not I-SPY2 alone.
- **Primary sources:** TCIA release by Bernreuter et al. (2022), DOI above; Li et al., [Predicting breast cancer response to neoadjuvant treatment using multi-feature MRI: results from the I-SPY 2 TRIAL](https://doi.org/10.1038/s41523-020-00203-7), *npj Breast Cancer*, 2020.
- **Data and intended tasks:** prospective multi-centre DCE-MRI at baseline, early, mid-, and post-treatment, with raw and derived DCE, PE/SER maps, clinical/molecular/treatment labels, and pCR. Base DICOM `SEG` data encode FTV analysis components (thresholding, rectangular VOI, omit regions), not an exact public count of expert whole-tumor masks. DWI is not in the I-SPY2 collection.
- **Mostly used for:** pCR prediction and longitudinal DCE biomarker modeling. The later [BreastDCEDL-ISPY2](https://www.cancerimagingarchive.net/analysis-result/breastdcedl_ispy2/) analysis result supplies NIfTI, binary tumor masks, and a fixed 784/99/99 split for 982 cases, but is a separate derived product.
- **Source preprocessing, model, and result:** Li et al. used FTV, longest diameter, sphericity, and contralateral background parenchymal enhancement in 384 eligible participants; $PE=(S_1-S_0)/S_0$ and $SER=(S_1-S_0)/(S_2-S_0)$. Logistic regression with repeated five-fold CV and bootstrap CIs produced combined-feature pCR AUC 0.81 (95% CI 0.76-0.86), versus 0.79 for the best single feature. This is not an external deep-learning test score.
- **Reproduction sequence:** choose the paper's 384-case all-feature cohort or the later 982-case derived benchmark; preserve patient/timepoint grouping; replicate PE/SER/FTV formulas and fixed threshold logic for the former, or use the published split for the latter; validate raw/derived DICOM series and `SEG` bit semantics before conversion; report pCR AUC with the matching repeated-CV protocol.
- **Decision gate:** results from 719 I-SPY2-only subjects are not comparable to studies based on the 985-case combined imaging cohort.

### Breast-MRI-NACT-Pilot

- **Download and access:** [TCIA Breast-MRI-NACT-Pilot](https://www.cancerimagingarchive.net/collection/breast-mri-nact-pilot/), DOI [10.7937/K9/TCIA.2016.QHsyhJKy]. Version 3 has 64 patients/189 studies (about 19.5 GB); use TCIA Data Retriever and policy. The collection is CC BY 3.0.
- **Primary papers:** Partridge et al., [MRI Measurements of Breast Tumor Volume Predict Response to Neoadjuvant Chemotherapy and Recurrence-Free Survival](https://doi.org/10.2214/ajr.184.6.01841774), *AJR*, 2005; Li et al., [Invasive Breast Cancer: Predicting Disease Recurrence by Using High-Spatial-Resolution Signal Enhancement Ratio Imaging](https://doi.org/10.1148/radiol.2481070846), *Radiology*, 2008.
- **Data and intended tasks:** single-site longitudinal unilateral sagittal DCE-MRI of stage II/III invasive breast cancer with treatment and outcome XLS data. Derived packages include PE/SER maps, PE-early/FTV-like threshold segmentations, and breast-tissue segmentation. These analysis masks are not independently annotated whole-tumor contours.
- **Mostly used for:** DCE/SER tumor-volume biomarkers and response/recurrence prognosis, not a canonical deep-learning segmentation benchmark.
- **Source preprocessing, model, and result:** the source papers quantify MRI longest diameter, tumor volume, and SER-volume measures, then use univariate/multivariable Cox models. Partridge et al. analysed 62 patients: baseline MRI volume was the strongest univariate RFS predictor ($p=0.002$); baseline volume and final volume change were independent multivariable predictors ($p=0.005$ and $p=0.003$). Li et al. analysed 48 women and found low-/high-SER volumes prognostic. No source DL architecture, optimizer, augmentation scheme, held-out split, or Dice result exists.
- **Reproduction sequence:** reproduce the statistical endpoint first: identify the paper cohort, treatment visit, DCE phase and SER threshold, calculate volumes with source geometry, and fit the matching Cox models. A subsequent DL experiment must use a separately stated task and subject-level evaluation.
- **Decision gate:** do not use PE/breast analysis masks as manual lesion labels or claim a DL paper reproduction without selecting a later DL study.

## Ultrasound

### UDIAT / Dataset B

- **Download and access:** [author dataset page](https://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php).
- **Primary paper:** Yap et al., [Automated Breast Ultrasound Lesions Detection Using Convolutional Neural Networks](https://doi.org/10.1109/JBHI.2017.2731873), *IEEE Journal of Biomedical and Health Informatics*, 2018.
- **Data and intended tasks:** 163 B-mode images with lesion masks, commonly reported as 110 benign and 53 malignant. It is a lesion detection/segmentation resource with binary diagnostic metadata, not a three-class dataset.
- **Mostly used for:** CNN candidate-based lesion detection and binary lesion segmentation; later work also uses it for mask segmentation benchmarks.
- **Source preprocessing, model, and result:** the primary paper is a candidate-region CNN detection method, not a universal image-classification/segmentation recipe. Its detection score is tied to its proposal generation and evaluation, so it should be reproduced from the full paper rather than reduced to a standalone Dice or accuracy claim.
- **Reproduction sequence:** verify all 163 image/mask pairs and label counts; recover patient grouping if available; inspect native dimensions and mask encoding; reproduce candidate generation, CNN inputs, and detection metric from the paper; separately run a U-Net-like mask baseline with training-only intensity scaling/augmentation and Dice/IoU.
- **Decision gate:** the source does not provide an official split; no patient identifier means image-level results require a caveat.

### BUSIS

- **Download and access:** begin with Zhang et al.'s [BUSIS benchmark paper](https://doi.org/10.3390/healthcare10040729) and use its data-availability route. The 562-image benchmark has historically been author-distributed rather than a stable DOI archive; obtain the exact source files and permission before adding a downloader.
- **Primary paper:** Zhang et al., [BUSIS: A Benchmark for Breast Ultrasound Image Segmentation](https://doi.org/10.3390/healthcare10040729), *Healthcare*, 2022.
- **Data and intended tasks:** 562 ultrasound images with expert lesion masks, assembled for segmentation comparison. It is not a diagnosis-labelled classifier dataset.
- **Mostly used for:** lesion segmentation-method benchmarking.
- **Source preprocessing, model, and result:** the paper compares 16 segmentation methods rather than defining a single canonical model. Image preparation, folds, and metrics are method-specific; there is no single BUSIS architecture or score that can be treated as the official result.
- **Reproduction sequence:** obtain the authors' release; validate exactly 562 image/mask pairs, dimensions, and mask values; check for overlap with BUSI/UDIAT before pooling; recover the paper's folds; resize images bilinearly and masks nearest-neighbour; reproduce the selected method and report fold-wise Dice, IoU, sensitivity, and specificity.
- **Decision gate:** no immutable public artifact or canonical split has been verified, so freeze a checksum and fold manifest before experiments.

### Breast Ultrasound Videos

- **Download and primary paper:** no authoritative public download, DOI, or release paper could be matched to the name plus the stated frame-level rectangular-box annotation. It is an annotation description, not a unique dataset title.
- **Likely use:** frame-level lesion detection/tracking, normally with detector architectures such as Faster R-CNN or YOLO-family models. No result is defensible until the underlying release and paper are identified.
- **Required discovery before intake:** obtain the original DOI/owner URL; licence; video/patient/case count; source frame rate; box coordinate schema; target taxonomy; and paper-defined split. Split by video/patient before sampling frames, transform boxes with every geometric augmentation, and assess at video/patient level rather than treating adjacent frames as independent.
- **Decision gate:** do not download an unproven mirror or enter this as a study dataset until provenance is supplied.

### Medical Image Database (eight diagnoses)

- **Download and primary paper:** unresolved. This is not a canonical public breast-ultrasound dataset title. Searches commonly resolve similarly named resources to mammography databases (for example, OMI-DB/OPTIMAM), which do not match the requested modality.
- **Likely use, preprocessing, model, and result:** none can be assigned without the original DOI/URL and label dictionary. An eight-label vocabulary may represent diagnoses, imaging findings, BI-RADS, or another task, each requiring a different target and evaluation.
- **Required discovery before intake:** require the release URL, owner, licence, modality confirmation, unit of analysis, exact eight-class mapping, annotations, and source paper. Then define patient-safe splitting, class imbalance treatment, and macro-F1/per-class recall as baseline reports.
- **Decision gate:** leave it out of the download queue until identity is resolved.

### Breast Ultrasound Lesions Dataset (three diagnoses)

- **Download and primary paper:** ambiguous alias, not an independently verified release. The phrase often denotes UDIAT Dataset B, which is binary benign/malignant. The stated three diagnoses instead match BUSI: normal, benign, and malignant.
- **Likely use, preprocessing, model, and result:** no separate recipe/result exists. If it is UDIAT, follow the UDIAT detection/segmentation protocol above. If it is BUSI, follow the BUSI three-class plan below.
- **Required discovery before intake:** recover the original URL/DOI or compare file provenance and metadata. Do not count it as a third dataset or merge it until identity is fixed.
- **Decision gate:** prevent duplicate inclusion and a binary-versus-three-class label error.

### BUSI / Breast Ultrasound Dataset

- **Download and access:** [Mendeley Data BUSI release](https://data.mendeley.com/datasets/wmy84gzngw/1), DOI [10.17632/wmy84gzngw.1](https://doi.org/10.17632/wmy84gzngw.1).
- **Primary paper:** Al-Dhabyani et al., [Dataset of Breast Ultrasound Images](https://doi.org/10.1016/j.dib.2019.104863), *Data in Brief*, 2020.
- **Data and intended tasks:** 780 images from 600 women: 437 benign, 210 malignant, and 133 normal. Lesion images have binary masks. Quality defects/duplicates reported in public copies are data-quality concerns, not an extra label.
- **Mostly used for:** three-class image diagnosis and binary lesion segmentation. These are separate tasks with different eligible image sets and metrics.
- **Source preprocessing, model, and result:** the primary paper is a dataset release with no canonical preprocessing, architecture, split, or model metric. Later scores are not comparable unless their split, cleaning, and target match exactly.
- **Reproduction sequence:** download the DOI version; inventory images, mask-file association, dimensions, duplicates/near-duplicates, and malformed records; preserve patients where linkage exists; create a documented stratified split or repeated cross-validation; fit normalization and augment only training data; train three-class classifier reporting macro-F1, balanced accuracy, per-class recall, and AUROC, and a separate segmentation model reporting Dice/IoU.
- **Decision gate:** a random image split can exaggerate results because public repackagings may contain duplicates and patient links are incomplete.

### BUID / Breast Ultrasound Images Database

- **Download and access:** [QAMEBI Breast Ultrasound Image Database](https://qamebi.com/breast-ultrasound-images-database/), with separate benign and malignant archives.
- **Primary paper:** Abbasian Ardakani et al., [An open-access breast lesion ultrasound image database: Applicable in artificial intelligence studies](https://doi.org/10.1016/j.compbiomed.2022.106438), *Computers in Biology and Medicine*, 2023.
- **Data and intended tasks:** histologically proven benign and malignant lesions paired with radiologist-defined masks. It supports binary diagnosis and lesion segmentation.
- **Mostly used for:** binary classification and mask segmentation, often as a complementary external cohort to BUSI/UDIAT.
- **Source preprocessing, model, and result:** the release paper establishes the data resource but does not specify a universal split, DL model, preprocessing protocol, or headline benchmark result. Treat later QAMEBI model scores as study-specific.
- **Reproduction sequence:** download both official archives; inventory image/mask pairing, shape, and binary encoding; inspect released patient linkage; use patient-level partitioning where possible, otherwise label the limitation; resize masks nearest-neighbour; compare binary classifier AUROC/F1 and segmentation Dice/IoU using a frozen split.
- **Decision gate:** confirm the page's current licence/redistribution terms and available patient-level metadata before publication-facing experiments.

## Download and Reproduction Order

1. Request VinDr-Mammo and CSAW-CC access; capture their licences and manifests.
2. Download UDIAT, BUSI, and BUID first because they are compact and expose the image/mask validation workflow needed for later ultrasound intake.
3. Obtain BUSIS directly from its authors and freeze the exact artifact before considering pooled segmentation experiments.
4. Download I-SPY1 plus I-SPY1-Tumor-SEG-Radiomics for the first MRI segmentation reproduction; only begin I-SPY2 after allocating DICOM and derived-data storage.
5. Download Breast-MRI-NACT-Pilot for a small statistical DCE/SER reproduction, not an unsupported DL benchmark.
6. Hold the video, eight-diagnosis, and ambiguous lesions entries pending primary provenance. They are research leads, not approved datasets.

## Exit Criteria Before Study Inclusion

- A source-versioned manifest joins every training item to its target and split.
- The source task is reproduced or documented as impossible because the original split/code/data are unavailable.
- A local baseline uses the same unit of evaluation and reports leakage controls.
- Licence, consent/access restrictions, target meaning, and dataset overlap are documented.
- Only then may the dataset enter a cross-dataset or multimodal experiment.