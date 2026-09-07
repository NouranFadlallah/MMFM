# Breast Cancer Datasets — Public Inventory

A comprehensive inventory of publicly available breast-cancer datasets across every
data type the field uses: mammography, ultrasound, MRI, CT and less-common
imaging modalities, pathology/histopathology, and clinical/genomic/biomarker
data. This is a research-lead intake list for future dataset acquisition — it
is broader than [dataset_reproduction_plan.md](dataset_reproduction_plan.md)
(what's locally implemented) and [public_dataset_intake_plan.md](public_dataset_intake_plan.md)
(the earlier, imaging-only intake list this supersedes in scope).

## Methodology and how to read this

Compiled September 2026 via parallel web research (one pass per modality),
citing primary papers and verifying access links against their live hosts
where the fetch tooling allowed it. Every entry lists:

- **Modality/specimen specifics** — acquisition detail (view, magnification, sequence, etc.)
- **Size & labels** — patient/image counts and what's annotated
- **Citation** — the primary dataset paper, with DOI/link
- **Access** — the actual data host and any access restriction (open download vs. DUA/application/registration)
- **License** — as stated by the source, where findable

Items that could not be independently verified (dead links, JS-rendered pages,
no confirmed primary source) are marked **[UNVERIFIED]** rather than silently
included as fact — treat those as leads to confirm before depending on them.
A dataset name appearing in two sections (e.g. TCGA-BRCA imaging vs.
TCGA-BRCA clinical/genomic) reflects that the same cohort has genuinely
distinct public releases for different data types; they are cross-referenced,
not duplicated by mistake.

**Before using any dataset**: confirm current license/access terms directly
at the source — data-sharing terms, hosting, and access processes change
independently of this document.

---

## Table of Contents

1. [Mammography](#1-mammography)
2. [Ultrasound](#2-ultrasound)
3. [MRI](#3-mri)
4. [CT & Less-Common Imaging Modalities](#4-ct--less-common-imaging-modalities)
5. [Pathology / Histopathology](#5-pathology--histopathology)
6. [Clinical, Genomic & Biomarker Data](#6-clinical-genomic--biomarker-data)
7. [Cross-Cutting Notes, Gaps & Naming Disambiguation](#7-cross-cutting-notes-gaps--naming-disambiguation)

---

## 1. Mammography

**BCDR (Breast Cancer Digital Repository)**
- Modality: Film mammography (BCDR-FM) and full-field digital mammography (BCDR-DM), MLO+CC views; also paired ultrasound
- Size: 1,734 patients total — BCDR-FM: 1,010 patients; BCDR-DM: 724 patients, 1,042 studies, 3,612 MLO/CC images, 452 lesions, 818 manual segmentations, BI-RADS classified
- Citation: Moura, D.C. et al., "BCDR: A Breast Cancer Digital Repository" ([ResearchGate](https://www.researchgate.net/publication/258243150_BCDR_A_BREAST_CANCER_DIGITAL_REPOSITORY))
- Access: http://bcdr.ceta-ciemat.es (mirror http://bcdr.inegi.up.pt) — request-based
- License: Not clearly stated; access by request
- **[UNVERIFIED]** — both known hosts were unreachable at check time; several 2023-2025 surveys describe it as discontinued/request-only. Verify live status before relying on it.

**CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**
- Modality: Digitized screen-film mammography (scanned DDSM films → DICOM); CC+MLO; masses and calcifications
- Size: 1,566 patients (6,671 DICOM patient IDs), 10,239 images, 163.6 GB; pathology-confirmed benign/malignant, updated ROI segmentations and bounding boxes
- Citation: Lee, R.S. et al., "A curated mammography data set for use in computer-aided detection and diagnosis research," *Scientific Data* 4, 170177 (2017). DOI: [10.1038/sdata.2017.177](https://doi.org/10.1038/sdata.2017.177). Data DOI: [10.7937/K9/TCIA.2016.7O02S9CY](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629), NBIA Data Retriever; also mirrored on Kaggle/TensorFlow Datasets
- License: CC BY 3.0

**CDD-CESM (Categorized Digital Database for Contrast-Enhanced Spectral Mammography)**
- Modality: Contrast-enhanced spectral mammography (dual-energy) — low-energy + recombined/subtracted images, CC+MLO
- Size: 326 patients, 2,006 images, 1.5 GB; BI-RADS, mass/calcification/architectural distortion/asymmetry annotations, manual segmentations; 751 normal images
- Citation: Khaled, R. et al., "Categorized contrast enhanced mammography dataset for diagnostic and artificial intelligence research," *Scientific Data* 9, 122 (2022). DOI: [10.1038/s41597-022-01238-0](https://doi.org/10.1038/s41597-022-01238-0). Data DOI: 10.7937/29kw-ae92
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611) via Faspex/Aspera
- License: CC BY 4.0

**CMMD (Chinese Mammography Database)**
- Modality: Digital mammography, biopsy-confirmed; molecular subtype for a subset
- Size: 1,775 patients, 1,775 studies, 5,202 images, 22.9 GB; molecular subtype for 749 patients
- Citation: Cui, C. et al., "An Online Mammography Database with Biopsy Confirmed Types," *Scientific Data* 10, 174 (2023). DOI: [10.1038/s41597-023-02025-1](https://www.nature.com/articles/s41597-023-02025-1). Data DOI: [10.7937/tcia.eqde-4b16](https://doi.org/10.7937/tcia.eqde-4b16)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508), NBIA Data Retriever
- License: CC BY 4.0

**CSAW-CC (Cohort of Screen-Aged Women — Case Control)**
- Modality: Full-field digital mammography screening, DICOM+PNG
- Size: 8,723 cases (873 first-time cancers + 7,850 healthy controls); pixel-level tumor annotations, BI-RADS, breast density, biopsy-confirmed
- Citation: Dembrower, K. et al., "A Multi-million Mammography Image Dataset and Population-Based Screening Cohort... (CSAW)," *J Digit Imaging* 33, 408-413 (2020). DOI: [10.1007/s10278-019-00278-0](https://doi.org/10.1007/s10278-019-00278-0). Data DOI: 10.5878/45vm-t798
- Access: [researchdata.se / DORIS](https://researchdata.se/en/catalogue/dataset/2021-204-1) — imaging on request, metadata CSV open
- License: CC BY 4.0

**DDSM (Digital Database for Screening Mammography)**
- Modality: Digitized screen-film mammography, original uncurated source for CBIS-DDSM
- Size: ~2,500 studies, ~10,000+ images across 4 scanner types
- Citation: Heath, M. et al., "Current status of the Digital Database for Screening Mammography," *Digital Mammography*, 1998
- Access: Original host (eng.usf.edu) largely defunct; use CBIS-DDSM instead. Community "Mini-DDSM" also exists ([arXiv:2010.00494](https://arxiv.org/pdf/2010.00494))
- License: Historically free for research; undocumented
- **[UNVERIFIED]** — recommend citing CBIS-DDSM as the actively maintained successor

**DMID (Digital Mammography dataset for Breast Cancer Diagnosis Research)**
- Modality: Digital mammography (Samved Hospital, India), DICOM+TIFF
- Size: 510 images; case type (normal/benign/malignant), BI-RADS, breast density, abnormality type, ROI masks with center/radius
- Citation: "Digital mammography dataset for breast cancer diagnosis research (DMID)," *Biomedical Engineering Letters* (2024). [Springer](https://link.springer.com/article/10.1007/s13534-023-00339-y) / [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10874363/)
- Access: [Figshare](https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883); mirrored on [Kaggle](https://www.kaggle.com/datasets/orvile/dmid-breast-cancer-mammography-dataset)
- License: **[UNVERIFIED]**

**EMBED (EMory BrEast imaging Dataset)**
- Modality: 2D digital mammography, synthetic 2D (C-view), and DBT — current public release covers 2D/C-view only (~20% of full corpus)
- Size: 3.4M images, 110,000 patients (2013-2020), 60,000 annotated lesions, racially diverse cohort
- Citation: Jeong, J.J. et al., "The EMory BrEast imaging Dataset (EMBED)," *Radiology: AI* (2023). DOI: [10.1148/ryai.220047](https://pubs.rsna.org/doi/full/10.1148/ryai.220047)
- Access: [AWS Open Data Registry](https://registry.opendata.aws/emory-breast-imaging-dataset-embed/), controlled access via request form; docs at [GitHub](https://github.com/Emory-HITI/EMBED_Open_Data)
- License: Custom Research Use Agreement — academic/non-commercial only, no redistribution

**INbreast**
- Modality: Full-field digital mammography (Siemens amorphous selenium detector); masses, calcifications, asymmetries, architectural distortions with XML contours
- Size: 115 cases, 410 images
- Citation: Moreira, I.C. et al., "INbreast: toward a full-field digital mammographic database," *Academic Radiology* 19(2), 236-248 (2012). PMID: [22078258](https://pubmed.ncbi.nlm.nih.gov/22078258/)
- Access: Request-based at medicalresearch.inescporto.pt (email medicalresearch@inescporto.pt). Unofficial Kaggle mirror exists but is not authoritative
- License: Requires signed data agreement
- **[UNVERIFIED]** — official host DNS-dead at check time; the group reportedly still responds to email requests

**KAU-BCMD (King Abdulaziz University Breast Cancer Mammogram Dataset)**
- Modality: Digital mammography, CC+MLO both breasts; includes 205 paired ultrasound cases
- Size: 1,416 cases, 5,662 mammogram images (+405 ultrasound); BI-RADS (3-radiologist majority vote), breast density
- Citation: Alsolami, A.S. et al., "King Abdulaziz University Breast Cancer Mammogram Dataset (KAU-BCMD)," *Data* 6(11), 111 (2021). DOI: [10.3390/data6110111](https://doi.org/10.3390/data6110111)
- Access: [Kaggle](https://www.kaggle.com/datasets/asmaasaad/king-abdulaziz-university-mammogram-dataset), [Mendeley Data](https://www.mendeley.com/catalogue/4a50041a-0c4a-331b-b69d-09414e39f54c/)
- License: **[UNVERIFIED]**
- Note: "BMCD" (sometimes cited separately) appears to be a naming confusion with this dataset — no distinct "BMCD" release was found.

**mini-MIAS**
- Modality: Digitized screen-film mammography, single MLO view per breast, fixed 1024×1024 PGM (this repo's currently-implemented mammography dataset — see [dataset_reproduction_plan.md](dataset_reproduction_plan.md))
- Size: 322 images (161 pairs) — 207 normal, 64 benign, 51 malignant; abnormality class, benign/malignant, centroid+radius ground truth
- Citation: Suckling, J. et al., "The Mammographic Image Analysis Society Digital Mammogram Database," *Exerpta Medica* 1069, 375-378 (1994)
- Access: [PEIPA, University of Essex](http://peipa.essex.ac.uk/info/mias.html); archived at [Cambridge repository](https://www.repository.cam.ac.uk/items/b6a97f0c-3b9b-40ad-8f18-3d121eef1459)
- License: CC BY (per repository metadata)
- Note: "mini-MIAS-2" does not appear to exist as a distinct official dataset — only unofficial reprocessed Kaggle copies were found under that name.

**Mammo-MX**
- Modality: Digital mammography, Hologic Selenia; Mexican population
- Size: 13,659 images / 3,368 patients (2023-2024); BI-RADS + density
- Citation: *Machine Learning: Science and Technology*, DOI [10.1088/2632-2153/ae275c](https://iopscience.iop.org/article/10.1088/2632-2153/ae275c)
- Access: [Zenodo](https://zenodo.org/records/17740027), fully open (74.6 GB)
- License: CC BY 4.0

**MammosighTR**
- Modality: Digital mammography (CR and FFDM), Turkish national breast screening program
- Size: 12,740 patient cases (2016-2022); BI-RADS + density + lesion quadrant labels
- Citation: *Radiology: AI*, DOI [10.1148/ryai.240841](https://pubs.rsna.org/doi/10.1148/ryai.240841)
- Access: **[UNVERIFIED]** access terms; likely request/DUA-based similar to other national-registry datasets
- License: **[UNVERIFIED]**

**NL-Breast-Screening (Newfoundland and Labrador)**
- Modality: Full-field digital mammography, screening program (GE Senograph Essential, 2008-2010)
- Size: 5,997 screening cases, 26,988 images, 149 biopsy-confirmed cancer-positive cases; includes radiologist false-positive labeling
- Citation: Kendall, E. et al., "Full Field Digital Mammography Dataset from a Population Screening Program," *Scientific Data* 12, 1479 (2025). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12378999/)
- Access: [FRDR](https://www.frdr-dfdr.ca/repo/dataset/cb5ddb98-ccdf-455c-886c-c9750a8c34c2), DOI 10.20383/103.01349, Globus/zip. Use Version 3 (v1 has directory-naming errors)
- License: CC BY 4.0

**OMI-DB / OPTIMAM (OPTIMAM Mammography Image Database)**
- Modality: FFDM + DBT (+MRI for a high-risk subset), processed and raw images
- Size: sources disagree — ~750,000 women in one source, >465,000/1.3M+ studies/~7M images in another (UK NHS Breast Screening Programme, 2008-present) — **confirm current number before publishing**
- Citation: Halling-Brown, M.D. et al., "OPTIMAM Mammography Image Database," *Radiology: AI* 3(1), e200103 (2021). [arXiv:2004.04742](https://arxiv.org/abs/2004.04742)
- Access: Application-based via [Cancer Research Horizons](https://www.cancerresearchhorizons.com/our-portfolio/our-licensing-opportunities/optimam-mammography-image-database-omi-db) / [medphys.royalsurrey.nhs.uk/omidb](https://medphys.royalsurrey.nhs.uk/omidb/) — Data Access Committee review
- License: Not open-CC; commercial/academic licensing + DUA required
- Also see this dataset's **clinical-data companion** in [Section 6](#6-clinical-genomic--biomarker-data).

**RSNA Screening Mammography Breast Cancer Detection (Kaggle/AI Challenge)**
- Modality: Screening FFDM, DICOM, 4 views typical
- Size: ~20,000 studies (Australia + U.S. screening programs); BI-RADS density (A-D), laterality, implant status, pathology-confirmed outcome
- Citation: [RSNA 2023 AI Challenge](https://www.rsna.org/rsnai/ai-image-challenge/screening-mammography-breast-cancer-detection-ai-challenge)
- Access: [Kaggle](https://www.kaggle.com/competitions/rsna-breast-cancer-detection); mirrored on [AWS Open Data](https://registry.opendata.aws/rsna-screening-mammography-breast-cancer-detection/)
- License: Non-commercial research/education only; redistribution prohibited (custom RSNA terms)

**VinDr-Mammo**
- Modality: Full-field digital mammography, 4-view screening exams, double-read with arbitration
- Size: 5,000 exams, 20,000 images (DICOM); breast-level BI-RADS + density, finding-level category/location/BI-RADS
- Citation: Nguyen, H.T. et al., "VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography," *Scientific Data* 10, 277 (2023). DOI: [10.1038/s41597-023-02100-7](https://www.nature.com/articles/s41597-023-02100-7). Data DOI: [10.13026/br2v-7517](https://doi.org/10.13026/br2v-7517)
- Access: [PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/), credentialed account + signed DUA required
- License: PhysioNet Restricted Health Data License 1.5.0

**BCS-DBT / Breast-Cancer-Screening-DBT (Duke, digital breast tomosynthesis)**
- Modality: 3D digital breast tomosynthesis, mostly 4 views
- Size: 5,060 participants, 5,610 studies, 22,032 series, 1,526 GB
- Citation: Buda, M. et al., "A Data Set and Deep Learning Algorithm for the Detection of Masses and Architectural Distortions in Digital Breast Tomosynthesis Images," *JAMA Network Open* 4(8), e2119100 (2021). DOI: [10.1001/jamanetworkopen.2021.19100](https://doi.org/10.1001/jamanetworkopen.2021.19100). Data DOI: [10.7937/E4WT-CD02](https://doi.org/10.7937/E4WT-CD02)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=64685580), NBIA Data Retriever
- License: **CC BY-NC 4.0**

---

## 2. Ultrasound

**ALN-Ultra**
- Modality: B-mode, images + video, paired with axillary imaging; 2D
- Size: 257 patients, ~16.4 TB total (images/videos + clinical/biopsy labels for axillary lymph node metastasis prediction)
- Citation: associated with DOI [10.3389/fonc.2023.1219838](https://doi.org/10.3389/fonc.2023.1219838) — **[UNVERIFIED]** exact BUS-specific paper title
- Access: [Zenodo](https://zenodo.org/records/15003119)
- License: CC BY 4.0

**BCMID (Breast Cancer Multimodal Imaging Dataset)**
- Modality: B-mode ultrasound **plus mammography** (mixed-modality)
- Size: 323 patients (ages 26-82), ultrasound + mammogram per patient, BI-RADS in CSV; 1.3 TB; Ayady Almostakbal Hospital, Alexandria
- Access: [Zenodo](https://zenodo.org/records/14970848) — files require login/restricted access. Published March 2025
- License: CC BY 4.0

**BrEaST-Lesions USG**
- Modality: B-mode, 2D; annotated color/Doppler feature presence where applicable (this repo's currently-implemented `breast` dataset — see [dataset_reproduction_plan.md](dataset_reproduction_plan.md))
- Size: 256 breast scans/patients, 266 segmented lesions (154 benign, 98 malignant, 4 normal); freehand tumor masks, full BI-RADS descriptor set, histopathological diagnosis (33 subtypes)
- Citation: Pawłowska, A. et al., "Curated benchmark dataset for ultrasound based breast lesion analysis," *Scientific Data*, 2024. DOI: [10.1038/s41597-024-02984-z](https://doi.org/10.1038/s41597-024-02984-z)
- Access: [TCIA "Breast-Lesions-USG"](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/), DOI 10.7937/9WKK-Q141
- License: CC BY 4.0

**BUID / QAMEBI (An Open-Access Breast Lesion Ultrasound Image Database)**
- Modality: B-mode, 2D, Aixplorer Ultimate scanner
- Size: 232 lesions (109 benign, 123 malignant), original + radiologist mask + fusion overlay per case
- Citation: Abbasian Ardakani et al., "An open-access breast lesion ultrasound image database," *Computers in Biology and Medicine* 152:106438, 2023. DOI: [10.1016/j.compbiomed.2022.106438](https://doi.org/10.1016/j.compbiomed.2022.106438)
- Access: [qamebi.com](https://qamebi.com/breast-ultrasound-images-database/), direct zip downloads
- License: not formally stated; attribution requested
- Note: "BUID" is an informal alias — same database as QAMEBI, not two datasets.

**BUS-BRA** *(this repo's currently-implemented `busbra` dataset)*
- Modality: B-mode, 2D
- Size: 1,875 images / 1,064 patients, biopsy-proven, BI-RADS 2-5, tumor + normal region masks
- Citation: Gómez-Flores, W. et al., "BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems," *Medical Physics* 51:3110-3123, 2024. DOI: [10.1002/mp.16812](https://doi.org/10.1002/mp.16812)
- Access: [Zenodo (current)](https://zenodo.org/records/8231412), CC BY 4.0, PNG+CSV; also [Kaggle](https://www.kaggle.com/datasets/orvile/bus-bra-a-breast-ultrasound-dataset), [GitHub](https://github.com/wgomezf/BUS-BRA)
- License: CC BY 4.0

**BUS-CoT**
- Modality: B-mode, 2D
- Size: 11,439 images / 11,850 lesions / 4,838 patients, all 99 WHO histopathology categories; curated 5,163-image high-quality subset
- Labels: chain-of-thought diagnostic reasoning annotations, lesion masks, histopathology category
- Citation: "A Chain-of-thought Reasoning Breast Ultrasound Dataset Covering All Histopathology Categories," *Scientific Data*, 2026. DOI: [10.1038/s41597-026-06702-9](https://doi.org/10.1038/s41597-026-06702-9)
- Access: [Figshare](https://doi.org/10.6084/m9.figshare.30838715)
- License: **[UNVERIFIED]**. Newly released (2026).

**BUSC (Mendeley, 250-image set)** *(this repo's currently-implemented `busc` dataset)*
- Modality: B-mode, 2D, original 64×64 px
- Size: 250 images (100 benign, 150 malignant)
- Citation: Rodrigues, P.S. (2017), "Breast Ultrasound Image," Mendeley Data V1, DOI: [10.17632/wmy84gzngw.1](https://doi.org/10.17632/wmy84gzngw.1)
- Access: [Mendeley Data](https://data.mendeley.com/datasets/vckdnhtw26/1)
- License: CC BY 4.0

**BUSI (Al-Dhabyani et al., "Dataset of Breast Ultrasound Images")** *(this repo's currently-implemented `busi` dataset)*
- Modality: B-mode, 2D, avg. 500×500 px
- Size: 780 images / ~600 patients; 437 benign, 210 malignant, 133 normal, pixel-wise masks
- Citation: Al-Dhabyani, W. et al., "Dataset of breast ultrasound images," *Data in Brief* 28:104863, 2020. DOI: [10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863)
- Access: original host [scholar.cu.edu.eg](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) (Cairo University); mirrored on [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- License: not formally stated
- Note: a correction letter (PMC10293973) flags mislabeling/duplicate issues in the original release.

**BUSIS**
- Modality: B-mode, 2D, Harbin/Qingdao/Hebei hospitals
- Size: 562 images; binary ground-truth masks via multi-radiologist majority voting
- Citation: Zhang, Y. et al., "BUSIS: A Benchmark for Breast Ultrasound Image Segmentation." [arXiv:1801.03182](https://arxiv.org/abs/1801.03182)
- Access: **[UNVERIFIED]** — no working public download link found; likely requires contacting authors
- License: unknown

**BUS_UC**
- Modality: B-mode, 2D, sourced from ultrasoundcases.info teaching archive, subsequently annotated
- Size: 811 images, 256×256 px (358 benign, 453 malignant), radiologist-added masks
- Citation: Iqbal, S. & Sharif, M., "Memory-efficient transformer network with feature fusion for breast tumor segmentation and classification task," *Engineering Applications of Artificial Intelligence*, 2023
- Access: [Mendeley Data](https://data.mendeley.com/datasets/3ksd7w7jkx/1)
- License: CC BY 4.0
- Note: distinct from BUS-UCLM and BUSC despite similar naming.

**BUSI_WHU**
- Modality: B-mode, 2D, Renmin Hospital of Wuhan University
- Size: 927 images (560 benign, 367 malignant), collected Dec 2020-Dec 2022; radiologist-annotated masks
- Access: [Mendeley Data](https://data.mendeley.com/datasets/k6cpmwybk3/1); mirrored on [Kaggle](https://www.kaggle.com/datasets/orvile/busi-whu-breast-cancer-ultrasound-image-dataset)
- License: CC BY 4.0
- **[UNVERIFIED]** — no matched journal paper found; appears to be a standalone Mendeley release.

**BUS-UCLM**
- Modality: B-mode, 2D, Siemens ACUSON S2000; per-image Doppler-feature-presence metadata
- Size: 683 images / 38 patients (174 benign, 90 malignant, 419 normal), 2022-2023; RGB segmentation masks
- Citation: "BUS-UCLM: Breast ultrasound lesion segmentation dataset," *Scientific Data*, 2025. DOI: [10.1038/s41597-025-04562-3](https://doi.org/10.1038/s41597-025-04562-3)
- Access: [Mendeley Data](https://data.mendeley.com/datasets/7fvgj4jsp7/1); also [Kaggle](https://www.kaggle.com/datasets/orvile/bus-uclm-breast-ultrasound-dataset)
- License: **CC BY-NC 3.0** (non-commercial — note this differs from most peers here). Newly released (2025).

**CADBUSI**
- Modality: B-mode images **and video (cine)**, 2D, Mayo Clinic Health System
- Size: 79,281 exams / 60,688 patients (2002-2025), 756,315 images + 136,197 videos, BI-RADS + pathology-verified diagnosis
- Citation: "Computer-Aided Diagnosis for Breast Ultrasound Imagery Dataset," PubMed 41530416; [project site](https://datascienceuwl.github.io/CADBUSI/)
- Access: **not publicly downloadable** — restricted, likely requires a DUA with Mayo Clinic. Creation code (not data) at [GitHub](https://github.com/Poofy1/CADBUSI-Database)
- License: n/a (restricted)
- Newly released (2026); the largest breast-US dataset found in this survey by a wide margin.

**CVA-Net / BUV (Breast Ultrasound Video dataset)**
- Modality: B-mode **video (cine)**, 2D
- Size: 188 videos / 25,272 frames (113 malignant, 75 benign), frame-level bounding boxes + video-level classification, dual-pathologist annotation
- Citation: Lin, Y. et al., "A New Dataset and A Baseline Model for Breast Lesion Detection in Ultrasound Videos," MICCAI 2022. [arXiv:2207.00141](https://arxiv.org/abs/2207.00141)
- Access: [GitHub](https://github.com/jhl-Det/CVA-Net) — Baidu Drive / Google Drive links
- License: non-commercial research/educational use only
- Note: also referred to as "BUSV"/"BreastVid" elsewhere — same dataset.

**GDPH&SYSUCC**
- Modality: B-mode, 2D, whole-image (no ROI cropping)
- Size: 2,405 images (886 benign, 1,519 malignant); Guangdong Provincial People's Hospital + Sun Yat-sen University Cancer Center
- Citation: Xu, Y. et al., "HoVer-Trans: Anatomy-aware HoVer-Transformer for ROI-free Breast Cancer Diagnosis in Ultrasound Images." [arXiv:2205.08390](https://arxiv.org/abs/2205.08390)
- Access: [GitHub](https://github.com/yuhaomo/HoVerTrans) — OneDrive link in repo
- License: not specified — **[UNVERIFIED]**

**OASBUD (Open Access Series of Breast Ultrasonic Data)**
- Modality: **raw RF ultrasound signal data** (not B-mode images), 2 orthogonal scans/lesion
- Size: 100 lesions / 78 women (52 malignant, 48 benign), 2013-2015, MATLAB files, ROI masks
- Citation: Piotrzkowska-Wróblewska, K. et al., "Open access database of raw ultrasonic signals acquired from malignant and benign breast lesions," *Medical Physics*, 2017
- Access: [Zenodo](https://zenodo.org/records/545928)
- License: CC BY 4.0 (per third-party listing)

**STU-Hospital dataset**
- Modality: B-mode, 2D, GE Voluson E10, First Hospital of Medical College of Shantou University
- Size: 42 images
- Citation: Zhuang, Z. et al., "An RDAU-NET model for lesion segmentation in breast ultrasound images," *PLOS ONE* 14(8), 2019
- Access: **[UNVERIFIED]** — no direct public download link; circulates only via third-party GitHub repos implementing RDAU-NET
- License: unknown

**TDSC-ABUS2023**
- Modality: **Automated 3D Breast Ultrasound (ABUS)**, 3D volumes, .nrrd, Invenia ABUS system
- Size: 200 volumes (100 train / 30 open val / 70 closed test); voxel-level tumor segmentation + classification
- Citation: Lin, Y. et al., "Tumor Detection, Segmentation and Classification Challenge on Automated 3D Breast Ultrasound: The TDSC-ABUS Challenge." [arXiv:2501.15588](https://arxiv.org/abs/2501.15588)
- Access: [tdsc-abus2023.grand-challenge.org](https://tdsc-abus2023.grand-challenge.org/Dataset/) — requires signed DUA emailed to tdscabus@gmail.com
- License: not stated beyond the signed agreement
- The only dedicated public 3D ABUS dataset with challenge infrastructure found.

**UDIAT / Dataset B**
- Modality: B-mode, 2D, Siemens ACUSON Sequoia C512
- Size: 163 images (109-110 benign, 53-54 malignant depending on source), pixel-wise masks
- Citation: Yap, M.H. et al., "Automated Breast Ultrasound Lesions Detection Using Convolutional Neural Networks," *IEEE J. Biomedical and Health Informatics* 22(4):1218-1226, 2017
- Access: [helward.mmu.ac.uk/STAFF/m.yap](https://helward.mmu.ac.uk/STAFF/m.yap/dataset.php) — not directly downloadable, requires contacting authors
- License: none stated

**US3M (Multimodal Breast Ultrasound Dataset)**
- Modality: described as multimodal B-mode; ~248 subjects per a secondary listing
- Access: [Kaggle](https://www.kaggle.com/datasets/timesxy/multimodal-breast-ultrasound-dataset-us3m)
- **[UNVERIFIED]** — Kaggle page not scrapeable, no independent paper found; recommend manual inspection before citing.

### Checked and ruled out (not breast, or not a real dataset)
- **MMOTU** — confirmed **ovarian** tumor ultrasound dataset, not breast.
- **Thammasat University Hospital dataset (TUHD)** — no primary paper/page/link located; unconfirmed.
- **MIDI-B (TCIA)** — general DICOM de-identification benchmark, not breast-US-specific.
- Breast elastography as a standalone public dataset — none found despite targeted searching (see [Section 4](#4-ct--less-common-imaging-modalities) for the same conclusion from the CT/other-modality pass).

### Naming disambiguation
"BUS-BRA" = "BUSBRA" (same dataset). "BUSC," "BUS_UC," "BUS-UCLM," and "BUSI_WHU" are **four distinct datasets** despite similar names. "BUID" is an alias for QAMEBI, not an independent dataset.

---

## 3. MRI

**ACRIN-6698 / I-SPY2 Breast DWI**
- Modality: Diffusion-weighted MRI (b=0,100,600,800 s/mm²) + T2w + DCE-MRI; 1.5T/3.0T
- Size: 385 subjects, 1,123 studies, 4 longitudinal timepoints, 842 GB; manual tumor ROI, ADC maps, DCE enhancement maps, pCR outcome
- Citation: Partridge, S.C. et al., "Diffusion-weighted MRI Findings Predict Pathologic Response... ACRIN 6698," *Radiology* 289(3):618-627, 2018. DOI: [10.1148/radiol.2018180273](https://doi.org/10.1148/radiol.2018180273)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=50135447), NBIA Data Retriever
- License: CC BY 4.0

**BMMR2 Challenge**
- Modality: DWI + DCE-MRI + T2w, subset of ACRIN-6698
- Size: 573 studies from 191 subjects, 60/40 train-test split; pCR labels (training set only)
- Citation: Partridge, S.C. et al., "Breast Multiparametric MRI for Prediction of Neoadjuvant Chemotherapy Response... BMMR2 Challenge," *Radiology: Imaging Cancer*, 2024. DOI: [10.1148/rycan.230033](https://doi.org/10.1148/rycan.230033)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=89096426)
- License: CC BY 3.0

**Breast-Diagnosis (TCIA)**
- Modality: MR (T2/STIR/BLISS) + MG + CT + PT (multi-modality collection); Philips 1.5T
- Size: 88 subjects, 148 studies, 105,144 images (60.8 GB); BI-RADS features, ER/PR/HER2, Oncotype scores
- Citation: Bloch, B.N. et al., "BREAST-DIAGNOSIS" [Data set], TCIA, 2015. DOI: [10.7937/K9/TCIA.2015.SDNRQXXR](http://doi.org/10.7937/K9/TCIA.2015.SDNRQXXR)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/display/Public/BREAST-DIAGNOSIS), NBIA Data Retriever
- License: CC BY 3.0

**Breast-MRI-NACT-Pilot**
- Modality: DCE-MRI, 3D FGRE T1w fat-suppressed; 1.5T GE Signa
- Size: 64 patients, 189 studies, 19.5 GB; tumor/breast segmentation via signal-enhancement-ratio mapping
- Citation: Newitt, D. & Hylton, N., TCIA, 2016. DOI: [10.7937/K9/TCIA.2016.QHsyhJKy](https://doi.org/10.7937/K9/TCIA.2016.QHsyhJKy)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22513764)
- License: CC BY 3.0

**BreastDCEDL**
- Modality: 3D pretreatment DCE-MRI, standardized NIfTI; combines I-SPY1 (172) + I-SPY2 (982) + Duke (916) = 2,070 patients
- Labels: 1,154 binary segmentation masks (I-SPY cohorts), 916 bounding boxes (Duke); HR/HER2 status; pCR for 1,452 patients (70.1%)
- Citation: Fridman, N. et al., "BreastDCEDL: A standardized deep learning-ready breast DCE-MRI dataset of 2,070 patients," *Scientific Data* 13:264, 2026. DOI: [10.1038/s41597-026-06589-6](https://doi.org/10.1038/s41597-026-06589-6)
- Access: [Zenodo](https://zenodo.org/records/17274053), [GitHub](https://github.com/naomifridman/BreastDCEDL)
- License: CC BY-NC-ND 4.0

**BreastDCEDL-ISPY2**
- Modality: NIfTI-converted DCE-MRI, standardized from I-SPY2 raw DICOM
- Size: 982 patients, 8,021 series (2010-2016); tumor segmentation masks, clinical metadata, predefined biomarker-stratified splits
- Citation: Fridman, N. et al., [arXiv:2506.12190](https://arxiv.org/abs/2506.12190), 2025
- Access: [TCIA](https://www.cancerimagingarchive.net/analysis-result/breastdcedl-ispy2/), 54 GB, Aspera Connect. Data DOI: 10.7937/42WQ-TH78
- License: CC BY 4.0

**BreastDM (BreaDM)** *(this repo's currently-implemented `breamdm` dataset)*
- Modality: DCE-MRI, pre-contrast + 8 post-contrast + subtraction sequences; Taizhou Central Hospital, 2018-2021
- Size: 232 cases (147 malignant, 85 benign); tumor segmentation + classification labels
- Citation: Zhao, X. et al., "BreastDM: A DCE-MRI dataset for breast tumor image segmentation and classification," *Computers in Biology and Medicine* 164:107255, 2023. DOI: [10.1016/j.compbiomed.2023.107255](https://doi.org/10.1016/j.compbiomed.2023.107255)
- Access: [GitHub](https://github.com/kamranisg/BreastDM) with Google Drive links
- License: **[UNVERIFIED]** — no explicit license in repo, confirm with authors before redistribution

**Duke-Breast-Cancer-MRI**
- Modality: DCE-MRI, non-fat-sat + fat-sat pre/3-4 post-contrast T1w; 1.5T/3T
- Size: 922 patients, 5,161 series, 773,888 images (368.4 GB); 3D tumor bounding boxes, molecular subtype, recurrence, 529 radiogenomic features
- Citation: Saha, A. et al., "A machine learning approach to radiogenomics of breast cancer," *British Journal of Cancer* 119(4):508-516, 2018. DOI: [10.1038/s41416-018-0185-8](https://doi.org/10.1038/s41416-018-0185-8)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903), NBIA Data Retriever
- License: CC BY-NC 4.0

**EA1141**
- Modality: Abbreviated Breast MRI (AB-MR) + Digital Breast Tomosynthesis + CT (trial comparing detection methods, not full multiparametric DCE-MRI)
- Size: 500 patients, 1,976 studies, 2.82 TB
- Citation: Comstock, C.E. et al., *JAMA* 323(8):746-756, 2020
- Access: [TCIA](https://www.cancerimagingarchive.net/collection/ea1141/), DOI 10.7937/2bas-hr33
- License: CC BY 4.0

**European Multi-Center Breast Cancer MRI Dataset**
- Modality: Multiparametric MRI, heterogeneous scanners/field strengths, 6 institutions/5 countries
- Size: 741 examinations; malignant/benign/non-lesion labels
- Citation: Müller-Franzes, G. et al., [arXiv:2506.00474](https://arxiv.org/abs/2506.00474) (v3, 2026)
- Access: [Zenodo](https://zenodo.org/records/15075570) (~500 subjects/13GB portion) and Health-RI XNAT (account required)
- License: **[UNVERIFIED]**

**fastMRI Breast**
- Modality: Radial k-space + DICOM, golden-angle GRASP DCE-MRI; 3T
- Size: NYU page states 300 radial DCE-MRIs; RYAI paper describes 275 labeled cases (70 malignant, 158 benign, 47 no-lesion) — counts differ slightly, reconcile against the paper directly
- Citation: Solomon, E. et al., "FastMRI Breast: A Publicly Available Radial k-Space Dataset of Breast DCE-MRI," *Radiology: AI* 7(1):e240345, 2025. DOI: [10.1148/ryai.240345](https://doi.org/10.1148/ryai.240345)
- Access: [fastmri.med.nyu.edu](https://fastmri.med.nyu.edu/) — data-sharing agreement required
- License: Custom NYU fastMRI data-sharing agreement

**I-SPY1 / ACRIN 6657**
- Modality: DCE-MRI, 3D fat-suppressed T1w GRE; 1.5T
- Size: 222 participants, 847 studies, 386,528 images, 76.2 GB; FTV/SER, LD, pCR, recurrence-free survival
- Citation: Hylton, N.M. et al., "Neoadjuvant Chemotherapy for Breast Cancer... I-SPY 1 TRIAL," *Radiology* 279(1):44-55, 2016. DOI: [10.1148/radiol.2015150013](https://doi.org/10.1148/radiol.2015150013)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=20643859)
- License: CC BY 3.0

**I-SPY1-Tumor-SEG-Radiomics**
- Modality: Same cohort as I-SPY1, harmonized/uniformly processed
- Size: 163 patients, 1,467 files (6.1 GB NIfTI); full 3D primary lesion segmentation + radiomic feature panel
- Citation: Chitalia, R. et al., "Expert tumor annotations and radiomics for locally advanced breast cancer in DCE-MRI for ACRIN 6657/I-SPY1," *Scientific Data* 9:37, 2022. DOI: [10.1038/s41597-022-01555-4](https://doi.org/10.1038/s41597-022-01555-4)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101942541), Aspera Connect
- License: CC BY 3.0

**I-SPY2**
- Modality: T2w + T1w DCE-MRI + DWI; 1.5T/3.0T. Combined with ACRIN-6698 as "I-SPY2 Imaging Cohort 1" (985 patients)
- Size: 719 patients (main), 2,688 studies, 1.6 TB (2.4TB combined); pCR, HR status, HER2 status
- Citation: Li, W. et al. (data) DOI: [10.7937/TCIA.D8Z0-9T85](https://doi.org/10.7937/TCIA.D8Z0-9T85); (paper) *npj Breast Cancer* 6:17, 2020. DOI: [10.1038/s41523-020-00203-7](https://doi.org/10.1038/s41523-020-00203-7)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230072)
- License: CC BY 4.0 (updated July 2022; was CC BY-NC 4.0 in v1)
- Also see this cohort's **clinical/biomarker companion** in [Section 6](#6-clinical-genomic--biomarker-data).

**MAMA-MIA**
- Modality: Pre-treatment DCE-MRI, aggregated/harmonized from I-SPY1, I-SPY2, Breast-MRI-NACT-Pilot, Duke
- Size: 1,506 cases; expert primary-tumor + non-mass-enhancement segmentations, 49 harmonized clinical variables, pretrained nnU-Net weights included
- Citation: Garrucho, L. et al., "A large-scale multicenter breast cancer DCE-MRI benchmark dataset with expert segmentations," *Scientific Data* 12:453, 2025. DOI: [10.1038/s41597-025-04707-4](https://doi.org/10.1038/s41597-025-04707-4)
- Access: [Synapse](https://www.synapse.org/) (DOI 10.7303/syn60868042, account required) or Health-RI XNAT; code at [GitHub](https://github.com/LidiaGarrucho/MAMA-MIA)
- License: CC BY-NC 4.0

**QIN-Breast (67-subject Philips cohort)**
- Modality: PET/CT + MRI (DWI, DCE, multi-flip T1-mapping); Philips Achieva 3.0T
- Size: 67 subjects, 216 studies; pathological response spreadsheet
- Citation: Li, X. et al., "Multiparametric MRI for predicting pathological response after the first cycle of neoadjuvant chemotherapy in breast cancer," *Investigative Radiology* 50(4):195-204, 2015
- Access: [TCIA](https://wiki.cancerimagingarchive.net/display/Public/QIN-Breast), DOI: 10.7937/K9/TCIA.2016.21JUEBH0
- License: CC BY 3.0

**QIN-BREAST-DCE-MRI (10-patient Siemens cohort, distinct from QIN-Breast above)**
- Modality: DCE-MRI, 3D GRE TWIST; Siemens 3T TIM Trio
- Size: 10 patients, 20 studies; 3 pCR / 7 non-responders, AIF timecourse data
- Citation: Huang, W. et al., "Variations of DCE-MRI in Evaluation of Breast Cancer Therapy Response," *Translational Oncology* 7(1):153-166, 2014
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=18514286) (companion QIN-BREAST-DCE-MRI-DA-RAD also on TCIA)
- License: CC BY 3.0

**RIDER Breast MRI**
- Modality: DCE-MRI + DWI ("coffee-break" repeat exams)
- Size: 5 subjects, 10 studies, 1,500 images (401 MB); test-retest reproducibility data
- Citation: Meyer, C.R. et al., TCIA, 2015. DOI: [10.7937/K9/TCIA.2015.H1SXNUXL](https://doi.org/10.7937/K9/TCIA.2015.H1SXNUXL)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/display/Public/RIDER+Breast+MRI)
- License: CC BY 3.0

### Checked but not included (synthetic, code-only, or unconfirmed)
- **"Dataset of artificial breast cancer MRIs from unpaired mammograms"** (2025) — CycleGAN-synthesized, not real MRI.
- **MSKCC/CCNY "High-Performance Open-Source AI for Breast Cancer Detection... MRI"** (*Radiology: AI*, 2025) — code + weights public, but the 30,672-exam imaging data itself is **not released**.
- **"BreastScreening-AI"** — could not verify as a distinct breast-MRI dataset.
- **BC-MRI-SEG** ([arXiv:2404.13756](https://arxiv.org/abs/2404.13756)) — a segmentation benchmark harness built on I-SPY1/BreastDM/RIDER/Duke, not a new imaging release.

---

## 4. CT & Less-Common Imaging Modalities

### Contrast-enhanced spectral mammography (CESM)
See **CDD-CESM** in [Section 1](#1-mammography) — a mammography variant (dual-energy), not standard CT, but the closest CT-adjacent modality with a real public dataset.

### Dedicated breast CT (incl. photon-counting)

No dataset of real clinical dedicated-breast-CT images with disease labels for CAD/AI use was found publicly released. What exists is limited to phantom/simulation data and multi-modality collections where CT is a minor component:

**UC Davis Patient-Derived 3D Digital Breast Phantom Dataset**
- Modality: Voxelized computational phantoms (tissue-class segmentation) from clinical dedicated breast CT scans, UC Davis BCT scanner
- Size: 150 patients' segmented volumes; tissue-class labels only (no cancer diagnosis) — for virtual clinical trial/dose simulation
- Citation: Sarno, A. et al., *Medical Physics* (2021). PubMed: [33683711](https://pubmed.ncbi.nlm.nih.gov/33683711/)
- Access: [Zenodo](https://zenodo.org/records/4529852), DICOM, 95.7 GB
- License: CC BY 4.0

**BREAST-DIAGNOSIS (TCIA)** — see full entry in [Section 3](#3-mri); includes a CT series alongside MR/PT/mammography, not a dedicated breast-CT protocol.

**Explicit gap**: No public dataset of real photon-counting dedicated-breast-CT clinical images exists. The only related public resource is the **2022 AAPM Deep-Learning Spectral CT Grand Challenge** — a *simulated* 2D breast phantom dataset (1,000 train/10 val/100 test pairs), not real acquired data. DOI: [10.1002/mp.16363](https://doi.org/10.1002/mp.16363). [Challenge page](https://www.aapm.org/GrandChallenge/DL-spectral-CT/).

### Thermography / infrared imaging

**DMR-IR (Database for Mastology Research – Infrared)**
- Modality: Infrared thermal images (FLIR SC620, <0.04°C sensitivity), plus digitized mammograms and clinical data for a subset
- Size: 293 patients originally; a Hugging Face mirror repackages it as 6,878 images, 4.97 GB
- Citation: Silva, L.F. et al., "A New Database for Breast Research with Infrared Image," *J. Medical Imaging and Health Informatics* 4(1), 92-100 (2014). DOI: [10.1166/jmihi.2014.1226](https://doi.org/10.1166/jmihi.2014.1226)
- Access: [visual.ic.uff.br/dmi](https://visual.ic.uff.br/dmi) (free registration; site showed signs of disrepair at check time); unofficial mirror [huggingface.co/datasets/SemilleroCV/DMR-IR](https://huggingface.co/datasets/SemilleroCV/DMR-IR)
- License: Not formally stated by the original site; HF mirror lists "not provided"

### PET / PET-CT

**QIN-Breast (TCIA)**
- Modality: Longitudinal PET/CT + quantitative MR, 3 timepoints across neoadjuvant chemotherapy
- Size: 68 subjects, 216 studies, 102,451 images (11.4 GB); treatment-response data via linked outcomes
- Citation: Li, X. et al., *Investigative Radiology* 50(4):195-204 (2015). Data DOI: [10.7937/K9/TCIA.2016.21JUEBH0](https://doi.org/10.7937/K9/TCIA.2016.21JUEBH0)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/display/Public/QIN-Breast)
- License: CC BY 3.0
- Note: this is the same TCIA collection referenced as the 67-subject MRI entry in [Section 3](#3-mri) — one dataset, both a PET/CT and an MRI resource.

### Microwave imaging

**UM-BMID (University of Manitoba Breast Microwave Imaging Dataset)**
- Modality: S-parameter measurements (1-8 GHz) from a breast microwave sensing system scanning MRI-derived 3D-printed phantoms
- Size: >1,250 phantom scans; labels are phantom construction parameters (not real-patient pathology)
- Citation: Reimer, T. et al., EuCAP 2020, DOI: [10.23919/EuCAP48036.2020.9135659](https://doi.org/10.23919/EuCAP48036.2020.9135659)
- Access: [IEEE DataPort](https://ieee-dataport.org/open-access/university-manitoba-breast-microwave-imaging-dataset-um-bmid) (free account) or [GitHub](https://github.com/UManitoba-BMS/UM-BMID)
- License: Open access (IEEE DataPort open-access terms)

**MRI-Derived Numerical Breast Models Repository**
- Modality: Not measured microwave data — 3D anatomical breast models segmented from 3.0T MRI with dielectric-property maps (3-10 GHz) for microwave-imaging simulation ground truth
- Size: 55 patients, 84 tumors (46 benign, 38 malignant)
- Citation: Pelicano, A.C. et al., *PLOS ONE* 19(5): e0302974 (2024). DOI: [10.1371/journal.pone.0302974](https://doi.org/10.1371/journal.pone.0302974)
- Access: [GitHub](https://github.com/acpelicano/breast_models_repository)
- License: CC BY 4.0

### Elastography (standalone)

**Explicit gap**: No standalone public breast-elastography dataset (shear-wave or strain, with real patient images/labels) was found. All identified elastography work uses private institutional cohorts. Confirmed independently by both the ultrasound and CT/other-modality research passes.

---

## 5. Pathology / Histopathology

**ACROBAT**
- Specimen/stain: WSI, multi-stain (H&E + up to 4 IHC: ER, PR, HER2, Ki67), 10×
- Size: 4,212 WSIs from ~1,153 patients; >54,000 registration landmark points
- Labels: registration task (paired H&E/IHC landmarks), not diagnosis
- Citation: Weitz, P. et al., "A multi-stain breast cancer histological whole-slide-image data set from routine diagnostics," *Scientific Data*, 2023; challenge paper *Medical Image Analysis*, 2024, DOI [10.1016/j.media.2024.103257](https://doi.org/10.1016/j.media.2024.103257)
- Access: [SND](https://researchdata.se/en/catalogue/dataset/2022-190-1); [challenge site](https://acrobat.grand-challenge.org/)
- License: CC-BY

**AIDPATH**
- Specimen/stain: WSI, breast slides with multiple IHC (H&E, ER, PR, Ki67, HER2); also non-breast tissue
- Size: ~1,200 slides total across tissue types (~80 GB); breast-only subset count **[UNVERIFIED]**
- Access: [mitel.dimi.uniud.it/aidpath-db](https://mitel.dimi.uniud.it/aidpath-db/), account registration required
- License: **[UNVERIFIED]**

**BACH (Breast Cancer Histology / ICIAR 2018 Grand Challenge)**
- Specimen/stain: Microscopy images + WSIs, H&E, 200×
- Size: 400 training images (100/class: normal, benign, in situ, invasive) + WSI portion, 13.4 GB
- Citation: Aresta, G. et al., "BACH: Grand Challenge on Breast Cancer Histology Images," *Medical Image Analysis*, 2019, DOI: [10.1016/j.media.2019.05.010](https://doi.org/10.1016/j.media.2019.05.010)
- Access: [Zenodo](https://zenodo.org/records/3632035); [challenge site](https://iciar2018-challenge.grand-challenge.org/)
- License: CC BY-NC-ND 4.0

**BCNB (Early Breast Cancer Core-Needle Biopsy WSI Dataset)**
- Specimen/stain: Core-needle biopsy WSIs, Iscan Coreo scanner
- Size: 1,058 WSIs / 1,058 patients (China) — only public Asian breast WSI dataset per a scoping review
- Labels: ALN metastasis status, histological grade, molecular subtype, ER/PR/HER2, Ki67
- Citation: Xu, F. et al., "Predicting Axillary Lymph Node Metastasis in Early Breast Cancer Using Deep Learning on Primary Tumor Biopsy Slides," *Frontiers in Oncology*, 2021
- Access: [bcnb.grand-challenge.org](https://bcnb.grand-challenge.org/)
- License: **[UNVERIFIED]**

**BRACS (BReAst Carcinoma Subtyping)**
- Specimen/stain: WSI + ROIs, H&E, Aperio AT2, 40×
- Size: 547 WSIs (189 patients) + 4,539 ROIs (387 WSIs/151 patients)
- Labels: 7 classes — Normal through Invasive Carcinoma
- Citation: Brancati, N. et al., "BRACS: A Dataset for BReAst Carcinoma Subtyping in H&E Histology Images," *Database* (Oxford), 2022, DOI: [10.1093/database/baac093](https://doi.org/10.1093/database/baac093)
- Access: [bracs.icar.cnr.it/download](https://www.bracs.icar.cnr.it/download/)
- License: CC BY-NC 4.0

**BreakHis (Breast Cancer Histopathological Database)**
- Specimen/stain: H&E, 4 magnifications (40×,100×,200×,400×), 700×460px PNG
- Size: 9,109 images / 82 patients (2,480 benign / 5,429 malignant)
- Labels: binary + 8 histological subtypes
- Citation: Spanhol, F. et al., "A Dataset for Breast Cancer Histopathological Image Classification," *IEEE Trans. Biomedical Engineering* 63(7):1455-1462, 2016
- Access: [inf.ufpr.br/vri/databases](http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz); [IEEE DataPort](https://ieee-dataport.org/documents/breast-cancer-histopathological-database-breakhis); Mendeley mirror
- License: Non-commercial research use only

**BreCaHAD**
- Specimen/stain: H&E, Zeiss Axiophot, 40× oil, 1360×1024px TIFF
- Size: 162 images
- Labels: 6 structure classes (mitosis, apoptosis, tumor/non-tumor nuclei, tubule/non-tubule) supporting Nottingham grading
- Citation: Aksac, A. et al., *BMC Research Notes* 12:82, 2019, DOI: [10.1186/s13104-019-4121-7](https://doi.org/10.1186/s13104-019-4121-7)
- Access: [Figshare](https://doi.org/10.6084/m9.figshare.7379186)
- License: CC BY 4.0

**CAMELYON16**
- Specimen/stain: WSI, H&E sentinel lymph node sections, TIFF
- Size: 400 WSIs (270 train, 130 test)
- Labels: slide-level normal/metastasis + lesion-level XML polygons
- Citation: Ehteshami Bejnordi, B. et al., *JAMA* 318(22):2199-2210, 2017
- Access: [camelyon16.grand-challenge.org/Data](https://camelyon16.grand-challenge.org/Data/); GigaScience/AWS/Baidu
- License: CC0

**CAMELYON17**
- Specimen/stain: WSI, H&E lymph node (+ cytokeratin IHC verification), TIFF
- Size: 1,000 WSIs / 200 patients
- Labels: lesion-level (micro/macro/ITC) + patient-level pN-stage
- Citation: Bándi, P. et al., *IEEE Trans. Medical Imaging* 38(2):550-560, 2019, DOI: [10.1109/TMI.2018.2867350](https://doi.org/10.1109/TMI.2018.2867350)
- Access: [camelyon17.grand-challenge.org/Data](https://camelyon17.grand-challenge.org/Data/); GigaScience/AWS/Baidu
- License: CC0
- **Camelyon+** (2025 re-curated version, 1,350 WSIs): Ling, X. et al., *Scientific Data*, 2025, DOI: [10.1038/s41597-025-05586-5](https://doi.org/10.1038/s41597-025-05586-5). Access: [ScienceDB](https://doi.org/10.57760/sciencedb.16442), [GitHub](https://github.com/lingxitong/CAMELYON-PLUS-BENCHMARK)

**CPTAC-BRCA (pathology slides)**
- Specimen/stain: WSI paired with proteogenomic data
- Size: 14 breast cases with CPTAC phase-2 proteomic data (overlaps TCGA-BRCA patient IDs)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70227748); proteomic data at [PDC](https://pdc.cancer.gov/pdc/browse)
- License: TCIA standard (CC BY 3.0 typical, not independently confirmed for this collection)
- Also see the **clinical/genomic companion** in [Section 6](#6-clinical-genomic--biomarker-data).

**HEROHE Challenge**
- Specimen/stain: WSI, H&E only, 3D Histech Pannoramic 1000, MIRAX
- Size: 510 cases (360 train, 150 test)
- Labels: binary HER2 status predicted from H&E alone
- Citation: Conde-Sousa, E. et al., *J. Imaging* 8(8):213, 2022, DOI: [10.3390/jimaging8080213](https://doi.org/10.3390/jimaging8080213)
- Access: [ecdp2020.grand-challenge.org/Dataset](https://ecdp2020.grand-challenge.org/Dataset/)
- License: CC BY-NC-ND 3.0

**HER2 Challenge Contest (Warwick)**
- Specimen/stain: WSI, paired H&E + HER2 IHC
- Size: 86 invasive breast carcinoma cases
- Labels: HER2 IHC score (0/1+/2+/3+), consensus ground truth
- Citation: Qaiser, T. et al., *Histopathology*, 2018, DOI: [10.1111/his.13333](https://doi.org/10.1111/his.13333)
- Access: University of Warwick TIA Centre — **[UNVERIFIED]**, page returned 403 on fetch
- License: not confirmed

**HER2-IHC-40x** *(new, 2025)*
- Specimen/stain: WSI + patches, HER2 IHC, 40×
- Size: 107 WSIs, 1024×1024px patches
- Labels: 4-class HER2 score
- Citation: *Data in Brief*, 2025
- Access: [Zenodo](https://zenodo.org/records/15179608)
- License: **[UNVERIFIED]**, likely open (Zenodo)

**IDC Histology Patches (Janowczyk/Kaggle)**
- Specimen/stain: Patches, H&E, 40×, 50×50px
- Size: 277,524 patches (198,738 negative / 78,786 IDC-positive) from 162 WSIs
- Citation: Cruz-Roa, A. et al., SPIE Medical Imaging, 2014; curated by Janowczyk & Madabhushi, *J. Pathology Informatics*, 7:29, 2016
- Access: [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
- License: Public domain / research use

**IMPRESS**
- Specimen/stain: Multi-stain WSI, H&E + multiplex IHC (PD-L1, CD8, CD163)
- Size: 126 WSIs / 126 patients (62 HER2+, 64 TNBC), post-NAC
- Citation: Huang, Z. et al., *npj Precision Oncology*, 2023, DOI: [10.1038/s41698-023-00352-5](https://doi.org/10.1038/s41698-023-00352-5)
- Access: pipeline code at [GitHub](https://github.com/huangzhii/IMPRESS); **raw WSI not published — contact corresponding author**
- License: code MIT; data access restricted

**MITOS (ICPR 2012 Contest)**
- Specimen/stain: H&E, 50 HPFs from 5 slides, 40×
- Size: 50 HPFs, 326 mitotic cells annotated
- Citation: Roux, L. et al., *J. Pathology Informatics*, 2013
- Access: original site defunct; unofficial mirror at [GitHub](https://github.com/znck/mitosis-detection/tree/master/datasets/ICPR%202012) — **[UNVERIFIED]**, not canonical
- License: not stated

**MITOS-ATYPIA-14**
- Specimen/stain: H&E, Aperio + Hamamatsu scanners, 20×/40×
- Size: 1,200 training frames (16 biopsies), 496 test images (5 biopsies)
- Labels: nuclear atypia score (1-3) + mitosis annotations
- Access: [mitos-atypia-14.grand-challenge.org/Dataset](https://mitos-atypia-14.grand-challenge.org/Dataset/)
- License: **[UNVERIFIED]**

**PatchCamelyon (PCam)**
- Specimen/stain: 96×96px patches, H&E lymph node, derived from CAMELYON16
- Size: 327,680 patches (262,144/32,768/32,768 train/val/test)
- Citation: Veeling, B.S. et al., MICCAI 2018, [arXiv:1806.03962](https://arxiv.org/abs/1806.03962)
- Access: [GitHub](https://github.com/basveeling/pcam); Zenodo, TensorFlow Datasets
- License: CC0 (data), MIT (code)

**Post-NAT-BRCA**
- Specimen/stain: WSI, H&E, Aperio 20×, .svs
- Size: 96 WSIs / 54 patients (43.2 GB)
- Labels: tumor cellularity, 12 cell-type classes
- Citation: Martel, A.L. et al., TCIA, 2019, DOI: [10.7937/TCIA.2019.4YIBTJNO](https://doi.org/10.7937/TCIA.2019.4YIBTJNO)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758117)
- License: CC BY 3.0

**SLN-Breast (Breast Metastases to Axillary Lymph Nodes)**
- Specimen/stain: WSI, H&E axillary lymph node, Leica Aperio AT2, 20×
- Size: 130 WSIs / 78 patients (36 positive/27 patients, 94 negative)
- Citation: Campanella, G. et al., TCIA, 2019, DOI: [10.7937/tcia.2019.3xbn2jcc](https://doi.org/10.7937/tcia.2019.3xbn2jcc)
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52763339), 53 GB
- License: CC BY 3.0

**TCGA-BRCA (pathology/tissue slides)**
- Specimen/stain: WSI, H&E, 40×, SVS, heterogeneous scanners
- Size: 977 diagnostic WSIs (779 IDC, 198 ILC) — slide-level subtype labels only
- Citation: TCGA Network, "Comprehensive molecular portraits of human breast tumours," *Nature* 490(7418):61-70, 2012, DOI: [10.1038/nature11412](https://doi.org/10.1038/nature11412)
- Access: [GDC Data Portal](https://portal.gdc.cancer.gov/projects/TCGA-BRCA) (standard, JS-rendered). **Note**: the TCIA "TCGA-BRCA" wiki page actually indexes *radiology* (230,167 MR/mammography images) — pathology slides are a separate GDC-hosted set, do not conflate.
- License: CC BY 3.0 (TCIA imaging); GDC data governed by NIH/GDC policies
- Also see the **clinical/genomic companion** in [Section 6](#6-clinical-genomic--biomarker-data).

**TIGER Challenge (Tumor-Infiltrating Lymphocytes)**
- Specimen/stain: WSI, H&E, core-needle + surgical resections
- Size: 370 training WSIs; cohort n=3,708 early-stage cases (1,938 TNBC, 1,770 HER2+)
- Labels: 7-class tissue segmentation, TIL/plasma-cell boxes, slide-level TILs scores
- Citation: TIGER consortium, *Nature Communications*, 2026
- Access: [tiger.grand-challenge.org/Data](https://tiger.grand-challenge.org/Data/); [AWS Open Data](https://registry.opendata.aws/tiger/)
- License: CC BY-NC 4.0 (RUMC/JB slides); TCGA-derived slides retain original TCGA license

**TUPAC16 (Tumor Proliferation Assessment Challenge)**
- Specimen/stain: WSI, H&E, Aperio .svs, from TCGA
- Size: 500 training + 321 test WSIs; auxiliary mitosis/ROI subsets
- Labels: mitotic score, PAM50 proliferation score
- Citation: Veta, M. et al., *Medical Image Analysis*, 2019, DOI: [10.1016/j.media.2018.10.010](https://doi.org/10.1016/j.media.2018.10.010)
- Access: [tupac.grand-challenge.org/Dataset](https://tupac.grand-challenge.org/Dataset/)
- License: **[UNVERIFIED]**

**Digital Pathology Dataset for Breast Cancer Diagnosis (Bahçeşehir University)** *(new, 2024)*
- Specimen/stain: WSI, IHC (163) + H&E (72), .svs
- Size: 235 slides, 11.2 TB
- Access: [Zenodo](https://zenodo.org/records/14131968)
- License: CC BY 4.0

**High-Resolution Digital Pathology Imaging of Breast Cancer** *(new, 2025)*
- Specimen/stain: Core-needle biopsy, H&E, 40×, .tif
- Size: 157 patients, pre-treatment NAC biopsies
- Labels: age, tumor size, nodal status, grade, ER/PR/HER2, pCR/RCB response
- Citation: Tran, W. et al., Health Data Nexus, 2025, DOI: [10.57764/pm3v-b131](https://doi.org/10.57764/pm3v-b131)
- Access: [healthdatanexus.ai](https://healthdatanexus.ai/content/breastcancerimaging/1.0.0/) — **restricted**: credentialed user status, TCPS 2 CORE 2022 training, signed DUA
- License: Health Data Nexus Contributor Review Health Data License 1.0

**Multi-Center Breast FNAC Whole-Slide Cytology Dataset (C1-C5)** *(new, 2026 preprint)*
- Specimen/stain: WSI **cytology** (fine needle aspiration, not tissue histopathology), Papanicolaou/May-Grünwald-Giemsa, 40×
- Size: 470 WSIs / 321 patients (India, multi-center)
- Citation: [arXiv:2606.30209](https://arxiv.org/abs/2606.30209)
- Access: Zenodo (manifest-linked, exact record ID **[UNVERIFIED]**)
- License: not confirmed

### Excluded
- **SICAPv2** — confirmed prostate cancer, not breast.
- **ANHIR** (registration, not breast-specific), **GTEx-breast** (normal tissue, not cancer), canine/synthetic breast datasets.

---

## 6. Clinical, Genomic & Biomarker Data

**BCSC (Breast Cancer Surveillance Consortium) Risk Factor Dataset**
- Data type: Clinical/tabular risk-factor data linked to mammography outcomes; multi-site US registry (public Risk Factor Dataset is a de-identified subset of millions of screening records)
- Key variables: age, family history, breast density, prior biopsy, race/ethnicity, menopausal status, BMI, screening/diagnostic outcome
- Access: [bcsc-research.org/datasets/rf](https://www.bcsc-research.org/index.php/datasets/rf) — **[UNVERIFIED]**, host unreachable at check time; custom linked datasets require a data request/DUA
- License: Not open-CC; citation/acknowledgment required (NCI grant P01CA154292)

**CPTAC-BRCA (proteogenomics)**
- Data type: Proteomics + phosphoproteomics + matched genomics/transcriptomics; 105-tumor original discovery cohort, 122-tumor expanded cohort
- Key variables: protein/phosphoprotein abundance, somatic mutations, CNA, RNA-seq, PAM50 subtype
- Citation: Mertins, P. et al., *Nature* 534, 55-62 (2016), DOI: [10.1038/nature18003](https://www.nature.com/articles/nature18003); Krug, K. et al., *Cell* 183(5), 1436-1456 (2020)
- Access: [Proteomic Data Commons](https://pdc.cancer.gov/pdc/browse); imaging companion at [TCIA](https://www.cancerimagingarchive.net/collection/cptac-brca/)
- License: CC BY (TCIA imaging portion); PDC terms apply otherwise

**GEO GSE2034** (Wang et al. lymph-node-negative expression series)
- Data type: Gene expression microarray (Affymetrix HG-U133A), 286 lymph-node-negative primary tumors
- Key variables: relapse-free survival, distant metastasis status
- Citation: Wang, Y. et al., *Lancet* 365(9460):671-679 (2005)
- Access: [GEO GSE2034](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034)
- License: GEO standard public-data terms

**GSE96058 / SCAN-B** (Sweden Cancerome Analysis Network-Breast)
- Data type: RNA-seq, 3,273-sample validation cohort (GSE96058) + 405-sample training cohort (GSE81538); prospective SCAN-B enrollment now >14,000 patients
- Key variables: ER/PR/HER2/Ki67, Nottingham grade, PAM50 subtype
- Citation: Brueffer, C. et al., *JCO Precision Oncology* (2018)
- Access: [GEO GSE96058](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058)
- License: GEO standard public-data terms

**I-SPY2 (clinical/biomarker annotations)**
- Data type: Clinical + biomarker data linked to serial breast MRI in a neoadjuvant adaptive trial (719 patients, 985 combined with ACRIN-6698)
- Key variables: pCR, HR status, HER2 status, MammaPrint risk level, functional tumor volume
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230072), clinical CSV
- License: CC BY 4.0
- Imaging entry: [Section 3](#3-mri).

**METABRIC**
- Data type: Clinical/tabular + targeted DNA sequencing + copy-number + microarray expression; 2,509 primary tumors, 548 matched normals
- Key variables: PAM50/IntClust subtype, ER/PR/HER2, stage/grade, treatment, survival, mutation/CNA calls
- Citation: Curtis, C. et al., *Nature* 486, 346-352 (2012), DOI: [10.1038/nature10983](https://www.nature.com/articles/nature10983); Pereira, B. et al., *Nat Commun* 7, 11479 (2016)
- Access: [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric); raw sequence data via EGA (controlled-access)
- License: cBioPortal terms; EGA raw data controlled-access

**MSK-IMPACT — Breast Cancer Cohort**
- Data type: Targeted panel sequencing (410-505 gene assay); 3,879 breast cases (parent pan-cancer cohort now 25,000+ patients)
- Key variables: somatic mutations, CNA, structural rearrangements, tumor purity, clinical stage
- Citation: Zehir, A. et al., *Nature Medicine* 23(6):703-713 (2017); breast-specific `breast_msk_2025` (*Nat Genet* 2025)
- Access: [cBioPortal](https://www.cbioportal.org/study/summary?id=breast_msk_2025); [MSKCC data catalog](https://datacatalog.mskcc.org/dataset/10438)
- License: cBioPortal standard terms

**OPTIMAM (OMI-DB) — clinical data companion**
- Data type: Screening/diagnostic clinical and pathology metadata linked to mammography images; ~750,000 cases (UK NHS Breast Screening Programme)
- Key variables: screening outcome, recall status, biopsy/pathology results, cancer diagnosis, laterality
- Access: [medphys.royalsurrey.nhs.uk/omidb](https://medphys.royalsurrey.nhs.uk/omidb/) — controlled access, Data Access Committee review
- License: No open CC license; data-sharing agreement required
- Imaging entry: [Section 1](#1-mammography).

**PLCO (Prostate, Lung, Colorectal, and Ovarian Cancer Screening Trial) — Breast Dataset**
- Data type: Clinical/tabular screening + risk-factor + outcome data; ~78,000 women
- Key variables: screening history, risk factors, ER/PR status, biopsy, histology, staging, mortality follow-up
- Access: [CDAS](https://cdas.cancer.gov/datasets/plco/19/) — controlled access, requires approved CDAS project + data transfer agreement
- License: Governed by CDAS Data Use Agreement

**SEER (Surveillance, Epidemiology, and End Results) — Breast Cancer Cohort**
- Data type: Population-based cancer registry, clinical/tabular; hundreds of thousands of breast cases (1975-2023 release)
- Key variables: demographics, tumor site/morphology, stage, first course of treatment, survival, ER/PR/HER2 (recent fields)
- Access: [seer.cancer.gov](https://seer.cancer.gov/) — SEER*Stat, mostly no-DUA public-use files; some limited-use files require a DUA
- License: Public domain (US government work); DUA for limited-use identifiable extracts

**TCGA-BRCA (clinical/genomic)**
- Data type: Multi-omic — clinical, WGS/WES mutations, copy-number, RNA-seq, miRNA-seq, methylation, RPPA; 1,098 cases
- Key variables: PAM50 subtype, ER/PR/HER2, pathologic stage, survival, mutation/CNA burden, histologic type
- Citation: TCGA Network, *Nature* 490, 61-70 (2012), DOI: [10.1038/nature11412](https://www.nature.com/articles/nature11412)
- Access: [GDC Data Portal](https://portal.gdc.cancer.gov/projects/TCGA-BRCA); also [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018). Open for de-identified summary data; raw BAM/VCF controlled-access via dbGaP
- License: NIH/GDC data use policy
- Pathology entry: [Section 5](#5-pathology--histopathology).

**WBCD / WDBC (Wisconsin Diagnostic Breast Cancer Dataset)**
- Data type: Tabular cytology features from FNA digitized images; 569 instances (357 benign, 212 malignant), 30 numeric features
- Key variables: radius/texture/perimeter/area/smoothness/compactness/concavity/symmetry/fractal dimension (mean/SE/worst), diagnosis
- Citation: Wolberg, W.H. et al., UCI ML Repository (1993/1995), DOI: [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B)
- Access: [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) — fully open
- License: CC BY 4.0

### Checked but not included
- **Gail Model / BCRAT training data** — not a standalone distributable dataset; built from BCDDP + SEER (already covered) plus race/ethnicity-specific extensions from other cohort studies.
- **Mirai / OncoNet-related cohorts** — no standalone public dataset distinct from private institutional mammography cohorts (MGH, Karolinska).
- **CPTAC Data Portal (2015)** — superseded by the Proteomic Data Commons; historical citation only.

---

## 7. Cross-Cutting Notes, Gaps & Naming Disambiguation

**Verification caveats** (apply across all sections): several primary hosts are JavaScript-rendered single-page apps (GDC, cBioPortal, PDC, Kaggle) that couldn't be scraped directly — where possible these were verified via their underlying JSON APIs instead; where not, the entry is marked **[UNVERIFIED]**. A handful of long-standing academic hosts (BCDR, INbreast, BUSIS, UDIAT, STU-Hospital) were unreachable at check time despite being frequently cited in the literature — this likely reflects aging infrastructure rather than the datasets not existing; access may still be possible by directly emailing the original authors.

**Modality gaps confirmed by independent research passes**: standalone public breast elastography datasets and real (non-phantom, non-simulated) photon-counting dedicated-breast-CT datasets do not appear to exist publicly as of this writing.

**Datasets spanning multiple sections** (same cohort, genuinely distinct public releases — not duplicates):
- **TCGA-BRCA**: pathology WSIs ([Section 5](#5-pathology--histopathology)) vs. clinical/genomic/multi-omic data ([Section 6](#6-clinical-genomic--biomarker-data)) — note the TCIA "TCGA-BRCA" collection page indexes *radiology*, not pathology; pathology slides are GDC-hosted separately.
- **CPTAC-BRCA**: pathology slides ([Section 5](#5-pathology--histopathology)) vs. proteogenomic data ([Section 6](#6-clinical-genomic--biomarker-data)).
- **I-SPY2**: MRI imaging ([Section 3](#3-mri)) vs. clinical/biomarker annotations ([Section 6](#6-clinical-genomic--biomarker-data)).
- **OPTIMAM/OMI-DB**: mammography images ([Section 1](#1-mammography)) vs. linked clinical/pathology metadata ([Section 6](#6-clinical-genomic--biomarker-data)).
- **QIN-Breast (67-subject cohort)**: appears as both a PET/CT entry ([Section 4](#4-ct--less-common-imaging-modalities)) and an MRI entry ([Section 3](#3-mri)) — it is one multi-sequence TCIA collection.
- **CDD-CESM**: filed under Mammography ([Section 1](#1-mammography)) as its primary home; cross-referenced from CT/other modalities ([Section 4](#4-ct--less-common-imaging-modalities)) since CESM is dual-energy/contrast-based.

**Naming collisions worth double-checking before use**:
- "BUS-BRA" = "BUSBRA" (same dataset); "BUSC," "BUS_UC," "BUS-UCLM," and "BUSI_WHU" are four **distinct** breast ultrasound datasets.
- "BUID" is an informal alias for the QAMEBI database, not an independent release.
- "BMCD" and "mini-MIAS-2" could not be confirmed as datasets distinct from KAU-BCMD and mini-MIAS respectively — likely naming confusion in secondary literature.

**License spread**: most recent releases use CC BY 4.0, but exceptions matter for downstream use — note **CC BY-NC** variants (Duke-Breast-Cancer-MRI, BCS-DBT, MAMA-MIA, BUS-UCLM, BACH, BRACS, TIGER) restrict commercial use, and several large screening-registry datasets (EMBED, OPTIMAM, VinDr-Mammo, PLCO, BCSC) require a signed DUA or institutional application rather than being open-download.
