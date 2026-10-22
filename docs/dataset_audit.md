# Dataset Audit — First Raw-Stage Run (2026-10-07)

Results of `scripts/analyze_datasets.py --dataset all --cross` (method:
[dataset_analysis_plan.md](dataset_analysis_plan.md)). Per-dataset tables are in
`analysis/<dataset>/` (`summary.json`, `duplicates.csv`, `leakage.csv`,
`outliers*.csv`); figures and per-image parquet files are local only.

"Used" = rows the training loader includes; splits are the CLI defaults
(seed 42, `--validation-fraction 0.1`, 5 folds).

## Claimed vs. verified (table A, abridged)

| Dataset | Reported | On disk | Used by loader | Notes |
|---|---|---|---|---|
| BUS-BRA | 1,875 img / 1,064 pts | 1,875 | 1,875 | 0 duplicates; text overlays on 898, calipers/markers on 349 |
| BUSI | 780 (437 B / 210 M / 133 N) | 780 (437 / 210 / 133) | 647 | full release is present (the "163/780" note in the reproduction plan is stale) |
| BUSC | not stated | 250 | 250 | 16 near-duplicate groups, consecutive IDs |
| BrEaST | 256 (154 B / 98 M / 4 N) | 256 | 252 | 0 duplicates |
| MIAS | 322 (64 B / 51 M / 207 N) | 322 | 111 images (lesion rows) | matches reported counts |
| CDD-CESM | 2,006 img / 326 pts | 2,001 | 586 (subtracted CESM, B/M) | 5 annotated images missing on disk; 2 machine types (1,875 / 126) |
| CMMD | 5,202 img / 1,775 pts | 5,194 | 3,738 (1,112 B / 2,626 M) | 1,456 DICOMs have no clinical label (other breast); DICOMs are **8-bit** |
| BCS-DBT (val release) | — | 1,163 views / 280 pts | 75 (38 B / 37 Cancer, 40 pts) | single vendor (Hologic Selenia Dimensions), 10-bit |
| BreaDM | 232 pts (85 B / 147 M) | 1,765 slices / 232 pts | 1,319 (train+val) | official **test** split (446 slices) is never used |
| ISPY2 | 982 pts, malignant only | 37,180 slices / 982 pts | 37,180 | see MRI findings |

## Findings that affect training or evaluation

1. **BUSI duplicates leak across splits.** 34 near-duplicate groups (SSIM ≥ 0.95,
   confirmed visually); **28 pairs straddle the hold-out split or the 5 folds**,
   and **4 pairs carry conflicting labels** (e.g. `benign (131)` ≡
   `malignant (51)`). BUSI has no patient IDs, so the splitter can't group them.
   → Group duplicate clusters before splitting; drop or adjudicate the conflicts.
2. **BUSC near-duplicates leak (36 pairs).** 16 groups of consecutive-ID frames
   (us18/us19, us61/us62, …), almost certainly repeat captures of one lesion.
   → Same fix: treat each duplicate group as one "patient" for splitting.
3. **CMMD has the same images under two patient IDs.** SSIM = 1.0 pairs:
   D1-0202 ≡ D2-0284 (4 images), D1-0014 ≡ D1-0269, and **D1-0808 ≡ D1-1292
   with Benign vs. Malignant labels**. No leak with the default seed (all in
   train), but other seeds/folds could split them.
   → Merge these IDs as one group; review the label conflict.
4. **MRI source shortcut in `combined`.** A logistic regression on five basic
   per-slice metrics (crop size, brightness, contrast, entropy) separates ISPY2
   from BreaDM *malignant* slices with **AUC 0.97** (ISPY2 crops median 87×75 px
   vs. BreaDM 26×26). ISPY2 is malignant-only, so the MRI branch can learn
   "ISPY2-looking ⇒ malignant".
   → Match crop geometry/intensity normalization between the two sources, and
   report MRI results per source.
5. **ISPY2 `img9Se` channels repeat phases.** 941 / 982 patients have fewer than
   9 DCE phases (range 4–11, median 7); `preprocess_ispy2_mri.py` samples 9 with
   `linspace`, so channels are duplicated and their temporal meaning differs
   from BreaDM's 9 channels.
6. **ISPY2 tumor masks are fragmented** (100+ connected components per case),
   so the per-slice crops include many tiny tumor fragments.
7. **Burnt-in annotations are common in ultrasound** (text on 898/1,875 BUS-BRA
   and 108/780 BUSI images; calipers or measurement lines on 349 BUS-BRA and
   79 BUSI). These are shortcut candidates to test (plan section R).

## Findings that affect storage

- **CMMD DICOMs are 8-bit**, so the cached 8-bit PNGs lose no bit depth.
- **BCS-DBT DICOMs are 10-bit**, so the 8-bit crops do lose precision; keep uint16
  if they are ever re-prepared.
- **BreaDM `seg3D`** holds integer 0–255 values for all 232 cases, but 147 of them
  are stored as float64; a lossless uint8 cast cuts the folder from ~7.8 G to ~1 G.

## Detector notes

- The paper's Doppler rule fires on 758/780 BUSI images and its >10-Hough-lines
  marker rule on 732/780. Both mostly respond to ordinary bright tissue, so they are kept only
  as `paper_*` columns. No dataset has real color-Doppler images by the HSV
  detector.
- No cross-dataset duplicates within ultrasound or within mammography.
- Outlier counts (z > 3 or isolation forest) are review queues, not
  exclusions: decisions go in each `outliers_review.csv`.

## Newly downloaded datasets (audit run 2026-10-13)

Neither is wired into training yet, so "used" = every benign/malignant sample
and "split" = the dataset's official train/test assignment.

| Dataset | On disk | Used | Notes |
|---|---|---|---|
| CBIS-DDSM (crops + full sample) | 2,006 lesion crops + 187 readable full mammograms, 1,226 patients | 2,006 crops: 839 benign, 379 benign-without-callback, 788 malignant (1,477 train / 529 test) | 16-bit; 1,902 crops come with a full-size ROI mask; 1 full mammogram (`Mass-Test_P_00516_LEFT_MLO`) has a corrupt header |
| BCS-DBT training + test (Benign/Cancer) | 321 views, 161 patients | 321 (185 Benign / 136 Cancer; 200 train / 121 test) | Hologic Selenia Dimensions only, 10-bit; no patient overlap with the validation release we already had |

Findings:

1. **No duplicates** within either dataset, and none across all six
   mammography sets (MIAS, CDD-CESM, CMMD, BCS-DBT validation + train/test,
   CBIS-DDSM; 4.1 M hash candidates checked with SSIM).
2. **CBIS-DDSM's official split mixes patients.** 16 patients have a mass in
   one split and a calcification in the other (57 crops), because CBIS-DDSM
   assigned train/test separately for the mass and calcification subsets.
   → When using mass and calcification crops together, split by patient
   (`patient_id`) instead of using the official train/test folders.
3. **"BENIGN_WITHOUT_CALLBACK" is not biopsy-confirmed.** These 379 crops
   (304 of them calcifications) were judged benign without further work-up.
   Only BENIGN and MALIGNANT are pathology-confirmed. → Report results with
   and without them, or drop them for a strictly pathology-confirmed set.
4. **Film labels are common in the full mammograms:** burnt-in labels outside
   the breast on 113 of 187 (60%). They are invisible in the tight crops, so
   crop-based training is unaffected; whole-image training would need masking.
5. **Detector caveat:** the implant heuristic fires on 79 crops and 26 full
   images; on crops it is reacting to bright calcifications and on film to
   saturation, not implants. Treat that flag as unreliable for CBIS-DDSM.

## BUS-CoT (audit run 2026-10-14)

BUS-CoT (Yu et al., arXiv 2509.17046; Figshare, CC BY 4.0) is an **aggregate of
public sources**, not new clinical data: "open-access papers, publicly
available case studies (Radiopaedia, PubMed) and open-access datasets", with
histopathology categories "extracted from description texts" and image
features / BI-RADS annotated by six senior radiologists. 11,439 images,
4,838 patients.

| Source inside BUS-CoT | Images | Overlap with this project | Internal duplicates |
|---|---|---|---|
| Web data / PMC / ROCO / PatternFigures | 5,121 | — | many (scraped figures) |
| TFUSL (likely *The Free UltraSound Library*, SonoSkills/FUJIFILM teaching cases) | 2,472 | none | 5% of images, 17 label-conflict pairs |
| BUS-BRA | 1,875 | **all 1,875 = our `busbra`** | — |
| TUHD (likely Thammasat University data, Thailand) | 886 | none | **36% of images, 88 label-conflict pairs** |
| BUSI | 780 | **all 780 = our `busi`** | — |
| UDIAT (Dataset B), STU-Hospital, Birads-ultrasound | 305 | none | ≤5% |

Findings:

1. **Never combine BUS-CoT's BUSI/BUS-BRA copies with our own** — every one of
   those 2,655 images is identical to an image we already train on.
2. **BUS-CoT relabels 8 BUSI images** (e.g. our `benign (433)` is Malignant in
   BUS-CoT, SSIM 1.0), several of them the same BUSI images our audit already
   flagged as conflicting duplicates. Useful as an independent second opinion.
3. **Genuinely new usable data: 3,599 images / 1,050 patients** (TFUSL, TUHD,
   UDIAT, Birads-ultrasound; 1,572 benign / 2,027 malignant), but the Thammasat
   part needs de-duplication first, and labels are only as good as each source
   (teaching-case text for TFUSL).
4. Two images failed to load (2 of 11,439).

## ODELIA Breast MRI Challenge 2025 (audit run 2026-10-14)

1,482 breasts / 741 patients from six European centres (unilateral
configuration, 256×256×32). Labels per breast: 978 no lesion, 195 benign,
309 malignant, assigned by radiologists "based on histopathological or 2-year
follow-up information" → **partially pathology-confirmed**. Files verified
against Hugging Face SHA-256 (94/94 shards). No load errors, no duplicates.

| Centre | Benign | Malignant | Benign share of lesions | Post-contrast phases |
|---|---|---|---|---|
| CAM (Cambridge) | 45 | 97 | 32% | 2 |
| MHA (Athens) | 15 | 52 | 22% | 4 |
| RSH (Valencia; challenge OOD test centre) | 16 | 61 | 21% | 5 |
| RUMC (Nijmegen) | 2 | 18 | 10% | 4 |
| **UKA (Aachen)** | **75** | **32** | **70%** | 4 |
| UMCU (Utrecht) | 42 | 49 | 46% | 5 |

Findings:

1. **Centre shortcut.** A logistic regression on 8 whole-breast statistics
   identifies the centre with 93% accuracy (chance 34%). The centre alone
   predicts benign vs malignant with AUC 0.71, and the same image statistics
   reach only 0.76 — most of that signal is the centre. UKA (2D, no fat
   suppression, TR ≈ 263 ms) contributes 38% of all benign lesions.
   → Evaluate leave-one-centre-out (at minimum keep RSH unseen), report per
   centre, and consider centre-balanced sampling.
2. **Phase count differs by centre** (CAM 2, others 4–5), so fixed multi-phase
   channel stacks would repeat or pad phases per centre (the same problem as
   ISPY2's img9Se). → Use Pre, Sub_1 and T2, which every centre has (the
   ODELIA paper's own input choice).

## AMBL — Advanced-MRI-Breast-Lesions (audit run 2026-10-19)

The 99 AMBL patients with a DICOM SEG (IDC v25: original DCE multiphase, axial
T2 and SEG; 297 series, 34.4 GB; GE 1.5 T, 5 DCE timepoints, 512×512,
0.66 mm × 0.66 mm × 2 mm). Clinical XLSX (Sep 2024): 200 patients with
per-lesion tumour/benign, side and pathology. Module `audit/ambl.py`; outputs
in `analysis/ambl/` (`cases.parquet` has one row per lesion,
`lesions_resolved.csv` the resolved labels).

| | Count |
|---|---|
| Patients / SEG segments | 99 / 165 |
| Clinical lesions for these patients | 166 (83 tumour / 83 benign) |
| Lesion count matches SEG | 98 of 99 patients (AMBL-024: 3 in XLSX, 2 segments) |
| **Labels resolvable from metadata** | **138 lesions (70 malignant / 68 benign), 92 patients** |
| Ambiguous (tumour and benign in the same breast) | 27 lesions |
| Load errors | 0 |

Findings:

1. **The SEG does not carry the labels, and its order is not reliable.**
   - The segmentation references the *registered* DCE series. That series
     shares the original's slice grid (positions and orientation agree to
     float32 rounding on every checked patient), so masks map onto the
     original DCE without the extra 30 GB download.
   - Every segment is labelled "Mass". The XLSX data dictionary says segments
     list tumours first, then benign lesions.
   - Checked against each lesion's L/R position, that rule puts the mask in
     the wrong breast for 20% of lesions (tumour and benign swapped, e.g.
     AMBL-008, AMBL-011), and it disagrees with side-resolved labels for 17%.
   - → Labels are resolved per patient (`resolve_labels`):
     - one label for all of a patient's lesions (109 lesions);
     - otherwise by the mask's side when each breast holds one label (29);
     - otherwise excluded as ambiguous (27).
2. **Benign labels are not all pathology-confirmed.** The dictionary says
   benign lesions not biopsied were called benign after at least 1 year of
   follow-up. → Treat as *partially* confirmed, like BrEaST and ODELIA.
3. **The resolved labels behave like real malignancy.**
   - Malignant lesions enhance faster: initial enhancement median 1.01 vs.
     0.61, AUC 0.77 on its own.
   - Most benign lesions keep enhancing to the last phase (46 of 68 peak at
     phase 4, vs. 23 of 70 malignant).
   - Lesion-to-surroundings contrast: AUC 0.73.
4. **Size alone separates the classes (AUC 0.83).** Malignant lesions are
   median 2.8 ml over 10 slices, benign 0.37 ml over 4 slices. A classifier on
   fixed-size crops can learn size, so report size-matched results or a
   size-only baseline next to any AMBL model.
5. **Masks are fragmented.** Median connected components are 15 (malignant)
   and 4 (benign), from the thresholded annotation (subtraction > 75), like
   ISPY2's. → Per-slice crops should use the lesion's bounding box over all
   components, not the largest component.
6. **Six patients (7 lesions, all malignant) are post-neoadjuvant scans**
   (referral reason 6). The tumours are imaged after treatment. → Exclude them
   from a diagnosis benchmark, or flag them.
7. **No duplicates.** The thumbnail detector grouped 142 of 165 lesion slices
   (SSIM ≥ 0.95 at 64 px), but full-resolution SSIM of the closest pairs is
   0.47–0.62. Single-scanner whole-breast slices look alike when downsampled.
   → For MRI, run duplicate checks at full resolution or on the lesion crop.
8. **The labels differ from the BreastDCEDL_AMBL benchmark**, which reports
   88 patients / 133 lesions with 89 benign and 44 malignant. Our 99 patients
   give 166 clinical lesions (83/83), so their selection or label mapping
   differs from the TCIA release. → Compare against their numbers only after
   matching their patient list (Zenodo 17189101).

## BUS-UCLM (audit run 2026-10-22)

Mendeley Data v3: 683 ultrasound images (856×606, one scanner, Siemens ACUSON
S2000), 38 patients by filename prefix. INFO.csv labels: 419 normal, 174 benign,
90 malignant. Masks are RGB (green benign, red malignant). Adapter
`bus_uclm` in `audit/sources.py`; outputs in `analysis/bus_uclm/`. No load
errors.

Findings:

1. **CHCO and CHVI are the same patient.** `CHCO_000`-`003` are pixel-identical
   (SSIM 1.000) to `CHVI_005/008/009/013`. → Merge the two prefixes into one
   patient before splitting. The other 10 near-duplicate pairs (SSIM ≥ 0.95)
   are consecutive frames of one patient, which a patient-level split already
   handles. No label conflicts.
2. **Labels and masks agree on every image.** Benign masks are green,
   malignant red, normal empty.
3. **Burnt-in calipers are common but carry little label information.** INFO
   marks them on 52% of benign and 60% of malignant images (AUC 0.54 on its
   own). Doppler appears on 21% of malignant vs. 10% of benign images (AUC
   0.56). → Keep them, but report a marks-only baseline.
4. **Our caliper detector does not recognise this scanner's markers.** It
   agrees with INFO's `Marks` flag on only 78% of images and fires on almost
   none. → Use INFO's flags for this dataset.
5. **Lesion size is a moderate cue.** Bounding-box area alone gives AUC 0.66
   (malignant median 249×207 px vs. benign 217×131), far weaker than AMBL's
   0.83.
6. **Patients contribute several labels.** 5 of 38 patients have both benign
   and malignant images, and 35 also have normal images. Median 16 images per
   patient, maximum 39. → Group by patient (with finding 1 merged), and expect
   only about 18 benign-only and 10 malignant-only patients for stratified
   splits.
7. **How labels were confirmed is not stated** on the dataset page, so the
   labels are treated as "not stated" (like BUSI) until the source paper is
   checked.
