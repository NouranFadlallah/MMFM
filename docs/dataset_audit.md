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
- **BreaDM `seg3D`** holds integer 0–255 values in float64 for all 232 cases, so
  a lossless uint8 cast cuts ~7.8 G to ~1 G.

## Detector notes

- The paper's Doppler rule fires on 758/780 BUSI images and its >10-Hough-lines
  marker rule on 732/780. Both mostly respond to ordinary bright tissue, so they are kept only
  as `paper_*` columns. No dataset has real color-Doppler images by the HSV
  detector.
- No cross-dataset duplicates within ultrasound or within mammography.
- Outlier counts (z > 3 or isolation forest) are review queues, not
  exclusions: decisions go in each `outliers_review.csv`.
