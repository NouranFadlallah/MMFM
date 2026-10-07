# Storage Management Plan

Goal: free enough local disk to download more datasets, without losing the
ability to (a) train exactly as today and (b) rebuild any dataset from scratch.

**Status: plan only — nothing implemented or deleted yet.** Deletion of any raw
data is gated on [dataset_analysis_plan.md](dataset_analysis_plan.md) having been
run and reviewed for that dataset first (see §5).

## 0. Current state (measured 2026-10-06)

| Disk | Size | Free |
|---|---|---|
| `/` (nvme, holds repo, `~/Downloads`, `~/.cache`) | 146 G | 25 G (83% used) |
| `/mnt/data` (sda) | 220 G | 75 G |

| Raw data | Size | Disk | What training reads instead | Reader |
|---|---|---|---|---|
| BCS-DBT DICOMs (`/mnt/data/mmfm_datasets/mammography/breast_cancer_screening_dbt`, symlinked as `datasets/mammography/bcsdbt_dicom`) | 79 G | data | `datasets/mammography/bcsdbt_png/` — 75 box-cropped slices, 3.4 M | `_dbt_metadata` / `_dbt_slice_to_png_cached` |
| ISPY2 NIfTI (`/mnt/data/mmfm_datasets/MRI/PKG - BreastDCEDL_ISPY2`) | 55 G | data | `datasets/MRI/ISPY2/cls/` — 37k uint8 `.npy`, 2.9 G | `scripts/preprocess_ispy2_mri.py` (offline) |
| CMMD DICOMs (`/mnt/data/mmfm_datasets/mammography/cmmd/dicom`, symlinked as `datasets/mammography/cmmd_dicom`; moved from `~/Downloads` 2026-10-06) | 22 G | data | `datasets/mammography/cmmd_png/` — 3,738 PNG, 2.7 G | `_cmmd_metadata` / `_dicom_to_png_cached` |
| Second ISPY2 copy (`~/Downloads/PKG - BreastDCEDL_ISPY2`) | 11 G | root | nothing (preprocess script points at the `/mnt/data` copy) | — |
| BreaDM `seg/` + `seg3D/` (`datasets/MRI/BreaDM/`) | 8.9 G | root | nothing — only `cls/img9Se` (76 M) is read | — |
| BreCaHAD.zip beside extracted copy | 1.0 G | root | n/a (histopathology, not wired in) | — |
| `all-mias.tar.gz` beside extracted copy | 0.1 G | root | `datasets/mammography/mias/all-mias` | — |
| CDD-CESM JPEGs (`~/Downloads/PKG - CDD-CESM`) | 1.5 G | root | read directly (already JPEG, small) | `_cdd_cesm_metadata` |
| Ultrasound sets (busbra, BUSI, BUSC, BrEaST, BUS) | 0.5 G | root | read directly | `_ultrasound_frame` |

Non-dataset space on `/`: `~/.cache/huggingface` 12 G (MedGemma, BiomedCLIP),
`~/.cache/BraveSoftware` 1.2 G, root-level `*.pth` checkpoints ~0.6 G (already
gitignored).

### How data is ingested today

Three patterns coexist in [training/train.py](../training/train.py):

1. **Raw on every epoch** — ultrasound sets, MIAS, CDD-CESM: PIL opens the
   original file; `BusbraTransform`/`MiasTransform` do mask/`crop_box` crop,
   contrast stretch, augmentation and resize per sample.
2. **Lazy DICOM → PNG cache** — CMMD, BCS-DBT: converted to 8-bit PNG on first
   use, reused afterwards. **But the frame builders still walk the raw DICOMs on
   every run** (`_cmmd_metadata` reads every header for `PatientID` +
   `ImageLaterality`; `_dbt_metadata` resolves DICOM paths before checking the
   cache). Deleting the DICOMs today breaks both datasets.
3. **Offline preprocessing to uint8 `.npy`** — ISPY2 (tumor-bbox crop, 9 DCE
   phases), BreaDM `cls/img9Se`.

## 1. Principles

- **Raw is disposable once it is reproducible**, not before. A raw dataset may
  be deleted only when it has (a) a tested registry entry that re-fetches it,
  (b) recorded checksums, and (c) completed analysis (§5 gate).
- **Prepared data is the canonical training input** and is backed up off-machine.
- **Every prepared dataset carries its own provenance** (`prep.json`): code
  commit, preprocessing parameters, source checksums — so it can be regenerated
  bit-for-bit or at least explained.
- **Only optimize what is big.** The on-the-fly datasets (ultrasound, MIAS,
  CDD-CESM) total ~2 G; re-plumbing them buys little and risks changing results.
  Leave them as-is.
- **Move before delete.** Where `/mnt/data` has room, first move root-disk raw
  data there (reversible), delete later.

## 2. Phase 1 — Make CMMD and BCS-DBT loadable without raw DICOMs

1. Add a "prepared manifest" step for each: on first build, write
   `datasets/prepared/<name>/manifest.csv` (`image, label, patient`, plus
   `laterality`/`view`/`slice`/`box` where relevant) next to the PNGs.
2. Frame builders read the manifest if present and only fall back to the DICOM
   walk when it's missing (or when `--rebuild-cache` is passed).
3. Write `prep.json` alongside (see §3).
4. **Verification:** build frames both ways and assert identical rows and
   identical train/val (and k-fold) splits for the default seed; then do one
   short `--device cpu` smoke epoch with the raw symlink temporarily renamed away.

Touches: `_cmmd_metadata`, `_cmmd_frame`, `_dbt_metadata`, `_dbt_frame`, and the
same paths in `_build_combined_sources`. No change to transforms.

## 3. Phase 2 — Prepared-data layout and formats

Target layout (all under the already-gitignored `datasets/`):

```
datasets/prepared/<name>/
    images/ or volumes.h5        # payload
    manifest.csv                 # one row per sample: path, label, patient, extra fields
    prep.json                    # provenance: git commit, script + args, params, source sha256 list, created date
```

Format rules:

| Data | Format | Why |
|---|---|---|
| 2D grayscale (mammo, US) | lossless **PNG**, uint8 (or uint16 where bit depth matters — see analysis plan §B) | `.npy` is uncompressed — typically 2–4× larger than PNG for these images |
| Multi-channel MRI (ISPY2, BreaDM 9-phase) | one **HDF5** (or zarr) per dataset, gzip/zstd chunked per sample; or `np.savez_compressed` per sample | far fewer files than 37k `.npy`; expected ≥2× smaller |
| Anything | never JPEG for derived data | lossy artifacts compound with existing compression |

Crop/resize rules for anything newly pre-cropped (e.g. future large datasets):
keep a **10–20% margin** around the lesion/breast ROI and store at a **512 px
short side**, not the training size (224). Augmentation (random-resized-crop,
small translations) stays possible and the training resolution can still change.

ISPY2 conversion: **measure first** (compress 500 samples, extrapolate). Do it
only if it saves >1 G; it requires a small loader change in `BreaDMDataset`.

## 4. Phase 3 — Dataset registry and one-command re-download

Replace "where did I get this from?" with a tracked registry:

- `data/sources/registry.yaml` (tracked; `datasets/` is gitignored so it can't
  live there). One entry per dataset:
  ```yaml
  cmmd:
    name: Chinese Mammography Database
    modality: mammography
    source: {type: tcia, manifest: data/sources/manifests/cmmd.tcia, doi: <TCIA collection DOI>}
    license: CC BY 4.0
    access: open | registration | dua
    raw_size_gb: 22
    raw_dir: /mnt/data/mmfm_datasets/mammography/cmmd   # where fetch puts it
    raw_checksums: data/sources/checksums/cmmd.sha256
    extra_files: [CMMD_clinicaldata_revision.xlsx]
    prepare: python3 scripts/datasets.py prepare cmmd
    prepared_dir: datasets/prepared/cmmd
    prepared_backup: hf://datasets/noe95/mmfm-cmmd   # private
    analysis: analysis/cmmd/                          # see dataset_analysis_plan.md
  ```
- `data/sources/manifests/*.tcia` — the NBIA manifest files for CMMD, BCS-DBT,
  CDD-CESM, ISPY2 (download from TCIA once, commit). The NBIA Data Retriever is
  already installed and accepts a manifest from the CLI, so fetch can be scripted
  or at least reduced to one documented command. BCS-DBT was pulled via Imaging
  Data Commons — record the `idc-index`/`s5cmd` query instead.
- `data/sources/checksums/<name>.sha256` — generated **before** deleting raw,
  so a re-download can be verified identical (matters for reproducing `prep.json`).
- `scripts/datasets.py` with subcommands:
  `list` · `status` (present? prepared? backed up? analyzed?) · `fetch <name>`
  (raw from source, or `--prepared` to pull the HF backup) · `prepare <name>` ·
  `verify <name>` (checksums) · `backup <name>`.
- Fold `scripts/download_datasets.py`'s existing open-HTTP sources into the
  same registry rather than keeping two lists; keep its "never route around
  access gates" rule — gated sources get a `manual_steps:` field instead.
- Point `train.py` default roots at `datasets/prepared/<name>` for the migrated
  datasets (keep old paths working via symlink during transition).

## 5. Phase 4 — Back up prepared data

- **Primary:** private Hugging Face dataset repos (account `noe95` is already
  authenticated), one per dataset, uploaded with `hf upload` including
  `manifest.csv` + `prep.json`. Restoring is `scripts/datasets.py fetch <name> --prepared`.
- **License check before upload** (record result in the registry):
  CMMD, CDD-CESM — CC BY 4.0 (private re-hosting fine); BCS-DBT — CC BY-NC
  4.0 (private, non-commercial fine); BreaDM — confirm license from the source
  page; anything under a DUA — do **not** upload, keep on `/mnt/data` instead.
  **BreastDCEDL-ISPY2 is CC BY-NC-ND 4.0** (per `breast_cancer_datasets.md`):
  "no derivatives" likely rules out sharing our preprocessed `img9Se` arrays,
  even in a private repo shared with others — keep the ISPY2 prepared data on
  `/mnt/data` only, or confirm with the BreastDCEDL authors first.
- **Secondary:** copy of `datasets/prepared/` on `/mnt/data` (it's small: ~6 G).

## 6. Phase 5 — Reclaim space (ordered, each item needs explicit sign-off)

Gate per dataset, all must be true: analysis done & reviewed · Phase 1 loader
works without raw (if applicable) · checksums recorded · registry entry with
tested fetch · prepared backup verified by re-download.

| # | Item | Frees | Disk | Action | Notes |
|---|---|---|---|---|---|
| 1 | `~/Downloads/PKG - BreastDCEDL_ISPY2` | 11 G | root | delete | first verify it's a subset of the `/mnt/data` copy (file list + checksums) |
| 2 | BreCaHAD.zip, all-mias.tar.gz | 1.1 G | root | delete archive, keep extracted | verify extracted copy is complete |
| 3 | CMMD DICOMs | 22 G | ~~root~~ data | **moved** to `/mnt/data` 2026-10-06 (checksum-verified, symlinks repointed); delete after gate | old copy in `~/Downloads/data for thesis/Mammography/` still to be removed by hand; deletion of the `/mnt/data` copy needs Phase 1 |
| 4 | BreaDM `seg/`, `seg3D/` | 8.9 G | root | delete after segmentation training + gate (or shrink: `seg3D` stores 0–255 values as float64 for most cases, so a lossless uint8 cast cuts ~7.8 G to ~1 G) | not read by current code. **Blocked until the BreaDM segmentation training (`dataset_reproduction_plan.md` §6 step 6) has been run**; also needs a registry entry and mask stats (analysis plan F/K) |
| 5 | HF cache | ≤12 G | root | `hf cache scan` → delete unused revisions | re-downloads automatically on next use |
| 6 | BCS-DBT DICOMs | 79 G → 6.6 G | data | **prune** to the 75 used volumes after gate (frees ~72 G); optionally delete those too later | only Benign/Cancer views with a lesion box are used (75 of 1,163; the rest are Normal/Actionable). Keeping the 75 full volumes (6.63 G, measured) keeps the option of multi-slice crops and re-cropping without a re-download. Needs Phase 1 for full deletion; pruning alone does not (loader only resolves those 75). Before pruning: write the list of kept series to the registry and record checksums for all 1,163 so a full re-fetch from IDC can be verified |
| 7 | ISPY2 raw NIfTI | 55 G | data | delete | `prepare` must be re-runnable from re-fetched raw |

Expected: ~34 G freed on `/` now (items 1–3; CMMD's 22 G already moved off `/`),
+8.9 G after the BreaDM segmentation training (item 4), up to +12 G more from
the HF cache; ~127 G on `/mnt/data` (items 6–7, net of the CMMD move adding 22 G there). Use `gio trash` / move-to-staging-folder for a week rather than
`rm`, then empty.

## 7. Housekeeping

- Move root-level `*.pth` into `checkpoints/` (gitignored), optionally push the
  ones cited in the report to a private HF model repo.
- Add LaTeX build outputs (`docs/latex/*.aux|bbl|blg|out`) to `.gitignore`.
- When downloading new datasets: download straight to `/mnt/data/mmfm_datasets/`,
  symlink into `datasets/`, add the registry entry **at download time**.

## 8. Decisions (2026-10-06)

1. BreaDM `seg`/`seg3D`: keep until the BreaDM segmentation training has been run, then delete after the analysis gate.
2. BCS-DBT: prune to the 75 Benign/Cancer volumes actually used (79 G → 6.6 G) after the analysis gate; Normal/Actionable volumes can be re-fetched from IDC if ever needed.
3. Private Hugging Face repos as the prepared-data backup: **approved**.
4. ISPY2 `.npy` → compressed HDF5: **approved**, still measure the saving first.
