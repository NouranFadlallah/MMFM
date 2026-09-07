# MMFM — Multi-Modal Fusion Model for Breast Cancer Imaging

Three-branch late-fusion classifier over mammography, ultrasound, and MRI, plus
single-modality reproduction baselines for five breast-imaging datasets. Binary
classification (benign vs. malignant) throughout.

## Layout (flat, not `fusion/`-prefixed)

Despite what the README's original Quickstart section says, there is no `fusion/`
subdirectory — everything lives at repo root:

- `config.py` — `cfg`: default hyperparameters (backbone, embedding dim, lr, epochs, fusion mode).
- `models/backbone.py` — `create_backbone(name, pretrained, remove_head, input_channels)`: torchvision → timm → tiny fallback CNN, in that order. Only resnet* backbones get their first conv adapted to non-3-channel input (`_adapt_first_conv`); vgg/densenet/other backbones silently skip channel adaptation.
- `models/fusion_model.py` — `FusionLateModel`: 3 independent backbones → per-branch projection (Linear+ReLU) → per-branch classifier head → learnable per-branch scalar logits, masked by a `presence_mask` and softmaxed into fusion weights, then a weighted sum of per-branch logits. Only `fusion_mode='masked_scalar'` is implemented; `'gating'` raises `NotImplementedError`.
- `data/dataset.py` — transforms (`BusbraTransform`, `MiasTransform`, `BreadmTransform`) and datasets (`TripleImageDataset`, `BreaDMDataset`, `SingleModalityBranchDataset`).
- `data/preprocess.py` — `build_manifest`: builds a `TripleImageDataset`-compatible CSV (img1/img2/img3/label) from a `benign/`/`malignant/` folder tree, replicating one image into all three columns.
- `training/train.py` — CLI entry point (`python3 training/train.py --dataset ... [--single-mode]`), per-dataset frame builders, the combined multi-dataset sampler, and the train/validate loop. This is the file to read to understand actual current behavior — it has grown organically and is the source of truth over the docs below.
- `utils/metrics.py` — just `accuracy()`.
- `docs/dataset_reproduction_plan.md` — per-dataset paper-reproduction recipe and "local implementation status" (what's actually wired up vs. still a gap). Read this before touching any single-dataset path.
- `docs/public_dataset_intake_plan.md` — datasets *not* yet downloaded locally; intake/licensing notes only.
- `docs/breast_imaging_vlms.md` — literature survey of breast-specialized VLMs/MLLMs, for future comparison baselines; not code-relevant.
- `notebooks/01_quick_start.ipynb` — minimal end-to-end demo using the fallback `simple` backbone (no torchvision/timm dependency), including a presence-mask example with intentionally-missing modalities.
- `tests/test_preprocess.py` — the only test file. Covers `BusbraTransform` and `build_manifest`. Nothing else in the repo has test coverage.

## Datasets and how to train each

`training/train.py --dataset {busbra,busi,busc,breast,mias,breamdm,combined} [--single-mode]`.
Single-mode trains a plain `SingleBackboneClassifier` (one backbone, no fusion,
no modality dropout — this is the paper-reproduction path). Without
`--single-mode` on a single dataset, it trains a full 3-branch `FusionLateModel`
with the *same* image triplicated across all three branches and modality
dropout applied independently per branch — this is a fusion-mechanism sanity
check, not a real multimodal experiment. `--dataset combined` is the real
multimodal path: it builds one `SingleModalityBranchDataset` per underlying
dataset (MIAS → mammography branch, busbra/busi/busc/breast → ultrasound
branch per `--ultrasound-datasets`, BreaDM → MRI branch), concatenates them,
and samples with weights that balance across branch → source-within-branch →
class-within-source. Modality dropout is force-disabled for `combined` because
each sample already carries exactly one real modality.

Dataset roots are hardcoded relative paths under `datasets/` (gitignored, not
in this checkout's tracked files) — see `_ultrasound_frame`, `_mias_frame`,
`_busi_frame`, `_busc_frame`, `_breast_frame` in [training/train.py](training/train.py)
for the exact expected subpaths. `docs/dataset_reproduction_plan.md` documents
what's actually present locally (e.g. the BUSI checkout is only 163/780 images).

Training defaults to `--device cuda` and **raises** if no GPU is available
(must pass `--device cpu` explicitly for local smoke tests). Every run logs to
W&B (`wandb.init` is unconditional in `_train_model` — there's no offline/no-op
path), and saves the best-val-loss checkpoint to `<dataset>_single_<backbone>_best.pth`
(or `combined_fusion_<backbone>_best.pth`) in the repo root.

Checkpoints from single-mode runs can warm-start the corresponding branch of a
combined `FusionLateModel` run via `--mammography-init-checkpoint` /
`--ultrasound-init-checkpoint` / `--mri-init-checkpoint` (`_warm_start_branch`,
best-effort strict `load_state_dict` per submodule, prints and skips on
mismatch rather than raising).

## Known issues / things to watch

See the fix plan delivered alongside this file for the current punch list
(stale README paths, an `_safe_list_images` filter bug in `data/preprocess.py`,
untracked 44MB `*.pth` checkpoints not covered by `.gitignore`, and a couple of
latent edge cases in `FusionLateModel.forward` / CSV `crop_box` round-tripping
that aren't hit by any current code path but would bite a future caller).
