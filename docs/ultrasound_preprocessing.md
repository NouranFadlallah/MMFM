# Ultrasound Preprocessing (optional)

Ported from the thesis notebook `Dataprocessing.ipynb` (Colab) into MMFM on
2026-10-16. Nothing here runs unless you ask for it with `--us-preprocess`.

## Turning it on

```bash
python3 training/train.py --dataset breast --single-mode \
    --us-preprocess "perc_hm:reference=datasets/ultrasound/busbra_train_hist.npy" cross_removal
```

- Steps run in the order given, before the dataset's usual transform. Masks
  and crop boxes still line up, because no step moves pixels.
- The flag applies to `busbra`, `busi`, `busc`, `breast`, `bus_cot`, their
  `--kfold` runs, and the ultrasound branch of `--dataset combined`.
- Processed images are cached under `datasets/ultrasound/_preprocessed/<hash>/`.
  The hash covers the step list and the contents of any referenced file, so a
  new reference or a retrained GAN gets a fresh cache. Override it with
  `--us-preprocess-cache`.

## Steps (`data/us_preprocessing.py`)

| Step | What it does | Notebook cell |
|---|---|---|
| `hist_eq` | Global histogram equalization | Histogram Equalization |
| `hist_match:reference=R` | CDF matching to a reference image or 256-bin `.npy` histogram (pixel-identical to skimage) | Histogram matching |
| `hist_spec` | Fuzzy c-means (3 clusters) + generalized-gamma histogram specification | Histogram Specification |
| `intensity_window:low=0.5;high=99.5` | Percentile window rescaled to 0–255 | Intensity Normalization |
| `perc_hm:reference=R` | `intensity_window`, then `hist_match` | Perc-HM |
| `line_removal:line_length=4;min_area=5` | Bright and dark thin lines: top-hat/black-hat + Otsu + inpaint | Line and cross removal |
| `cross_removal` | Bright caliper crosses only, area-filtered (8–120 px) | Line and cross removal |
| `cyclegan:checkpoint=C;direction=A2B` | A trained generator (see below) | Deep learning |

Differences from the notebook, and why:

1. **Intensity normalization.** The notebook's MONAI `ScaleIntensityRanged`
   mapped 0–255 to 0–255 with clipping, which does nothing to uint8 images, so
   "normalized" and Perc-HM's first stage were no-ops. MMFM uses a percentile
   window instead.
2. **Histogram specification.** skfuzzy returns clusters in arbitrary order,
   so the hypo / normal / hyper prototypes were paired with random clusters.
   MMFM sorts clusters by centre intensity. Fuzzy c-means runs on the 256-bin
   histogram (same objective, about 1,000× faster), so scikit-fuzzy isn't needed.
3. **Histogram matching reference.** The notebook matched to one randomly
   chosen image. Use a pooled training-split histogram instead:
   `python3 scripts/preprocess_ultrasound.py reference-hist --dataset busbra --out datasets/ultrasound/busbra_train_hist.npy`.
4. **Line removal is destructive.** It inpaints 15–91% of the pixels on BUS-BRA
   and BrEaST samples at both notebook settings (speckle passes Otsu as
   "lines"). `cross_removal` changes 2–22%. Preview before using either:
   `python3 scripts/preprocess_ultrasound.py preview --dataset breast --steps line_removal`.

## CycleGAN variants (`models/cyclegan.py`, `scripts/train_cyclegan.py`)

| Variant | Source | Defaults |
|---|---|---|
| `cyclegan` | junyanz/pytorch-CycleGAN-and-pix2pix | ResNet-9, PatchGAN, LSGAN, cycle 10, identity 0.5, lr 2e-4, 100 + 100 decay epochs, pool 50 |
| `ncia` | gkdl3000/USenhance-submission | Aux-conditioned 8-block generator, 50 k-means pseudo-classes refreshed every 10 epochs, VGG19 perceptual 2, aux 0.1, identity 5, lr 1e-4, grad clip 5, 300 epochs |
| `poas` | YoonChiHo/USenhance23_POAS `--is_mtl` | `cyclegan` + bottleneck classifier on each image's benign/malignant label. The original never optimized or saved the classifier and called an undefined `netG_A_pretrained`. |

- Dataset names as domains contribute only their **training split**, so a
  translator never sees the classifier's validation or test images.
- The vanilla generator uses junyanz's exact module layout, so the notebook's
  Colab checkpoints (`latest_net_G_A.pth`, 3-channel) load directly as
  `cyclegan:checkpoint=...`.
- `ncia` downloads ImageNet VGG19 weights (about 550 MB) on its first run.

```bash
python3 scripts/train_cyclegan.py --variant cyclegan --domain-a breast --domain-b busbra
python3 scripts/train_cyclegan.py --variant poas --domain-a breast --domain-b busbra
```

**Marker removal.** `python3 scripts/preprocess_ultrasound.py markers` draws
synthetic calipers, boxes, arrows and scanner text on BrEaST images using
their lesion masks: 252 marked/clean pairs. Train on them with
`--paired --domain-a datasets/ultrasound/markers/marked --domain-b datasets/ultrasound/markers/clean`,
which adds an L1 loss between `G_A2B(marked)` and the clean image.
