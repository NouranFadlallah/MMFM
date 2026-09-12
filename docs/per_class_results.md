# Per-Class Results and False-Positive/False-Negative Bias

Companion to [results_report.md](results_report.md). Every number below is
computed directly from the already-trained checkpoints in `runs/` with
[scripts/per_class_analysis.py](../scripts/per_class_analysis.py) — no
retraining. For each cross-validated dataset, the script reconstructs the
same 5 official/stratified test folds used during training (same seed, same
split helper) and re-runs inference; BreastDM uses its single official
validation split, same as elsewhere in this repo.

**Definitions**, in this benign(0)/malignant(1) task:

- **False positive (FP)**: predicts malignant on an actually-benign case
  (over-diagnosis — an unnecessary biopsy/follow-up).
  **False positive rate** = FP / (FP + TN) = 1 − specificity.
- **False negative (FN)**: predicts benign on an actually-malignant case
  (under-diagnosis — a missed cancer). **False negative rate** =
  FN / (FN + TP) = 1 − sensitivity.
- **Bias verdict**: `false_positive_leaning` if the pooled FP rate exceeds
  the pooled FN rate, `false_negative_leaning` if the reverse, `balanced` if
  equal. Pooled means the confusion-matrix counts are **summed across all 5
  folds** before computing the rate, not averaged per-fold — this avoids a
  small-fold-size verdict flipping just from rounding.

## Results table

Confusion counts (`tn`/`fp`/`fn`/`tp`) are pooled across all 5 test folds
(all 6 datasets except BreastDM, which is a single 117-sample run).

| Dataset | Backbone | n | TN | FP | FN | TP | Sensitivity | Specificity | Malignant precision | Benign precision (NPV) | FP rate | FN rate | **Bias** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BUS-BRA | ResNet-18 | 1875 | 1055 | 213 | 109 | 498 | 0.820 | 0.832 | 0.700 | 0.906 | 0.168 | 0.180 | FN-leaning (weak) |
| BUS-BRA | ResNet-50 | 1875 | 1065 | 203 | 119 | 488 | 0.804 | 0.840 | 0.706 | 0.899 | 0.160 | 0.196 | FN-leaning (weak) |
| BUSI | ResNet-18 | 647 | 397 | 40 | 35 | 175 | 0.833 | 0.908 | 0.814 | 0.919 | 0.092 | 0.167 | **FN-leaning (consistent, 5/5 folds)** |
| BUSI | EfficientNet-B0 | 647 | 387 | 50 | 27 | 183 | 0.871 | 0.886 | 0.785 | 0.935 | 0.114 | 0.129 | FN-leaning (weak) |
| BUSC | ResNet-18 | 250 | 98 | 2 | 0 | 150 | 1.000 | 0.980 | 0.987 | 1.000 | 0.020 | 0.000 | FP-leaning, but tiny (2 errors total) |
| BrEaST-Lesions USG | ResNet-18 | 252 | 112 | 42 | 27 | 71 | 0.724 | 0.727 | 0.628 | 0.806 | 0.273 | 0.276 | Essentially balanced, both high |
| BrEaST-Lesions USG | EfficientNet-B0 | 252 | 101 | 53 | 24 | 74 | 0.755 | 0.656 | 0.583 | 0.808 | 0.344 | 0.245 | FP-leaning |
| mini-MIAS | ResNet-18 | 119 | 46 | 22 | 26 | 25 | 0.490 | 0.676 | 0.532 | 0.639 | 0.324 | 0.510 | **FN-leaning (consistent, 4/5 folds, severe)** |
| BreastDM | ResNet-18 | 117 | 13 | 11 | 3 | 90 | 0.968 | 0.542 | 0.891 | 0.812 | 0.458 | 0.032 | **FP-leaning (severe)** |

## What this actually says, per model

- **mini-MIAS is the most clinically concerning result in this report.** It
  misses just over half of actual malignant cases (FN rate 0.510,
  sensitivity 0.490), and this direction holds in 4 of 5 folds — not a
  coin-flip. Combined with the already-flagged small-sample caveat in
  `results_report.md`, this model should not be treated as usable for
  cancer screening in its current form.
- **BUSI leans false-negative consistently** (5/5 folds for ResNet-18):
  despite a respectable 0.884 aggregate accuracy, it misses malignant cases
  (FN rate 0.167) about twice as often as it false-alarms on benign ones
  (FP rate 0.092). EfficientNet-B0 on the same dataset is closer to
  balanced (FN 0.129 vs. FP 0.114) but still leans the same direction.
- **BreastDM leans false-positive severely**: it over-calls malignant on
  46% of actually-benign cases (FP rate 0.458), while almost never missing
  a real malignant case (FN rate 0.032). This is the "over-calls malignant"
  behavior already flagged qualitatively in `results_report.md`'s BreastDM
  row — now quantified. In a screening context this is the safer failure
  direction than mini-MIAS's, but it would still generate a lot of
  unnecessary follow-up on a real deployment.
- **BUSC is not meaningfully biased either way** — 2 false positives and 0
  false negatives out of 250 pooled test predictions is too few errors to
  call a direction, consistent with the "near-perfect, investigate before
  trusting" flag already in `results_report.md` and `docs/latex/main.tex`'s
  preprocessing discussion. A false-positive lean this small is not the
  signal that discussion was looking for; the Grad-CAM check proposed there
  is still the right next step (see below).
- **BUS-BRA and BrEaST-Lesions USG do not have a stable bias direction.**
  Their pooled verdicts (FN-leaning for BUS-BRA, mixed for BrEaST) hide
  real fold-to-fold flips: BUS-BRA ResNet-18 is FP-leaning in 3 of 5 folds
  and FN-leaning in 2; BrEaST ResNet-18 flips 3 times across its 5 folds.
  Treat the pooled "lean" for these two as a weak aggregate tendency, not a
  property of the model.

## Grad-CAM follow-up

`docs/latex/main.tex`'s preprocessing discussion (Section 4) flagged BUSC
specifically for a saliency check, since it has the least preprocessing
(no ROI crop at all) and the best result. A Grad-CAM utility now exists at
[utils/gradcam.py](../utils/gradcam.py); [scripts/gradcam_visualize.py](../scripts/gradcam_visualize.py)
reconstructs a test fold the same way `per_class_analysis.py` does, finds
which examples the checkpoint gets wrong, and saves a heatmap overlay for
every misclassified example plus a handful of correct ones per class.

Run for BUSC fold 1 (`runs/run-2/busc_single_resnet18_best_fold1.pth`,
9 images in `docs/gradcam/busc_resnet18_fold1/`): the correctly classified
malignant example shows a compact, centrally-localized hotspot consistent
with attention on an actual mass. But the correctly classified benign
example's heatmap covers a broad diffuse region rather than a focal point,
and the one misclassified example (a benign case predicted malignant at
p=0.57) also has a large, non-lesion-specific hot region reaching the image
border. With only 250 total images and no ROI crop, this is consistent
with, though does not prove, the concern raised in the preprocessing
discussion: the model may be partly using broad image texture or
device-specific framing rather than a localized lesion signal. This
warrants the same follow-up already proposed there — recover lesion
boundaries by another means and re-run with an actual crop — before trusting
BUSC's 0.992 accuracy as a working classifier.

Regenerate for any other checkpoint/fold with:
```bash
.venv/bin/python3 scripts/gradcam_visualize.py --dataset busi --backbone efficientnet_b0 \
    --checkpoint runs/run-2/extra-backbones/busi_single_efficientnet_b0_best_fold1.pth --fold 1
```

## Reproducing this

```bash
.venv/bin/python3 scripts/per_class_analysis.py
```

Writes `docs/per_class_results_raw.json` (full per-fold detail: confusion
counts, per-class precision/recall/F1/support, and the bias verdict for
every individual fold, not just the pooled numbers in the table above).
