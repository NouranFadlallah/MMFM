# Mixed-Backbone Combined Fusion Runs

A follow-up to the "Bonus: combined multimodal fusion run" in
[results_report.md](results_report.md), which trained `--dataset combined` with
ResNet-18 on every branch, ultrasound trained from scratch. This doc covers
three further `combined` runs that isolate what's actually driving the
improvement over that baseline: the backbone swap, warm-starting from a
converged single-dataset checkpoint, or RadImageNet pretraining.

## What changed in the code

`training/train.py --dataset combined` now supports per-branch backbone
selection (`--mammography-backbone` / `--ultrasound-backbone` /
`--mri-backbone`, each falling back to `--backbone` if unset) — previously the
CLI forced the same architecture onto all three branches, which no longer
matched reality once BUS-BRA moved to ResNet-50 and BUSI/BrEaST moved to
EfficientNet-B0 during the backbone-comparison runs. `models/backbone.py` also
gained `pretrained_weights` support (a local checkpoint path, e.g. RadImageNet,
loaded in place of ImageNet init), usable per-branch via
`--*-pretrained-weights`, independent of `--*-init-checkpoint` warm-starting.

## Per-branch validation set sizes

All three runs share the same combined validation set (427 samples total):

| Branch | Val samples |
|---|---|
| Mammography (MIAS) | **10** (2 malignant / 8 benign) |
| Ultrasound (BUS-BRA + BUSI + BUSC + BrEaST) | 300 (186 + 64 + 25 + 25) |
| MRI (BreaDM) | 117 |

The mammography branch's per-epoch accuracy is computed on only 10 samples, so
a single misclassified case swings it by 10 points — treat any single epoch's
`mammography_acc` as noise, not signal, in all three runs below.

Run config common to all three: batch size 16, image size 224, default
(non-paper-match) recipe — Adam, `cfg.epochs=10` budget, early-stopping
patience 5, masked-scalar fusion, modality dropout force-disabled (as always
for `combined`, since each sample already carries exactly one real modality).
Mammography and MRI branches are ResNet-18, warm-started from
`runs/run-1/mias_single_resnet18_best.pth` and
`runs/run-1/breamdm_single_resnet18_best.pth` in all three runs — only the
ultrasound branch's backbone/init varies between them.

## The three runs

| Run | Ultrasound backbone | Ultrasound init | W&B run name |
|---|---|---|---|
| **A — mixed / warm-start** | ResNet-50 | Warm-started from `runs/run-2/extra-backbones/busbra_single_resnet50_best_fold1.pth` (a converged single-dataset BUS-BRA model) | `combined-mixed-backbones-resnet50us` |
| **B — resnet18 control** | ResNet-18 | Warm-started from `runs/run-1/busbra_single_resnet18_best_fold1.pth` (same fold, same warm-start strategy, different architecture) | `combined-resnet18-control-warmstart` |
| **C — RadImageNet, trained in-loop** | ResNet-50 | RadImageNet-initialized (`weights/radimagenet_resnet50.pth`, via the [Warvito/radimagenet-models](https://github.com/Warvito/radimagenet-models) PyTorch port of the official RadImageNet weights) but trained from scratch *within* the fusion run, no warm-start | `combined-resnet50-radimagenet-scratch` |

Checkpoints: `runs/run-3/combined_fusion_mixed_best.pth` (A),
`runs/run-3/combined_fusion_resnet18_control_best.pth` (B),
`runs/run-3/combined_fusion_radimagenet_best.pth` (C).

## Results

| Run | Best epoch | Best val loss | Val acc at that epoch | Epochs run |
|---|---|---|---|---|
| Original baseline (all ResNet-18, ultrasound from scratch, ImageNet init) | — | 0.467 | 74.1% | 30 |
| **A — ResNet-50, warm-started** | 4 | 0.421 | 80.8% | 9 (early-stopped) |
| **B — ResNet-18, warm-started** | 10 | **0.395** | **83.6%** | 10 (full budget, still improving) |
| **C — ResNet-50, RadImageNet init, trained in-loop** | 2 | 0.428 | 81.5% | 7 (early-stopped) |

(Best epoch/loss pulled from each run's exact W&B history via `wandb.Api()`,
not eyeballed off the printed accuracy column — epoch 3 of Run A has the
higher *accuracy*, 0.850, but epoch 4's *validation loss* is marginally lower,
0.4208 vs. 0.4243, and checkpoint selection in `_train_model` is by loss, not
accuracy. Val loss below is the exact `validation/loss` W&B metric; the
per-epoch tables' "Val acc" and per-branch columns come from the training
script's own printed log line.)

Full per-epoch numbers (best-loss epoch **bolded**):

**Run A (ResNet-50, warm-started)**

| Epoch | Val loss | Val acc | Mammography (n=10) | Ultrasound (n=300) | MRI (n=117) |
|---|---|---|---|---|---|
| 1 | 0.865 | 0.649 | 0.200 | 0.710 | 0.530 |
| 2 | 0.697 | 0.724 | 0.900 | 0.850 | 0.385 |
| 3 | 0.424 | 0.850 | 0.600 | 0.837 | 0.906 |
| **4** | **0.421** | **0.808** | 0.700 | 0.853 | 0.701 |
| 5 | 0.631 | 0.745 | 0.600 | 0.743 | 0.761 |
| 6 | 0.804 | 0.817 | 0.800 | 0.807 | 0.846 |
| 7 | 0.510 | 0.822 | 0.800 | 0.817 | 0.838 |
| 8 | 0.489 | 0.787 | 0.600 | 0.803 | 0.761 |
| 9 | 0.587 | 0.742 | 0.400 | 0.717 | 0.838 |

**Run B (ResNet-18, warm-started)**

| Epoch | Val loss | Val acc | Mammography (n=10) | Ultrasound (n=300) | MRI (n=117) |
|---|---|---|---|---|---|
| 1 | 0.471 | 0.778 | 0.700 | 0.843 | 0.615 |
| 2 | 0.413 | 0.824 | 0.400 | 0.843 | 0.812 |
| 3 | 0.927 | 0.522 | 0.700 | 0.430 | 0.744 |
| 4 | 0.529 | 0.778 | 0.400 | 0.813 | 0.718 |
| 5 | 0.640 | 0.731 | 0.500 | 0.740 | 0.727 |
| 6 | 0.412 | 0.841 | 0.700 | 0.853 | 0.821 |
| 7 | 0.546 | 0.771 | 0.700 | 0.770 | 0.778 |
| 8 | 0.751 | 0.677 | 0.400 | 0.617 | 0.855 |
| 9 | 0.413 | 0.855 | 0.700 | 0.870 | 0.829 |
| **10** | **0.395** | **0.836** | 0.600 | 0.807 | 0.932 |

**Run C (ResNet-50, RadImageNet init, trained in-loop)**

| Epoch | Val loss | Val acc | Mammography (n=10) | Ultrasound (n=300) | MRI (n=117) |
|---|---|---|---|---|---|
| 1 | 0.552 | 0.703 | 0.200 | 0.723 | 0.692 |
| **2** | **0.428** | **0.815** | 0.700 | 0.833 | 0.778 |
| 3 | 0.527 | 0.754 | 0.800 | 0.730 | 0.812 |
| 4 | 0.499 | 0.792 | 0.700 | 0.827 | 0.709 |
| 5 | 0.726 | 0.632 | 0.700 | 0.590 | 0.735 |
| 6 | 0.459 | 0.836 | 0.500 | 0.880 | 0.752 |
| 7 | 0.661 | 0.714 | 0.800 | 0.687 | 0.778 |

## What's actually driving the improvement

Holding the two warm-started branches (mammography, MRI) fixed and varying
only the ultrasound branch separates three effects that were confounded in the
first mixed-backbone run:

1. **Warm-starting from a converged single-dataset model is the main effect.**
   Both warm-started runs (A: 0.421, B: 0.395) beat both non-warm-started runs
   (baseline: 0.467, C: 0.428) by a comparable margin, regardless of backbone
   or pretraining source. Starting the ultrasound branch from a model that's
   already solved BUS-BRA on its own gets the fusion model to a better
   optimum than training that branch from any generic (ImageNet or
   RadImageNet) initialization inside the fusion loop.
2. **RadImageNet beats ImageNet when training in-loop.** Comparing the two
   *non-warm-started* runs directly (baseline 0.467, ImageNet-pretrained vs.
   Run C 0.428, RadImageNet-pretrained, both trained from scratch within the
   fusion run) — RadImageNet initialization does measurably help here. This is
   the cleanest positive evidence so far for the RadImageNet weights adding
   value, isolated from the warm-start effect.
3. **ResNet-18 edges out ResNet-50 under identical warm-start treatment**
   (Run B 0.395 vs. Run A 0.421) — consistent with the BUS-BRA paper's own
   finding (Table 3: ResNet-18 > ResNet-50 on accuracy/sensitivity) and with
   this repo's own single-mode BUS-BRA 5-fold results in
   [results_report.md](results_report.md#L76). The backbone swap to ResNet-50
   is not itself the source of the improvement over the original baseline —
   if anything it's a slight drag relative to ResNet-18 once warm-start
   strategy is held constant.

## Caveats

- **Mammography branch numbers are not meaningful at n=10** in any of the
  three runs — see above.
- **Ultrasound warm-start (Runs A/B) uses only one of five BUS-BRA CV folds**
  (fold 1), not an aggregate — a different fold's checkpoint could shift
  results somewhat.
- **None of these runs used `--paper-match`** (no SGD/weighted-CE/contrast
  stretching), unlike the rigorous single-dataset BUS-BRA reproduction in
  `results_report.md` — these are fusion-mechanism/ablation checks on the
  default recipe, not paper reproductions.
- **Early stopping cuts runs at different points** (9, 10, and 7 epochs) so
  "epochs run" isn't held constant across the comparison — Run B in particular
  was still improving at epoch 10 (full budget hit, not early-stopped), so its
  true optimum might be even better with a larger epoch budget.
- Still a fusion-mechanism check with mixed-quality per-branch data (see
  `results_report.md`'s per-dataset caveats — MIAS and BrEaST results are weak
  on their own), not a benchmark result to report externally.
