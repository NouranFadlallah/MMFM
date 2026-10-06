# Breast-Imaging Vision-Language Models

## Scope

This is an intake inventory of vision-language models (VLMs), multimodal large
language models (MLLMs), and contrastive image-text models that were trained or
substantially adapted for breast cancer or breast imaging. It distinguishes
breast-specialized training from generic medical VLMs that were only evaluated
on breast images. Reported metrics remain study-specific until the dataset,
split, target, and preprocessing have been matched.

Research was reviewed through August 2026. Links point to the primary paper and
official code/model locations when they are available.

## At A Glance

| Model | Modality | Breast-specific training | Public artifacts | Most relevant local data |
| --- | --- | --- | --- | --- |
| Mammo-CLIP (Batman Lab) | Mammography | Image-report contrastive pretraining | Code and checkpoints | MIAS, future VinDr-Mammo |
| Mammo-FM | Mammography | Large-scale multi-view image-report contrastive pretraining | Code and weights | MIAS, future VinDr-Mammo |
| Mammo-CLIP (multi-view adapter) | Mammography | Four-view CLIP adaptation | No verified code/weights | Design reference only |
| BreastGPT | Mammography, ultrasound, MRI, pathology | BreastStage multimodal instruction training | Code, checkpoint, data/benchmark pages | Future VinDr-Mammo, I-SPY MRI, BUS datasets |
| CorePath | Pathology WSI | Core-biopsy WSI-report adaptation | Repository; weights pending | Not currently applicable |
| XBusNet | Ultrasound | Prompt-conditioned BUS segmentation | No verified code/weights | BUSI, BUS-BRA, BrEaST-Lesions |

## Mammography Models

### Mammo-CLIP: Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography

- **Paper:** Nguyen et al., [Mammo-CLIP: A Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography](https://arxiv.org/abs/2405.12255), MICCAI 2024.
- **Artifacts:** [source code](https://github.com/batmanlab/Mammo-CLIP), [project page](https://shantanu-ai.github.io/projects/MICCAI-2024-Mammo-CLIP/), [pretrained checkpoints](https://huggingface.co/shawn24/Mammo-CLIP/tree/main/Pre-trained-checkpoints/), and [downstream checkpoints](https://huggingface.co/shawn24/Mammo-CLIP/tree/main/Downstream-checkpoints). The code licence is CC BY-NC-SA 4.0.
- **Task and architecture:** a 2D mammography image-text contrastive model for zero-shot, linear-probe, fine-tuned finding/cancer classification, and weakly supervised lesion localization. It pairs an ImageNet-initialized EfficientNet-B2 or B5 visual encoder with BioClinicalBERT. Projected and L2-normalized image/text features are optimized with a symmetric CLIP-style contrastive objective. Multi-View Supervision aligns original/augmented images and report sections; the Mammo-FActOR component produces sentence-aligned heatmaps.
- **Breast-specific pretraining:** 13,829 UPMC patient-report pairs containing 25,355 mammograms with BI-RADS 0-2. It also uses VinDr attributes converted to radiologist-templated report-like text. The UPMC source data are private.
- **Preprocessing and inputs:** CC/MLO images receive breast-ROI extraction, intensity values below 40 are set to zero, constant background is removed, orientation is standardized, and images are resized to 1520 x 912. Text inputs are the report `FINDINGS` and `IMPRESSION` sections. The training study uses image augmentations and Italian-English back translation for text augmentation.
- **Reported results:** with EfficientNet-B5, RSNA cancer AUC is 0.60 zero-shot and 0.91 after full fine-tuning; VinDr calcification AUC is 0.96 for an all-data linear probe and 0.98 after fine-tuning. Mammo-FActOR localization IoU at 0.50 is 0.37 for masses and 0.17 for calcifications. These values are tied to the paper's task definitions and splits.
- **Reproduction assessment:** the most practical transparent mammography image-text baseline. Reproduce its DICOM/PNG orientation, ROI, threshold, and high-resolution input path before comparing features. The private report corpus prevents full pretraining reproduction, but the public checkpoint supports frozen-encoder, linear-probe, and fine-tuning experiments.
- **MMFM fit:** good fit for future VinDr-Mammo work. mini-MIAS has single lower-resolution PGM images rather than full four-view digital mammography, so treat use as domain-transfer evaluation rather than a direct reproduction.

### Mammo-FM: Breast-specific Foundational Model for Integrated Mammographic Diagnosis, Prognosis, and Reporting

- **Paper:** [Mammo-FM: Breast-specific Foundational Model for Integrated Mammographic Diagnosis, Prognosis, and Reporting](https://arxiv.org/abs/2512.00198), first posted 2025.
- **Artifacts:** [source code](https://github.com/batmanlab/Mammo-FM) and [model weights](https://huggingface.co/batmanLab/Mammo-FM). Read the model card before use: weights are research-only/non-commercial and prohibit clinical use, redistribution, serving, and distillation. The repository currently supports diagnostic validation, linear probing, fine-tuning, and selected detection workflows; some zero-shot, prognostic, and report-generation code remains forthcoming.
- **Task and architecture:** multi-view mammography representation learning for diagnosis, detection, risk prediction, and reporting. The foundation model combines an ImageNet-initialized EfficientNet-B5 encoder with a ModernBERT text encoder, additionally adapted on 219,451 UPMC chest-CT reports. It uses multi-view contrastive InfoNCE between original/augmented image and report views. The associated Mammo-GRG report generator passes four shared encoder outputs through attention pooling, a two-layer projector, and Llama-3.1-8B.
- **Breast-specific pretraining:** 821,326 mammograms from 140,677 patients across Mayo Clinic, UPMC, Boston University/BMC, and EMBED. Private report-paired data are from Mayo, UPMC, and BU; EMBED structured attributes are rendered as report-like text.
- **Preprocessing and inputs:** raw DICOM is converted to 8-bit PNG; MONOCHROME1 is corrected; background is thresholded/removed; orientation is normalized; and each view is resized to 1520 x 912. Mammo-GRG uses LCC, LMLO, RCC, and RMLO in that order.
- **Reported results:** zero-shot cancer AUC is 0.75 on RSNA and 0.83 on VinDr; full VinDr cancer fine-tuning reaches AUC 0.93. VinDr mass-detection mAP at 0.5 is 0.58. Mammo-GRG reports UPMC GREEN factuality 0.42 +/- 0.35 and VinDr finding recall of 0.59 for calcification and 0.43 for mass. These measures span distinct evaluation tasks and are not directly comparable.
- **Reproduction assessment:** strongest released mammography representation-transfer candidate. Full pretraining cannot be reproduced from its private image-report datasets, but checkpoint evaluation/fine-tuning is feasible after matching the high-resolution four-view preprocessing and model-card licence.
- **MMFM fit:** start with frozen features and linear probing on future VinDr-Mammo, then fine-tune only after establishing a conventional ResNet baseline. It should not be treated as MRI or ultrasound pretraining.

### Mammo-CLIP: CLIP for Enhanced Breast Cancer Diagnosis with Multi-view Mammography

- **Paper:** [Mammo-CLIP: Leveraging Contrastive Language-Image Pre-training (CLIP) for Enhanced Breast Cancer Diagnosis with Multi-view Mammography](https://arxiv.org/abs/2404.15946), later published in *Medical Physics* (2026), DOI [10.1002/mp.70261](https://doi.org/10.1002/mp.70261).
- **Identity warning:** this is a separate model from Batman Lab's Mammo-CLIP above. Keep the paper DOI/repository reference with any experiment name to avoid conflating the two.
- **Task and architecture:** four-view bilateral mammography malignancy diagnosis. It adapts a CLIP backbone with image/text plug-in adapters while updating about 1% of parameters, and applies early fusion to bilateral CC/MLO views.
- **Breast-specific adaptation:** internal cohort of 470 malignant and 479 benign cases; external cohort of 60 malignant and 294 benign cases.
- **Preprocessing and availability:** the reviewed abstract does not disclose enough intensity, crop, or resolution detail for a faithful pipeline. No official code or weights were verified.
- **Reported results:** internal AUC 0.841 +/- 0.017 and external AUC 0.837 +/- 0.034. The paper reports corresponding cross-view-transformer AUCs of 0.817 +/- 0.012 and 0.807 +/- 0.036.
- **Reproduction assessment:** use it as an architectural reference for parameter-efficient, early-fusion CLIP adaptation. It is not currently a drop-in reproducible baseline.

## Cross-Modality Breast MLLM

### BreastGPT

- **Paper:** [BreastGPT: A Multimodal Large Language Model for the Full Spectrum of Breast Cancer Clinical Routine](https://arxiv.org/abs/2606.04911), 2026.
- **Artifacts:** [source code](https://github.com/YangYY-Liu/BreastGPT), [project page](https://yangyy-liu.github.io/BreastGPT.io/), [checkpoint](https://www.modelscope.cn/models/YYangYang/BreastGPT-8B), [BreastStage training data](https://www.modelscope.cn/datasets/YYangYang/BreastStage), and [BreastStage-Bench](https://www.modelscope.cn/datasets/YYangYang/BreastStage-Bench). The model/data are CC BY-NC 4.0.
- **Task and architecture:** an eight-billion-parameter multimodal model for mammography, breast ultrasound, multiparametric MRI, pathology WSI, and chest CT. It supports VQA, grounded captioning, report generation, lesion characterization, and workflow-stage reasoning. The base LLM is Qwen3-VL-8B-Instruct; radiology uses its native ViT, while pathology uses frozen CONCH v1.5 tile features plus a trainable LongNet branch. A concept selector compresses visual representations to 128 tokens.
- **Breast-specific instruction training:** BreastStage contains 1.86 million instruction pairs, about 662,000 images/volumes, 17 source datasets, five modalities, and 136 templates. Disclosed portions include 10,405 ultrasound images, 592,470 mammograms from MIAS/VinDr/RSNA/EMBED among others, 36,124 private multiparametric-MRI volumes, and 2,510 breast WSIs.
- **Training and preprocessing:** training first aligns the radiology ViT/adapter with the LLM frozen, then fine-tunes end to end. MRI/CT volumes are capped at 384 x 384 x 48; WSIs are tiled at 512 x 512 at 20x magnification. Full training used 32 H100 GPUs.
- **Reported results:** on BreastStage-Bench (12,182 held-out records), the cluster variant reports closed-ended VQA accuracy 75.66%, open-ended score 89.92%, ultrasound grounded-caption IoU 79.59, MRI report weighted score 67.67, and MRI 3D-grounding IoU 33.49. These are the authors' multimodal benchmark metrics, not standard classification AUROCs.
- **Reproduction assessment:** the only identified breast-specialized VLM spanning the repository's mammography, MRI, and ultrasound modalities. Inference/fine-tuning may be feasible; training from scratch is not. Private source MRI data are not redistributed, and much of the corpus is not patient-linked across modalities.
- **MMFM fit:** evaluate it as a task-specific zero/few-shot or instruction-tuning reference only after matching its modality preparation. It does not directly consume the repository's nine-channel BreaDM arrays or mask-cropped 224 x 224 ultrasound classifier inputs.

## Pathology VLM

### CorePath

- **Paper:** [CorePath: A Breast-Specialized Pathology Foundation Model for Core Needle Biopsy Diagnosis and Risk-Controlled Report Generation](https://arxiv.org/abs/2608.03079), 2026.
- **Artifacts:** [repository](https://github.com/danninglee/CorePath). At the time of review it contains project information and licence material; the paper states that code and trained weights will be released upon acceptance.
- **Task and architecture:** H&E core-needle-biopsy WSI diagnosis, invasion assessment, subtype classification, and controlled report generation. It adapts PRISM with Perceiver-only adapters; Virchow patch features feed the PRISM slide encoder.
- **Breast-specific training:** 7,901 paired core-biopsy WSI-report cases from two Chinese centres; the study includes 10,946 slides from six centres. The paired source data are private.
- **Training and input:** contrastive and distillation losses are used while the generative loss is disabled during adaptation. Slides use up to 2,560 tissue tiles and reports up to 768 tokens; reports are structured in Chinese then translated to English for alignment.
- **Reported results:** BCNB invasive-carcinoma three-class subtyping AUC 0.7780; BRACS lesion stratification AUC 0.8178 and fine-grained classification AUC 0.8252. Private-centre five-class subtype AUCs range from 0.9526 to 0.9735. These are pathology results, not radiology metrics.
- **MMFM fit:** not applicable to the current repository, which has no WSI data. Retain it as a future pathology-extension reference rather than adding it to the current experimental matrix.

## Ultrasound VLM-Adjacent Model

### XBusNet

- **Paper:** [XBusNet: Text-Guided Breast Ultrasound Segmentation via Multimodal Vision-Language Learning](https://doi.org/10.3390/diagnostics15222849), *Diagnostics*, 2025.
- **Task and architecture:** BI-RADS-oriented prompt-conditioned breast-ultrasound lesion segmentation. It combines a CLIP ViT global branch with a local U-Net-style branch.
- **Classification:** this is a task-specific vision-language segmentation hybrid, not a released general-purpose breast-ultrasound foundation VLM.
- **Availability and reproduction assessment:** the primary metadata reviewed does not establish public code, weights, a reusable release, or enough source detail to recommend a faithful baseline. Revisit it when a stable artifact and precise paper protocol are available.
- **MMFM fit:** potentially relevant only for segmentation experiments on BUSI, BUS-BRA, BrEaST-Lesions, UDIAT, or BUID. It is not a substitute for a classification representation baseline.

## Generic Models: Useful Comparators, Not Breast-Specialized Pretraining

Generic medical/natural-image VLMs can be evaluated on breast images, but should
not be described as trained for breast cancer without breast-domain adaptation.
Examples include generic CLIP variants, LLaVA-Med, MedGemma/MedSigLIP,
HuatuoGPT-Vision, Lingshu, and the pathology models PRISM, CONCH, TITAN, and
UNI. Likewise, breast-ultrasound studies that only test a generic VLM measure
transfer performance rather than provide breast-specific pretraining.

## MMFM Fine-Tuning Attempt (2026-09-28)

The user asked to actually try training MedGemma or another relevant VLM on this
repo's local data, and to write a full plan for whatever couldn't be attempted
directly. Both happened; this section records what was checked, what ran, and
what's next.

### Access check: MedGemma and MedSigLIP were gated, now unblocked

Checked via the Hugging Face Hub API (account `noe95`, 2026-09-28):

| Model | Params | License | Gated | This account's access (2026-09-28) |
| --- | --- | --- | --- | --- |
| `google/medgemma-4b-it` | 4.3B | other (Health AI Developer Foundations) | Yes | File listing works; `config.json`/weights download is `HF_FS_ACCESS_DENIED` — access not yet granted |
| `google/medsiglip-448` | 878M | other | Yes | Same — access not yet granted |
| `google/paligemma2-3b-pt-224` | 3.0B | gemma | Yes | Not checked further, same license family as MedGemma |

**Update (2026-10-06):** the user accepted the Health AI Developer Foundations
terms; re-checked via the Hub API and both `google/medgemma-4b-it` and
`google/medsiglip-448` now download `config.json` successfully (no more
`HF_FS_ACCESS_DENIED`). Installed `peft` and `bitsandbytes` (added to
`requirements.txt`) and proceeded directly to the MedGemma LoRA plan below.

### What actually ran tonight: BiomedCLIP fine-tune

`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (MIT license, not
gated, ViT-B/16 image tower + PubMedBERT text tower, 196M params total / 86M in
the image tower) was fine-tuned as a flat benign/malignant classifier —
`scripts/finetune_biomedclip.py`. Unlike `FusionLateModel`, this is not a
three-branch model: BiomedCLIP's image tower takes one ordinary 3-channel
224x224 image, so every locally reproduced dataset is flattened into a single
manifest (`build_manifest`), reusing each dataset's own frame builder and
already-patient-safe split from `training/train.py`:

- Mammography: MIAS (its own lesion `crop_box` applied), CDD-CESM, CMMD, BCS-DBT
- Ultrasound: BUS-BRA, BUSI, BUSC, BrEaST
- MRI: BreaDM + BreastDCEDL-ISPY2 `img9Se` patches, collapsed from 9 DCE-phase
  channels to 1 representative channel (index 4) replicated to RGB, since
  BiomedCLIP has no multi-phase-DCE convention

Combined manifest: 41,476 train / 4,565 val rows across the 10 sources above.
ISPY2 alone is 33,475 of those (80.7%, all malignant) — a flat classifier
trained on raw example counts would mostly learn "MRI-patch texture ->
malignant" rather than a real benign/malignant signal. `sample_weights()`
applies the same modality -> source-within-modality -> class-within-source
balancing `training/train.py`'s `_build_combined_datasets` already uses for the
fusion model, verified empirically to spread draws roughly evenly across all 10
sources (~150-380 draws each per 2,000 samples) rather than being ISPY2-dominated.

Training: AdamW, encoder lr 1e-5 / fresh linear-head lr 1e-3, batch size 32,
up to 20 epochs with patience 5, logged to W&B project `mmfm-vlm`. Launched
2026-09-28 22:34, running concurrently with the `combined_resnet18_ispy2_bcsdbt`
ResNet run (GPU headroom was ample — 2.2/12GB used).

**Result:** early-stopped after 6 epochs; the saved checkpoint
(`biomedclip_combined_best.pth`) is epoch 1 (lowest val loss, 0.1441 — val loss
got monotonically worse every epoch after that while train accuracy climbed to
0.994, i.e. epochs 2-6 were pure overfitting)
([run](https://wandb.ai/nouran-fadlallah-none/mmfm-vlm/runs/0u9gr8tr)). Overall
val accuracy on that checkpoint: **0.9433** (4,565 samples) — but per-source
breakdown shows this number is exactly as misleading as the ResNet
`mri_accuracy` caveat above warned it would be:

| Source | n | Benign | Malignant | Accuracy |
| --- | --- | --- | --- | --- |
| ispy2 | 3,705 | 0 | 3,705 | 0.9987 |
| busc | 25 | 10 | 15 | 1.0000 |
| breamdm | 117 | 24 | 93 | 0.8718 |
| busi | 64 | 43 | 21 | 0.8594 |
| breast | 25 | 20 | 5 | 0.8400 |
| busbra | 186 | 132 | 54 | 0.8172 |
| cmmd | 368 | 96 | 272 | 0.5707 |
| cdd_cesm | 57 | 27 | 30 | 0.5614 |
| bcsdbt | 8 | 2 | 6 | 0.5000 |
| mias | 10 | 8 | 2 | 0.5000 |

ISPY2 alone is 81% of the validation set and sits at 99.87% (it's
malignant-only, so this is close to the trivial ceiling, same as the ResNet
run). Weighted accuracy over the other 9 sources only: **0.705** — a much more
honest number. Ultrasound sources (busbra/busi/busc/breast, 81-100%) and
BreaDM MRI (87%) show real signal; every DICOM-derived mammography source
(cmmd, cdd_cesm, bcsdbt, mias — the four *hardest-won* datasets in this repo,
each needing an external label file) sits at or barely above chance. Read this
as: **BiomedCLIP's frozen-ish, briefly fine-tuned image tower has not learned
mammography discrimination here** — plausibly because mammography is the
smallest, most source-fragmented modality in the manifest (571 total samples
across 4 sources vs. ultrasound's 2,127 and MRI's 38,382), and because a single
epoch of encoder fine-tuning at lr 1e-5 may simply not be enough signal for the
harder, lower-contrast mammography domain before the ISPY2-dominated gradient
direction (even with balanced sampling, ISPY2's sheer size means more distinct
images per epoch) pulls the shared visual encoder toward MRI/ultrasound
features. Next steps if this is picked up again: report per-source accuracy by
default (not just overall) for any future VLM run, try a longer frozen-encoder
linear-probe-first warmup before unfreezing, and/or a lower ISPY2 sampling
weight so more distinct mammography images are seen per epoch relative to the
huge ISPY2 pool.

### MedGemma QLoRA fine-tune (2026-10-06)

`scripts/finetune_medgemma.py` implements the plan above: reuses
`scripts/finetune_biomedclip.py`'s `build_manifest()`/`sample_weights()` as-is
for the image/label pairs and balanced sampling, loads `google/medgemma-4b-it`
4-bit (NF4, bnb), freezes `model.model.vision_tower` entirely, and LoRA-adapts
only the Gemma3 language model's attention/MLP projections (`r=16, alpha=32`,
target modules restricted via a regex anchored on `language_model` — the
vision tower's SigLIP attention reuses the same `q_proj`/`k_proj`/`v_proj`
names, so an unanchored target list would have put LoRA there too). Each
example is a single chat turn: the image + a fixed prompt ("Is this breast
imaging finding benign or malignant? Answer with one word...") per MedGemma's
own `chat_template.jinja`, with loss masked to only the assistant's one-word
answer tokens (5-6 tokens; everything else, including the 256 image soft
tokens, is `-100`).

**Two environment issues hit and fixed, in case they recur:**
- This box has no `python3-dev` (no `Python.h`) and no passwordless `sudo`, so
  PyTorch's JIT-compiled Triton override of `aten::bmm` for the
  outer-product case (used by Gemma3's rotary embeddings) failed to build.
  Fixed without installing anything by calling
  `torch._native.registry.deregister_op_overrides(disable_dsl_names='aten', disable_op_symbols='bmm', disable_dispatch_keys='CUDA')`
  at import time, which falls `bmm` back to the standard (non-Triton) kernel.
- The vision tower lives at `model.model.vision_tower`, not `model.vision_tower`
  — `Gemma3ForConditionalGeneration` wraps an inner `Gemma3Model`.

**Evaluation is unbatched generation (~1.7s/sample on this GPU)**, so
`evaluate()` takes two different sampling modes: `max_samples` (flat random
subset, for quick progress checks during training) and `max_per_source` (caps
each source independently, used for the final report) — the flat mode would
otherwise let ISPY2's 3,705 validation rows dominate wall-clock the same way
they'd dominate accuracy if read naively.

Launched 2026-10-06: `python3 scripts/finetune_medgemma.py --max-steps 250
--grad-accum-steps 8 --eval-every 50 --eval-samples 150
--final-eval-per-source 40`. Timing measured directly (a 3-step/16-accum
dry run): ~6.6s per forward+backward micro-step, so grad-accum 8 gives ~53s/
logical step — 250 steps is roughly a 4-hour run (2,000 training examples
seen, evenly balanced by modality/source/class, out of the 41,476-row
manifest). Logged to W&B project `mmfm-vlm`, run `medgemma_combined_lora`;
LoRA adapter checkpoints saved to `medgemma_combined_lora/step<N>/`.

**Result:** completed all 250 steps (~4h12m wall clock, 16:59-21:11). Train
loss fell smoothly and monotonically throughout (0.1534 at step 110 -> 0.0926
at step 250), but that alone doesn't mean generalization improved — same
caveat as the BiomedCLIP run. Final adapter: `medgemma_combined_lora/step250`
([run](https://wandb.ai/nouran-fadlallah-none/mmfm-vlm/runs/ng1ry7d5)).

Final eval, capped at 40 samples/source (mias and bcsdbt have fewer than 40
validation rows total, so those two are the full validation set; every other
source below is a 40-sample *subset* of a larger validation pool, for
wall-clock reasons given unbatched ~1.7s/sample generation):

| Source | n | Accuracy | BiomedCLIP accuracy (full val set) |
| --- | --- | --- | --- |
| busi | 40 | 0.9000 | 0.8594 |
| mias | 10 (full) | 0.9000 | 0.5000 |
| busc | 25 (full) | 0.8400 | 1.0000 |
| breast | 25 (full) | 0.8000 | 0.8400 |
| busbra | 40 | 0.7750 | 0.8172 |
| ispy2 | 40 | 0.9000 | 0.9987 |
| cdd_cesm | 40 | 0.6000 | 0.5614 |
| cmmd | 40 | 0.5250 | 0.5707 |
| breamdm | 40 | 0.3500 | 0.8718 |
| bcsdbt | 8 (full) | 0.2500 | 0.5000 |
| **overall** | 308 | **0.6948** | 0.705 (non-ispy2-weighted) |

**Read this carefully — the sample sizes above are too small to call most of
these a win or a loss for MedGemma over BiomedCLIP:**

- **MIAS jumped from chance (0.50, n=10) to 0.90 (n=10, same full set)** —
  this is the single comparison here with equal, full sample sizes on both
  sides, so it's the most trustworthy signal in this table. Worth noting since
  MIAS is the smallest, historically noisiest branch in every ResNet run too.
- **BreaDM MRI dropped from 0.87 (n=117, full set) to 0.35 (n=40, a subset)**
  — this is a real concern, not just the smaller sample: 0.35 is *below*
  chance, meaning the model is systematically getting BreaDM wrong in a
  specific direction (plausibly confusing it with ISPY2's all-malignant
  signal, given both are MRI and ISPY2 is 80% of the training manifest).
  Should be re-measured on the full 117-row BreaDM val set before trusting
  this, but it's the most actionable finding here if this gets picked up
  again.
- CMMD and BCS-DBT are still at or below chance, same as BiomedCLIP's result
  — the DICOM-derived mammography sources remain the weak point across both
  VLM attempts, not something specific to one model.
- Intermediate evals during training (the flat, ISPY2-dominated 150-sample
  quick checks at steps 100/150/200/250) swung between 0.64 and 0.95 overall
  — training at this scale (250 steps x grad-accum 8 = 2,000 examples seen,
  out of 41,476) is noisy run-to-run, consistent with how little data a LoRA
  adapter this size has actually been shown.

**Bottom line:** MedGemma-LoRA and BiomedCLIP land at a similar *overall*
accuracy (~0.69-0.70) by two different routes — MedGemma recovers MIAS,
BiomedCLIP is far stronger on BreaDM — neither VLM attempt should be reported
as beating the other or beating the ResNet `FusionLateModel` baselines in
`docs/dataset_reproduction_plan.md` without a same-size, same-split,
multi-seed comparison. If this is picked up again: re-run the final eval on
full (not 40-capped) validation sets per source now that the one-time cost is
known (~9 minutes for the 40-cap version; the full 4,565-row val set would be
~2+ hours per checkpoint, so batch `model.generate()` calls rather than
looping one-by-one before doing that), and investigate the BreaDM regression
specifically (try excluding ISPY2 from the sampler, or down-weighting it
further, and see if BreaDM recovers). `google/paligemma2-3b-pt-224` (also
gated, same unblock path) and `Qwen/Qwen2-VL-2B-Instruct` (Apache-2.0, **not
gated**, 2.2B params) remain reasonable fallback/comparison generative VLMs
using the same recipe.

### Ungated generative-VLM alternative not yet attempted: LLaVA-Med

`microsoft/llava-med-v1.5-mistral-7b` (Apache-2.0, **not gated**, verified via
the Hub API) is a genuine medical-domain instruction-tuned VLM (biomedical
figures + PubMed captions, not breast-specialized — see the Generic Models
section above) that could be QLoRA fine-tuned tonight's way without waiting on
any access grant. It was not attempted in this session to avoid a third
concurrent GPU job (7B params even at 4-bit, plus the two runs already going,
risked destabilizing all three on a single 12GB card) and because the prompt
formatting/QLoRA setup is materially more engineering than BiomedCLIP's
classifier head. It's the next concrete candidate if MedGemma access is still
pending next time this repo is worked on.

## Recommended MMFM Evaluation Order

1. Establish existing single-modality classifier/segmenter baselines with the
   exact manifests and splits in [docs/dataset_reproduction_plan.md](dataset_reproduction_plan.md).
2. For future full-field mammography, reproduce Mammo-CLIP preprocessing and
   run its released checkpoint as frozen features, then a linear probe. Compare
   against a conventionally pretrained ResNet-18 using the same split.
3. Repeat the mammography transfer experiment with Mammo-FM, respecting its
   research-only model terms and four-view/high-resolution requirements.
4. Use BreastGPT only for a separately designed instruction/VQA, grounding, or
   report task. Do not report its BreastStage-Bench scores as classifier
   performance on MMFM data.
5. Keep modality-specific representations separate unless mammography, MRI,
   and ultrasound are linked to the same patient/exam. Dataset-level multimodal
   training is not evidence of longitudinal patient-level multimodal fusion.

## Reproduction Checklist

- Freeze the source release, model revision, code commit, licence, checkpoint
  hash, prompt/template, and preprocessing configuration.
- Match input orientation, bit-depth/windowing, breast/lesion crop, resolution,
  views, text fields, and patient-level split before comparing a published
  metric.
- Avoid prompt leakage: report construction must not include labels, future
  outcome text, or report content unavailable at inference.
- Evaluate classification at the paper's unit (view, breast, exam, patient, or
  lesion) and grounding against the documented annotation type (box, mask, or
  point).
- Report a conventional non-VLM baseline alongside any VLM result. This makes
  the representation benefit measurable rather than merely impressive-sounding.