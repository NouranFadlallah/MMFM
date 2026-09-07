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