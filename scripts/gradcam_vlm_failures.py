"""Grad-CAM on misclassified validation examples from the two VLM experiments
(scripts/finetune_biomedclip.py, scripts/finetune_medgemma.py), adapted for
ViT image towers via utils/gradcam.py's ViTGradCAM (CNN Grad-CAM reshapes a
[B,C,H,W] feature map; a ViT block outputs a [B,N,D] token sequence, so
ViTGradCAM drops any prefix/CLS tokens and reshapes the remaining patch
tokens into a spatial grid before doing the same weighted-channel-sum CAM).

Picks a handful of wrong cases per source (prioritizing the sources flagged
in docs/breast_imaging_vlms.md: CMMD/BCS-DBT/MIAS for BiomedCLIP's
near-chance mammography result, and BreastDM for MedGemma's regression),
scores Grad-CAM on the *predicted* (wrong) class/token so the heatmap shows
what the model actually looked at when it got the label wrong.

Usage: .venv/bin/python3 scripts/gradcam_vlm_failures.py
Writes docs/gradcam_vlm/{biomedclip,medgemma}/<source>/*.png and a JSON
manifest at docs/gradcam_vlm_manifest.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch._native import registry as _native_registry  # noqa: E402

_native_registry.deregister_op_overrides(disable_dsl_names='aten', disable_op_symbols='bmm', disable_dispatch_keys='CUDA')

import open_clip  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig  # noqa: E402

from peft import prepare_model_for_kbit_training  # noqa: E402

from scripts.finetune_biomedclip import (  # noqa: E402
    BIOMEDCLIP_HF_ID, BiomedCLIPClassifier, ManifestDataset, build_manifest,
)
from scripts.finetune_medgemma import LABEL_TEXT, MODEL_ID, PROMPT, load_row_image  # noqa: E402
from utils.gradcam import ViTGradCAM, overlay_heatmap  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_ROOT = Path('docs/gradcam_vlm')
CLASS_NAMES = ('benign', 'malignant')

# sources to sweep and how many wrong cases to save per source
BIOMEDCLIP_SOURCES = {'cmmd': 3, 'bcsdbt': 2, 'mias': 2, 'cdd_cesm': 2, 'breamdm': 2}
MEDGEMMA_SOURCES = {'breamdm': 3, 'cmmd': 2, 'bcsdbt': 2, 'mias': 2}

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _unnormalize_clip(tensor_chw):
    array = tensor_chw.numpy().transpose(1, 2, 0)
    array = array * CLIP_STD + CLIP_MEAN
    return np.clip(array, 0.0, 1.0).transpose(2, 0, 1)


def biomedclip_failures():
    args_namespace = type('Args', (), {'seed': 42, 'validation_fraction': 0.1})()
    _, val_rows = build_manifest(args_namespace)

    biomedclip_model, preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP_HF_ID)
    model = BiomedCLIPClassifier(biomedclip_model)
    model.load_state_dict(torch.load('biomedclip_combined_best.pth', map_location='cpu'))
    model.to(DEVICE).eval()

    cam = ViTGradCAM(model.visual.trunk.blocks[-1], num_prefix_tokens=1, grid_size=(14, 14))

    dataset = ManifestDataset(val_rows, preprocess)
    results = []
    remaining = dict(BIOMEDCLIP_SOURCES)
    out_dir_by_source = {}
    for index, row in enumerate(val_rows):
        source = row['source']
        if remaining.get(source, 0) <= 0:
            continue
        pixel_values, label = dataset[index]
        pixel_values = pixel_values.unsqueeze(0).to(DEVICE)
        pixel_values.requires_grad_(False)

        logits = model(pixel_values)
        predicted = int(logits.argmax(dim=1))
        if predicted == label:
            continue  # only interested in mistakes

        score = logits[:, predicted]
        cam_map = cam.compute(score, image_hw=(224, 224))[0]
        original = _unnormalize_clip(pixel_values[0].detach().cpu())
        overlay = overlay_heatmap(original, cam_map)

        out_dir = out_dir_by_source.setdefault(source, OUT_ROOT / 'biomedclip' / source)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(row['path']).stem
        Image.fromarray((original.transpose(1, 2, 0) * 255).astype(np.uint8)).save(out_dir / f'{stem}_original.png')
        Image.fromarray(overlay).save(out_dir / f'{stem}_cam.png')

        results.append({
            'model': 'biomedclip', 'source': source, 'path': row['path'],
            'true_label': CLASS_NAMES[label], 'predicted_label': CLASS_NAMES[predicted],
            'original': str(out_dir / f'{stem}_original.png'), 'cam': str(out_dir / f'{stem}_cam.png'),
        })
        remaining[source] -= 1
        if not any(v > 0 for v in remaining.values()):
            break
    model.visual.trunk.blocks[-1]._forward_hooks.clear()
    model.visual.trunk.blocks[-1]._backward_hooks.clear()
    return results


def medgemma_failures():
    args_namespace = type('Args', (), {'seed': 42, 'validation_fraction': 0.1})()
    _, val_rows = build_manifest(args_namespace)

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, quantization_config=quant_config, device_map='cuda:0')
    # a full backward through this 4B model without checkpointing OOMs a 12GB GPU
    # (same setup finetune_medgemma.py used during training)
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base_model, 'medgemma_combined_lora/step250')
    # HF's gradient-checkpointing path only activates in train() mode (it checks
    # self.training), and without it this backward OOMs a 12GB GPU; dropout is 0
    # everywhere in this model's config so train() vs eval() changes nothing else
    model.train()
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    benign_id, malignant_id = (processor.tokenizer(LABEL_TEXT[label], add_special_tokens=False)['input_ids'][0] for label in (0, 1))

    base = model.get_base_model()
    vision_tower = base.model.vision_tower
    vision_config = base.config.vision_config
    grid_side = vision_config.image_size // vision_config.patch_size  # 896 / 14 = 64
    cam = ViTGradCAM(vision_tower.encoder.layers[-1], num_prefix_tokens=0, grid_size=(grid_side, grid_side))

    results = []
    remaining = dict(MEDGEMMA_SOURCES)
    out_dir_by_source = {}
    for row in val_rows:
        source = row['source']
        if remaining.get(source, 0) <= 0:
            continue
        image = load_row_image(row)
        prompt_messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': PROMPT}]}]
        inputs = processor.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt'
        ).to(DEVICE)
        # the vision tower is entirely frozen (no LoRA there), so nothing upstream
        # of it has requires_grad=True by default; seed the graph at the input
        # or autograd never builds one down to the hooked block
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16).requires_grad_(True)

        logits = model(**inputs).logits[:, -1, :]
        predicted_token = int(logits.argmax(dim=-1))
        predicted = 1 if predicted_token == malignant_id else 0 if predicted_token == benign_id else None
        if predicted is None or predicted == row['label']:
            continue  # only interested in clean benign/malignant mistakes

        score = logits[:, malignant_id] if predicted == 1 else logits[:, benign_id]
        cam_map = cam.compute(score, image_hw=image.size[::-1])[0]
        original = np.asarray(image.resize(image.size), dtype=np.float32).transpose(2, 0, 1) / 255.0
        overlay = overlay_heatmap(original, cam_map)

        out_dir = out_dir_by_source.setdefault(source, OUT_ROOT / 'medgemma' / source)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(row['path']).stem
        Image.fromarray((original.transpose(1, 2, 0) * 255).astype(np.uint8)).save(out_dir / f'{stem}_original.png')
        Image.fromarray(overlay).save(out_dir / f'{stem}_cam.png')

        results.append({
            'model': 'medgemma', 'source': source, 'path': row['path'],
            'true_label': CLASS_NAMES[row['label']], 'predicted_label': CLASS_NAMES[predicted],
            'original': str(out_dir / f'{stem}_original.png'), 'cam': str(out_dir / f'{stem}_cam.png'),
        })
        remaining[source] -= 1
        if not any(v > 0 for v in remaining.values()):
            break
    return results


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print('BiomedCLIP Grad-CAM sweep...')
    biomedclip_results = biomedclip_failures()
    print(f'  found {len(biomedclip_results)} wrong cases')

    print('MedGemma Grad-CAM sweep...')
    medgemma_results = medgemma_failures()
    print(f'  found {len(medgemma_results)} wrong cases')

    manifest = biomedclip_results + medgemma_results
    with open('docs/gradcam_vlm_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'wrote {len(manifest)} entries to docs/gradcam_vlm_manifest.json')


if __name__ == '__main__':
    main()
