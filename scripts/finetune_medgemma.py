"""QLoRA fine-tune google/medgemma-4b-it as a benign/malignant classifier over
every locally reproduced dataset in this repo (the generative-VLM follow-up to
scripts/finetune_biomedclip.py, now that this HF account has MedGemma access
— see docs/breast_imaging_vlms.md for the access-check history and plan).

Recipe, per that plan: freeze the vision tower entirely, LoRA-adapt only the
Gemma3 language model's attention/MLP projections, 4-bit (NF4) base weights.
Each example is a single-turn chat: one image + a fixed prompt, with the
assistant's one-word answer as the only tokens that get a loss (everything
before it, including the image's 256 soft-image tokens, is masked to -100).

Reuses training/train.py's own frame builders via scripts/finetune_biomedclip
(build_manifest, sample_weights) for the image/label manifest and the same
modality -> source-within-modality -> class-within-source balanced sampling —
see that script's docstring and docs/breast_imaging_vlms.md for why: ISPY2
alone is 80% of the manifest and is malignant-only, so unweighted sampling
would just teach "MRI texture -> malignant" instead of a real signal.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# This box has no python3-dev (no Python.h), so torch's JIT-compiled Triton
# override of aten::bmm for the outer-product case (used by Gemma3's rotary
# embeddings) fails to build. Disable that override so bmm falls back to the
# standard CUDA kernel instead of trying to compile a Triton kernel.
from torch._native import registry as _native_registry  # noqa: E402

_native_registry.deregister_op_overrides(disable_dsl_names='aten', disable_op_symbols='bmm', disable_dispatch_keys='CUDA')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.finetune_biomedclip import MRI_CHANNEL, build_manifest, sample_weights  # noqa: E402

MODEL_ID = 'google/medgemma-4b-it'
PROMPT = 'Is this breast imaging finding benign or malignant? Answer with one word: benign or malignant.'
LABEL_TEXT = {0: 'Benign.', 1: 'Malignant.'}
# restrict LoRA to the language model only; the vision tower's self_attn also
# has q_proj/k_proj/v_proj, so an unanchored target_modules list would hit it too
LORA_TARGET_REGEX = r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'


def load_row_image(row):
    if row['kind'] == 'mri_npy':
        array = np.load(row['path'])
        channel = array[..., min(MRI_CHANNEL, array.shape[-1] - 1)]
        return Image.fromarray(channel, mode='L').convert('RGB')
    image = Image.open(row['path']).convert('RGB')
    if row['crop_box'] is not None:
        image = image.crop(row['crop_box'])
    return image


def build_inputs(processor, image, label, device):
    prompt_messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': PROMPT}]}]
    full_messages = prompt_messages + [{'role': 'assistant', 'content': [{'type': 'text', 'text': LABEL_TEXT[label]}]}]

    prompt_inputs = processor.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt'
    )
    full_inputs = processor.apply_chat_template(
        full_messages, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors='pt'
    )
    prompt_length = prompt_inputs['input_ids'].shape[1]

    labels = full_inputs['input_ids'].clone()
    labels[:, :prompt_length] = -100
    full_inputs['labels'] = labels
    return {k: v.to(device) for k, v in full_inputs.items()}


@torch.no_grad()
def evaluate(model, processor, rows, device, max_samples=None, max_per_source=None):
    """Greedy-decode the one-word answer for each row and report accuracy,
    overall and per source — the per-source breakdown is the point (see
    scripts/finetune_biomedclip's own result for why the flat number alone is
    misleading on this ISPY2-dominated manifest). Generation is unbatched
    (~1-2s/sample on this GPU), so both samplers below cap it:
    max_samples draws a flat random subset (quick progress check during
    training); max_per_source caps each source independently (used for the
    final eval, so ISPY2's 3,705 rows don't dominate wall-clock the way they'd
    dominate accuracy if sampled flatly)."""
    model.eval()
    if max_per_source:
        rng = np.random.default_rng(0)
        by_source = {}
        for row in rows:
            by_source.setdefault(row['source'], []).append(row)
        rows = []
        for source_rows in by_source.values():
            idx = rng.choice(len(source_rows), size=min(max_per_source, len(source_rows)), replace=False)
            rows.extend(source_rows[i] for i in idx)
    elif max_samples:
        rng = np.random.default_rng(0)
        rows = [rows[i] for i in rng.choice(len(rows), size=min(max_samples, len(rows)), replace=False)]

    from collections import defaultdict
    correct_by_source = defaultdict(int)
    total_by_source = defaultdict(int)
    for row in rows:
        image = load_row_image(row)
        prompt_messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': PROMPT}]}]
        inputs = processor.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt'
        ).to(device)
        generated = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        answer = processor.tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().lower()
        predicted_label = 1 if 'malignant' in answer else 0 if 'benign' in answer else -1
        total_by_source[row['source']] += 1
        if predicted_label == row['label']:
            correct_by_source[row['source']] += 1
    model.train()

    for source in sorted(total_by_source):
        n = total_by_source[source]
        acc = correct_by_source[source] / n
        print(f'  {source}: {acc:.4f} (n={n})')
    overall = sum(correct_by_source.values()) / sum(total_by_source.values())
    print(f'  overall: {overall:.4f}')
    return overall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--grad-accum-steps', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--eval-samples', type=int, default=200)
    parser.add_argument('--final-eval-per-source', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('medgemma_combined_lora'))
    parser.add_argument('--wandb-project', default='mmfm-vlm')
    parser.add_argument('--wandb-run-name', default='medgemma_combined_lora')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda')

    print('building manifest...')
    train_rows, val_rows = build_manifest(args)
    print(f'train={len(train_rows)} val={len(val_rows)}')

    print('loading MedGemma in 4-bit...')
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, quantization_config=quant_config, device_map='cuda:0')
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model.model.vision_tower.requires_grad_(False)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias='none',
        target_modules=LORA_TARGET_REGEX,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    weights = sample_weights(train_rows)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=args.max_steps * args.grad_accum_steps, replacement=True)
    indices = iter(sampler)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    print('baseline (pre-fine-tune) eval:')
    baseline_acc = evaluate(model, processor, val_rows, device, max_samples=args.eval_samples)
    wandb.log({'step': 0, 'validation/overall_accuracy': baseline_acc})

    step = 0
    t0 = time.time()
    running_loss = 0.0
    optimizer.zero_grad()
    while step < args.max_steps:
        accum_loss = 0.0
        for _ in range(args.grad_accum_steps):
            row = train_rows[next(indices)]
            image = load_row_image(row)
            inputs = build_inputs(processor, image, row['label'], device)
            outputs = model(**inputs)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        running_loss = 0.95 * running_loss + 0.05 * accum_loss if step > 1 else accum_loss
        if step % 10 == 0:
            elapsed = time.time() - t0
            print(f'step {step}/{args.max_steps} loss={running_loss:.4f} ({elapsed:.0f}s, {elapsed/step:.1f}s/step)')
            wandb.log({'step': step, 'train/loss': running_loss})
        if step % args.eval_every == 0:
            print(f'eval at step {step}:')
            acc = evaluate(model, processor, val_rows, device, max_samples=args.eval_samples)
            wandb.log({'step': step, 'validation/overall_accuracy': acc})
            model.save_pretrained(args.checkpoint_dir / f'step{step}')
            print(f'saved {args.checkpoint_dir / f"step{step}"}')

    print(f'final eval (per source, up to {args.final_eval_per_source} samples each):')
    evaluate(model, processor, val_rows, device, max_per_source=args.final_eval_per_source)
    wandb.finish()


if __name__ == '__main__':
    main()
