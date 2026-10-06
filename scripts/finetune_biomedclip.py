"""Fine-tune BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
as a benign/malignant classifier over every locally reproduced dataset in this repo.

Why BiomedCLIP and not MedGemma: MedGemma-4b-it and MedSigLIP are both gated on
Hugging Face and this account (noe95) has not been granted access yet (confirmed
2026-09-28 via the Hub API — file listing works, file content/config download is
access-denied). BiomedCLIP is MIT-licensed and ungated, so it is what can actually
be trained here tonight. See docs/breast_imaging_vlms.md for the fuller VLM survey
and the MedGemma follow-up plan.

This is a single flat binary classifier (not per-modality branches like
FusionLateModel): BiomedCLIP's ViT-B/16 image tower expects ordinary 3-channel
224x224 photos, so every source is flattened into one (image, label) manifest:

- Mammography: MIAS (lesion-cropped via its own crop_box), CDD-CESM, CMMD, BCS-DBT
- Ultrasound: BUS-BRA, BUSI, BUSC, BrEaST
- MRI: BreaDM + BreastDCEDL-ISPY2 9-channel img9Se patches, collapsed to a single
  representative DCE-phase channel (index 4, mid acquisition) replicated to RGB
  since there is no established BiomedCLIP convention for multi-phase DCE-MRI.

Each dataset keeps its own already-patient-safe train/val split from
training/train.py's own frame builders; this script only concatenates them.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train import (  # noqa: E402
    _breast_frame, _busbra_frame, _busc_frame, _busi_frame, _cdd_cesm_frame,
    _cmmd_frame, _dbt_frame, _mias_frame,
)

MRI_CHANNEL = 4
BIOMEDCLIP_HF_ID = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'


MODALITY_OF_SOURCE = {
    'mias': 'mammography', 'cdd_cesm': 'mammography', 'cmmd': 'mammography', 'bcsdbt': 'mammography',
    'busbra': 'ultrasound', 'busi': 'ultrasound', 'busc': 'ultrasound', 'breast': 'ultrasound',
    'breamdm': 'mri', 'ispy2': 'mri',
}


def _image_rows(frame, source):
    rows = []
    for _, row in frame.iterrows():
        rows.append({
            'path': str(row['image']),
            'label': int(row['label']),
            'source': source,
            'modality': MODALITY_OF_SOURCE[source],
            'kind': 'image',
            'crop_box': row['crop_box'] if 'crop_box' in row and row['crop_box'] is not None else None,
        })
    return rows


def _mri_rows(root, split, source):
    rows = []
    for class_name, label in (('Benign', 0), ('Malignant', 1)):
        for path in sorted(Path(root, split, class_name).rglob('*.npy')):
            rows.append({'path': str(path), 'label': label, 'source': source, 'modality': 'mri',
                         'kind': 'mri_npy', 'crop_box': None})
    return rows


def sample_weights(rows):
    """Balance sampling probability across modality -> source-within-modality ->
    class-within-source, the same scheme training/train.py's _build_combined_datasets
    uses for the fusion model, so a flat classifier trained on this heavily
    ISPY2-dominated manifest (33,475 of 41,476 train rows) doesn't just learn an
    'MRI-style texture -> malignant' shortcut from raw example counts."""
    modalities = sorted(set(r['modality'] for r in rows))
    sources_by_modality = {m: sorted(set(r['source'] for r in rows if r['modality'] == m)) for m in modalities}
    counts = {}
    for source in set(r['source'] for r in rows):
        source_labels = [r['label'] for r in rows if r['source'] == source]
        counts[source] = pd.Series(source_labels).value_counts().to_dict()

    weights = []
    for row in rows:
        num_modalities = len(modalities)
        num_sources = len(sources_by_modality[row['modality']])
        num_classes = len(counts[row['source']])
        class_count = counts[row['source']][row['label']]
        weights.append((1.0 / num_modalities) * (1.0 / num_sources) * (1.0 / num_classes) * (1.0 / class_count))
    return weights


def build_manifest(args):
    """Returns (train_rows, val_rows), each a list of dicts, by reusing every
    dataset's own frame builder (and hence its own patient-level split)."""
    train_rows, val_rows = [], []

    def add(train_frame, validation_frame, source):
        train_rows.extend(_image_rows(train_frame, source))
        val_rows.extend(_image_rows(validation_frame, source))

    mias_train, mias_val = _mias_frame(Path('datasets/mammography/mias/all-mias'), args.seed, args.validation_fraction)
    add(mias_train, mias_val, 'mias')

    cdd_train, cdd_val = _cdd_cesm_frame(
        Path('datasets/mammography/cdd_cesm'), Path('datasets/mammography/cdd_cesm_annotations.xlsx'),
        args.seed, args.validation_fraction,
    )
    add(cdd_train, cdd_val, 'cdd_cesm')

    cmmd_train, cmmd_val = _cmmd_frame(
        Path('datasets/mammography/cmmd_dicom'), Path('datasets/mammography/cmmd_clinicaldata.xlsx'),
        Path('datasets/mammography/cmmd_png'), args.seed, args.validation_fraction,
    )
    add(cmmd_train, cmmd_val, 'cmmd')

    dbt_labels = Path('datasets/mammography/bcsdbt_labels.csv')
    if dbt_labels.exists():
        dbt_train, dbt_val = _dbt_frame(
            Path('datasets/mammography/bcsdbt_dicom'), dbt_labels,
            Path('datasets/mammography/bcsdbt_file_paths.csv'), Path('datasets/mammography/bcsdbt_boxes.csv'),
            Path('datasets/mammography/bcsdbt_png'), args.seed, args.validation_fraction,
        )
        add(dbt_train, dbt_val, 'bcsdbt')

    busbra_train, busbra_val = _busbra_frame(Path('datasets/ultrasound/busbra/BUSBRA'), None, args.seed, args.validation_fraction)
    add(busbra_train, busbra_val, 'busbra')

    busi_train, busi_val = _busi_frame(Path('datasets/ultrasound/Dataset_BUSI/Dataset_BUSI_with_GT'), args.seed, args.validation_fraction)
    add(busi_train, busi_val, 'busi')

    busc_train, busc_val = _busc_frame(Path('datasets/ultrasound/us-dataset'), args.seed, args.validation_fraction)
    add(busc_train, busc_val, 'busc')

    breast_train, breast_val = _breast_frame(
        Path('datasets/ultrasound/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023'), args.seed, args.validation_fraction
    )
    add(breast_train, breast_val, 'breast')

    train_rows.extend(_mri_rows('datasets/MRI/BreaDM/cls/img9Se', 'train', 'breamdm'))
    val_rows.extend(_mri_rows('datasets/MRI/BreaDM/cls/img9Se', 'val', 'breamdm'))
    if Path('datasets/MRI/ISPY2/cls/img9Se').exists():
        train_rows.extend(_mri_rows('datasets/MRI/ISPY2/cls/img9Se', 'train', 'ispy2'))
        val_rows.extend(_mri_rows('datasets/MRI/ISPY2/cls/img9Se', 'val', 'ispy2'))

    return train_rows, val_rows


class ManifestDataset(Dataset):
    def __init__(self, rows, preprocess):
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row['kind'] == 'mri_npy':
            array = np.load(row['path'])
            channel = array[..., min(MRI_CHANNEL, array.shape[-1] - 1)]
            image = Image.fromarray(channel, mode='L').convert('RGB')
        else:
            image = Image.open(row['path']).convert('RGB')
            if row['crop_box'] is not None:
                image = image.crop(row['crop_box'])
        pixel_values = self.preprocess(image)
        return pixel_values, row['label']


class BiomedCLIPClassifier(nn.Module):
    def __init__(self, biomedclip_model, embed_dim=512, num_classes=2):
        super().__init__()
        self.visual = biomedclip_model.visual
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, pixel_values):
        features = self.visual(pixel_values)
        return self.head(features)


def run_epoch(model, loader, device, optimizer=None, criterion=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for pixel_values, labels in tqdm(loader, leave=False):
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, total_correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--checkpoint-path', type=Path, default=Path('biomedclip_combined_best.pth'))
    parser.add_argument('--wandb-project', default='mmfm-vlm')
    parser.add_argument('--wandb-run-name', default='biomedclip_combined')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda requested but no GPU is available')

    print('building manifest...')
    train_rows, val_rows = build_manifest(args)
    train_by_source = pd.Series([r['source'] for r in train_rows]).value_counts().to_dict()
    val_by_source = pd.Series([r['source'] for r in val_rows]).value_counts().to_dict()
    print(f'train={len(train_rows)} {train_by_source}')
    print(f'val={len(val_rows)} {val_by_source}')

    print('loading BiomedCLIP...')
    biomedclip_model, preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP_HF_ID)
    model = BiomedCLIPClassifier(biomedclip_model).to(device)

    train_dataset = ManifestDataset(train_rows, preprocess)
    val_dataset = ManifestDataset(val_rows, preprocess)
    weights = sample_weights(train_rows)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # sampling is already modality/source/class-balanced, so the loss itself
    # doesn't need class weighting on top of that
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.visual.parameters(), 'lr': args.lr},
        {'params': model.head.parameters(), 'lr': args.head_lr},
    ], weight_decay=1e-4)

    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc = run_epoch(model, val_loader, device, None, criterion)
        elapsed = time.time() - t0
        print(f'epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.0f}s)')
        wandb.log({'epoch': epoch + 1, 'train/loss': train_loss, 'train/accuracy': train_acc,
                    'validation/loss': val_loss, 'validation/accuracy': val_acc})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f'saved {args.checkpoint_path}')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f'early stopping after {epoch+1} epochs')
                break
    wandb.finish()


if __name__ == '__main__':
    main()
