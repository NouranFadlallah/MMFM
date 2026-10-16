"""Reproduce BreastDCEDL's ViT pCR baseline on our local I-SPY2 cohort
(Fridman et al. 2025, arXiv 2506.12190, Table 2: I-SPY2 test n=99, image-only
AUC 0.78, accuracy 0.73).

Paper recipe: tumour-containing axial slices; R/G/B = pre-contrast, early and
late post-contrast DCE timepoints (the metadata CSV's `pre`, `post_early`,
`post_late` acquisition indices: 0, 2 and min(last, 6) for I-SPY); min-max
normalized to 8-bit; ImageNet ViT. The paper and repo give no ViT
hyperparameters, so this script's choices (ViT-B/16 at 224, AdamW 2e-5, weight
decay 0.05, batch 32, 10 epochs, flips + small rotations, patient-level score =
mean slice probability, model selection on validation patient AUC) are ours.

Needs BreastDCEDL_metadata.csv (pCR labels, split, tumour slice/row/column
ranges) from github.com/naomifridman/BreastDCEDL; only `spy2` rows are used,
since Duke and I-SPY1 are not downloaded locally. Its `test` column codes the
split as 0/1/2: 0 = train, 1 = validation, 2 = test (across all three cohorts
with pCR, 1 holds 177 patients at 29.9% pCR and 2 holds 176 at 30.1%, the
paper's section 2.6 validation and test sets; I-SPY2 alone is 784/99/99).

Usage:
  python3 scripts/reproduce_breastdcedl_pcr.py slices
  python3 scripts/reproduce_breastdcedl_pcr.py train
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ISPY2_ROOT = Path('/mnt/data/mmfm_datasets/MRI/PKG - BreastDCEDL_ISPY2/BreastDCEDL_ISPY2')
SLICE_DIR = Path('/mnt/data/mmfm_datasets/MRI/ispy2_pcr_rgb')
METADATA_CSV = Path('datasets/MRI/BreastDCEDL_metadata/BreastDCEDL_metadata.csv')


def load_metadata(path, val_value=None, test_value=None):
    meta = pd.read_csv(path)
    meta = meta[(meta['dataset'] == 'spy2') & meta['pCR'].notna()].copy()
    counts = meta['test'].value_counts()
    print('I-SPY2 rows per `test` value:', counts.to_dict())
    train_value = counts.idxmax()
    names = {train_value: 'train'}
    if val_value is not None and test_value is not None:
        names.update({float(val_value): 'val', float(test_value): 'test'})
    meta['split'] = meta['test'].map(names)
    return meta


def _to_uint8(array):
    low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    return ((array - low) / (high - low) * 255.0).astype(np.uint8)


def build_slices(meta, out_dir=SLICE_DIR):
    """One RGB PNG per tumour-containing slice: [mask_start, mask_end] on axis 0
    (slices) of the (z, row, col) NIfTI volumes."""
    import nibabel as nib

    rows = []
    for _, patient in meta.iterrows():
        dce_dir = ISPY2_ROOT / patient['pid'] / 'dce'
        if not dce_dir.exists():
            continue
        phases = [int(patient[c]) for c in ('pre', 'post_early', 'post_late')]
        paths = {int(p.name.split('_aqc_')[1].split('.')[0]): p for p in dce_dir.glob('*_aqc_*.nii.gz')}
        if any(p not in paths for p in phases):
            print(f"{patient['pid']}: missing DCE phase among {phases}, skipped")
            continue
        volumes = [np.asarray(nib.load(paths[p]).dataobj, dtype=np.float32) for p in phases]
        for z in range(int(patient['mask_start']), int(patient['mask_end']) + 1):
            out_path = out_dir / patient['pid'] / f'z{z:03d}.png'
            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                rgb = np.stack([_to_uint8(v[z]) for v in volumes], axis=-1)
                Image.fromarray(rgb).save(out_path)
            rows.append({'pid': patient['pid'], 'path': str(out_path)})
    index = pd.DataFrame(rows)
    index.to_csv(out_dir / 'index.csv', index=False)
    print(f'{len(index)} slices from {index.pid.nunique()} patients -> {out_dir}')


class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, frame, augment=False, size=224, strong=False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
        self.size = size
        self.strong = strong

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        from torchvision.transforms import functional as TF

        row = self.frame.iloc[index]
        image = Image.open(row['path']).convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(image)
        if self.augment:
            if random.random() < 0.5:
                tensor = tensor.flip(-1)
            if random.random() < 0.5:
                tensor = tensor.flip(-2)
            tensor = TF.rotate(tensor, random.uniform(-15.0, 15.0))
            if self.strong:
                # random resized crop (80-100% area) plus brightness / contrast jitter
                scale = random.uniform(0.8, 1.0) ** 0.5
                crop = int(self.size * scale)
                top, left = random.randint(0, self.size - crop), random.randint(0, self.size - crop)
                tensor = TF.resize(tensor[:, top:top + crop, left:left + crop], [self.size, self.size], antialias=True)
                tensor = TF.adjust_contrast(TF.adjust_brightness(tensor, random.uniform(0.85, 1.15)),
                                            random.uniform(0.85, 1.15))
        tensor = TF.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return tensor, int(row['pCR'])


@torch.no_grad()
def patient_scores(model, frame, device, batch_size):
    """Mean slice pCR probability per patient."""
    model.eval()
    loader = torch.utils.data.DataLoader(SliceDataset(frame), batch_size=batch_size, num_workers=4)
    probabilities = []
    for images, _ in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(images.to(device))
        probabilities.append(F.softmax(logits.float(), dim=-1)[:, 1].cpu())
    scored = frame.assign(prob=torch.cat(probabilities).numpy())
    return scored.groupby('pid').agg(prob=('prob', 'mean'), pCR=('pCR', 'first'))


def pcr_metrics(patients, threshold=0.5):
    from sklearn.metrics import roc_auc_score

    labels, scores = patients['pCR'].to_numpy().astype(int), patients['prob'].to_numpy()
    predicted = scores >= threshold
    positives, negatives = labels == 1, labels == 0
    return {
        'n': int(len(labels)), 'auc': float(roc_auc_score(labels, scores)),
        'accuracy': float((predicted == positives).mean()),
        'sensitivity': float(predicted[positives].mean()), 'specificity': float((~predicted[negatives]).mean()),
    }


def train(args):
    import timm
    import wandb

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    meta = load_metadata(args.metadata, args.val_value, args.test_value)
    if meta['split'].isna().any():
        raise SystemExit('pass --val-value and --test-value (see the counts above)')
    index = pd.read_csv(SLICE_DIR / 'index.csv').merge(meta[['pid', 'pCR', 'split']], on='pid')
    frames = {name: index[index['split'] == name] for name in ('train', 'val', 'test')}
    for name, frame in frames.items():
        print(f'{name}: {frame.pid.nunique()} patients, {len(frame)} slices, '
              f'pCR rate {frame.groupby("pid").pCR.first().mean():.3f}')

    device = torch.device(args.device)
    model = timm.create_model(args.model, pretrained=True, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def epoch_loader(epoch):
        """All training slices, or (--slices-per-patient K) a fresh draw of K
        slices per patient each epoch: about 38 near-identical slices per
        patient let the ViT memorize patients (first run, 2026-10-16)."""
        frame = frames['train']
        if args.slices_per_patient:
            frame = frame.sample(frac=1.0, random_state=args.seed + epoch).groupby('pid').head(args.slices_per_patient)
        return torch.utils.data.DataLoader(SliceDataset(frame, augment=True, strong=args.strong_augment),
                                           batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    run = wandb.init(project='mmfm-ispy2-pcr', name=args.run_name, config=vars(args))
    checkpoint = Path(f'{args.run_name}_best.pth')
    best_auc = -1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        loader = epoch_loader(epoch)
        for images, labels in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = F.cross_entropy(model(images.to(device)).float(), labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        val = pcr_metrics(patient_scores(model, frames['val'], device, args.batch_size))
        print(f'epoch {epoch + 1}: train_loss={total / len(loader.dataset):.4f} val_auc={val["auc"]:.3f}', flush=True)
        run.log({'epoch': epoch + 1, 'train/loss': total / len(loader.dataset),
                 **{f'validation/{k}': v for k, v in val.items()}})
        if val['auc'] > best_auc:
            best_auc = val['auc']
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test = pcr_metrics(patient_scores(model, frames['test'], device, args.batch_size))
    print('I-SPY2 test (patient level): ' + ' '.join(f'{k}={v:.3f}' for k, v in test.items()))
    for k, v in test.items():
        run.summary[f'test/{k}'] = v
    Path(f'{args.run_name}_results.json').write_text(json.dumps({'best_val_auc': best_auc, 'test': test}, indent=2))
    run.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('slices', 'train'):
        p = sub.add_parser(name)
        p.add_argument('--metadata', type=Path, default=METADATA_CSV)
        p.add_argument('--val-value', type=float, default=1.0)
        p.add_argument('--test-value', type=float, default=2.0)
    t = sub.choices['train']
    t.add_argument('--model', default='vit_base_patch16_224.augreg_in21k_ft_in1k')
    t.add_argument('--lr', type=float, default=2e-5)
    t.add_argument('--weight-decay', type=float, default=0.05)
    t.add_argument('--batch-size', type=int, default=32)
    t.add_argument('--epochs', type=int, default=10)
    t.add_argument('--slices-per-patient', type=int, default=0, help='0 = every tumour slice each epoch')
    t.add_argument('--strong-augment', action='store_true')
    t.add_argument('--device', default='cuda')
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--run-name', default='ispy2_pcr_vit_b16')
    args = parser.parse_args(argv)
    if args.command == 'slices':
        build_slices(load_metadata(args.metadata, args.val_value, args.test_value))
    else:
        train(args)


if __name__ == '__main__':
    main()
