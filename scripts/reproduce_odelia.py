"""Reproduce the ODELIA dataset paper's MST baseline (Mueller-Franzes et al. 2025,
arXiv 2506.00474, Table 2: in-distribution macro AUC 79.0, RSH out-of-distribution 63.1).

Three subcommands:

  cache          parquet shards -> one .npz per breast (Pre, Sub_1, T2) + meta.csv,
                 so training doesn't re-parse 256x256x32 nested lists every epoch.
  eval-released  score the released ODELIA-AI/MST weights (Sub_1 only, in_ch=1)
                 on one fold's test split: checks our loading/orientation first.
  train          train MST on one fold, paper recipe: Pre + Sub_1 + T2, random
                 224x224x32 crop, rotation, flips, Gaussian noise, AdamW lr 1e-5,
                 cross-entropy, early stopping; RSH never trained on.

The model class is not vendored: models.py is fetched from the gated
ODELIA-AI/MST Hugging Face repo at run time (as its README instructs), which
needs the repo's and DINOv3's licences accepted on huggingface.co, plus
`pip install x_transformers einops`.

Usage:
  python3 scripts/reproduce_odelia.py cache
  python3 scripts/reproduce_odelia.py eval-released --fold 0
  python3 scripts/reproduce_odelia.py train --fold 0
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ODELIA_ROOT = Path('/mnt/data/mmfm_datasets/MRI/odelia')   # raw release (parquet), re-fetchable
# prepared copy kept outside the raw folder so the raw parquet can be pruned
PREPARED_ROOT = Path('/mnt/data/mmfm_datasets/MRI/odelia_prepared')
CACHE_DIR = PREPARED_ROOT / 'npz'
SPLIT_CSV = PREPARED_ROOT / 'split_unilateral.csv'
SEQUENCES = ('Pre', 'Sub_1', 'T2')
OOD_CENTRE = 'RSH'
LESION_NAMES = ('no lesion', 'benign', 'malignant')
CROP = (224, 224, 32)


# ----------------------------------------------------------------- cache

def build_cache(root=ODELIA_ROOT, cache_dir=CACHE_DIR):
    """One compressed .npz per breast holding the paper's three input sequences
    as int16 (1, X, Y, Z) arrays, in the parquet's (torchio) axis order."""
    import pyarrow.parquet as pq

    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_columns = ['UID', 'PatientID', 'Institution', 'Lesion']
    rows = []
    for shard in sorted((root / 'unilateral').glob('*.parquet')):
        parquet = pq.ParquetFile(shard)
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=meta_columns + [f'Image_{s}' for s in SEQUENCES])
            records = table.to_pylist()
            for record in records:
                out_path = cache_dir / f"{record['UID']}.npz"
                if not out_path.exists():
                    np.savez_compressed(out_path, **{
                        s: np.asarray(record[f'Image_{s}'], dtype=np.int16) for s in SEQUENCES
                    })
                rows.append({k: record[k] for k in meta_columns})
        print(f'{shard.name}: {len(rows)} breasts cached so far', flush=True)
    meta = pd.DataFrame(rows).drop_duplicates('UID')
    meta.to_csv(cache_dir / 'meta.csv', index=False)
    print(f'cached {len(meta)} breasts to {cache_dir}')


# ----------------------------------------------------------------- data

def _crop_or_pad(volume, shape, start=None):
    """Center (or `start`-anchored) crop / zero-pad a (C, X, Y, Z) array."""
    out = volume
    for axis, target in enumerate(shape, start=1):
        size = out.shape[axis]
        if size > target:
            offset = (size - target) // 2 if start is None else start[axis - 1]
            out = np.take(out, range(offset, offset + target), axis=axis)
        elif size < target:
            before = (target - size) // 2
            pad = [(0, 0)] * out.ndim
            pad[axis] = (before, target - size - before)
            out = np.pad(out, pad, constant_values=out.min())
    return out


def _z_normalize(volume):
    """torchio ZNormalization with the released model's mask: voxels strictly
    between the volume's min and max."""
    mask = (volume > volume.min()) & (volume < volume.max())
    values = volume[mask] if mask.any() else volume
    return (volume - values.mean()) / (values.std() + 1e-8)


class OdeliaDataset(torch.utils.data.Dataset):
    """Breasts as (C, D, H, W) float tensors in the released model's layout:
    tio.Flip((1, 0)) then crop to 224x224x32, per-sequence z-normalization,
    then swap X and Z (predict_attention.py's `swapaxes(1, -1)`)."""

    def __init__(self, meta, sequences=SEQUENCES, augment=False, cache_dir=CACHE_DIR):
        self.meta = meta.reset_index(drop=True)
        self.sequences = sequences
        self.augment = augment
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        arrays = np.load(self.cache_dir / f"{row['UID']}.npz")
        volume = np.concatenate([arrays[s] for s in self.sequences]).astype(np.float32)
        volume = np.flip(volume, axis=(1, 2))
        start = None
        if self.augment:
            start = [random.randint(0, max(0, volume.shape[a + 1] - CROP[a])) for a in range(3)]
        volume = _crop_or_pad(volume, CROP, start)
        volume = np.stack([_z_normalize(channel) for channel in volume])
        tensor = torch.from_numpy(np.ascontiguousarray(volume)).transpose(1, 3)  # (C, Z, Y, X)
        if self.augment:
            tensor = self._augment(tensor)
        return tensor, int(row['Lesion'])

    @staticmethod
    def _augment(tensor):
        from torchvision.transforms import functional as TF

        if random.random() < 0.5:
            tensor = tensor.flip(-1)
        if random.random() < 0.5:
            tensor = tensor.flip(-2)
        channels, depth, height, width = tensor.shape
        tensor = TF.rotate(tensor.reshape(channels * depth, height, width), random.uniform(-10.0, 10.0))
        tensor = tensor.reshape(channels, depth, height, width)
        return tensor + 0.1 * torch.randn_like(tensor) * (random.random() < 0.5)


def fold_frames(fold, meta):
    """In-distribution train/val/test for one fold of split_unilateral.csv with
    the RSH centre removed everywhere, plus every RSH breast as the OOD test."""
    splits = pd.read_csv(SPLIT_CSV)
    frame = splits[splits['Fold'] == fold].merge(meta[['UID', 'Lesion']], on='UID')
    in_dist = frame[frame['Institution'] != OOD_CENTRE]
    ood = meta[meta['Institution'] == OOD_CENTRE]
    return (in_dist[in_dist['Split'] == 'train'], in_dist[in_dist['Split'] == 'val'],
            in_dist[in_dist['Split'] == 'test'], ood)


# ----------------------------------------------------------------- model

def _mst_class():
    """MSTRegression from the released repo's models.py (fetched, not vendored)."""
    from huggingface_hub import hf_hub_download

    models_py = hf_hub_download(repo_id='ODELIA-AI/MST', filename='models.py')
    sys.path.insert(0, str(Path(models_py).parent))
    from models import MSTRegression
    return MSTRegression


def load_released_model():
    from huggingface_hub import hf_hub_download

    config = json.loads(Path(hf_hub_download('ODELIA-AI/MST', 'model_config.json')).read_text())
    model = _mst_class()(weights=False, **config['hparams'])
    state_dict = torch.load(hf_hub_download('ODELIA-AI/MST', 'state_dict.pt'), map_location='cpu')
    # saved with transformers 4.x; transformers 5 nests DINOv3ViTModel's blocks under .model
    expected = model.state_dict()
    if 'mst.backbone.model.layer.0.norm1.weight' in expected:
        state_dict = {k.replace('mst.backbone.layer.', 'mst.backbone.model.layer.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


# ----------------------------------------------------------------- metrics

def lesion_metrics(labels, probabilities):
    """The paper's metrics: one-vs-rest AUC per class, macro and micro AUC,
    sensitivity at 90% specificity and specificity at 90% sensitivity (both
    averaged over the three one-vs-rest tasks). Also benign-vs-malignant AUC
    among lesion breasts, which is the task the fusion model needs."""
    from sklearn.metrics import roc_auc_score, roc_curve

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    onehot = np.eye(3)[labels]
    result, sens_at_spec, spec_at_sens = {}, [], []
    for c, name in enumerate(LESION_NAMES):
        result[f'auc_{name.replace(" ", "_")}'] = roc_auc_score(onehot[:, c], probabilities[:, c])
        fpr, tpr, _ = roc_curve(onehot[:, c], probabilities[:, c])
        sens_at_spec.append(np.interp(0.10, fpr, tpr))
        spec_at_sens.append(1.0 - np.interp(0.90, tpr, fpr))
    result['macro_auc'] = float(np.mean([result[f'auc_{n.replace(" ", "_")}'] for n in LESION_NAMES]))
    result['micro_auc'] = roc_auc_score(onehot.ravel(), probabilities.ravel())
    result['sensitivity_at_90_specificity'] = float(np.mean(sens_at_spec))
    result['specificity_at_90_sensitivity'] = float(np.mean(spec_at_sens))
    lesion = labels > 0
    if len(set(labels[lesion])) == 2:
        malignant_share = probabilities[lesion, 2] / probabilities[lesion, 1:].sum(axis=1)
        result['benign_vs_malignant_auc'] = roc_auc_score(labels[lesion] == 2, malignant_share)
    return {k: float(v) for k, v in result.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for volumes, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(volumes.to(device))
        probabilities.append(F.softmax(logits.float(), dim=-1).cpu())
        labels.append(targets)
    return torch.cat(labels).numpy(), torch.cat(probabilities).numpy()


# ----------------------------------------------------------------- commands

def _loader(frame, sequences, batch_size, augment=False, shuffle=False, workers=4):
    return torch.utils.data.DataLoader(
        OdeliaDataset(frame, sequences, augment), batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )


def eval_released(args):
    meta = pd.read_csv(CACHE_DIR / 'meta.csv')
    _, _, test, ood = fold_frames(args.fold, meta)
    device = torch.device(args.device)
    model = load_released_model().to(device)
    for name, frame in (('in-distribution test', test), ('RSH (OOD)', ood)):
        labels, probabilities = predict(model, _loader(frame, ('Sub_1',), args.batch_size), device)
        print(f'released MST, fold {args.fold}, {name} (n={len(frame)}): ' +
              ' '.join(f'{k}={v:.3f}' for k, v in lesion_metrics(labels, probabilities).items()))


def train(args):
    import wandb

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    meta = pd.read_csv(CACHE_DIR / 'meta.csv')
    train_frame, val_frame, test_frame, ood_frame = fold_frames(args.fold, meta)
    sequences = tuple(args.sequences)
    device = torch.device(args.device)
    model = _mst_class()(in_ch=len(sequences), out_ch=3, backbone_type='dinov3', model_size='s',
                         slice_fusion_type='transformer', weights=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    loader = _loader(train_frame, sequences, args.batch_size, augment=True, shuffle=True)
    val_loader = _loader(val_frame, sequences, args.batch_size)
    checkpoint = Path(f'odelia_mst_fold{args.fold}_best.pth')
    run = wandb.init(project='mmfm-odelia', name=args.run_name or f'odelia_mst_fold{args.fold}',
                     config={**vars(args), 'train': len(train_frame), 'val': len(val_frame),
                             'test': len(test_frame), 'ood': len(ood_frame)})
    print(f'fold {args.fold}: train={len(train_frame)} val={len(val_frame)} '
          f'test={len(test_frame)} ood={len(ood_frame)} sequences={sequences}')

    best_score, stale = -float('inf'), 0
    for epoch in range(args.max_epochs):
        model.train()
        optimizer.zero_grad()
        total = 0.0
        for step, (volumes, targets) in enumerate(loader, start=1):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = criterion(model(volumes.to(device)).float(), targets.to(device))
            (loss / args.accumulate).backward()
            total += loss.item() * len(targets)
            if step % args.accumulate == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad()
        labels, probabilities = predict(model, val_loader, device)
        val_loss = F.nll_loss(torch.log(torch.from_numpy(probabilities) + 1e-8), torch.from_numpy(labels)).item()
        val_metrics = lesion_metrics(labels, probabilities)
        print(f'epoch {epoch + 1}: train_loss={total / len(train_frame):.4f} val_loss={val_loss:.4f} '
              f'val_macro_auc={val_metrics["macro_auc"]:.3f}', flush=True)
        run.log({'epoch': epoch + 1, 'train/loss': total / len(train_frame), 'validation/loss': val_loss,
                 **{f'validation/{k}': v for k, v in val_metrics.items()}})
        # validation loss bottoms out early while macro AUC keeps improving
        # (fold 0, 2026-10-16), so select and early-stop on macro AUC by default
        score = val_metrics['macro_auc'] if args.select_on == 'auc' else -val_loss
        if score > best_score:
            best_score, stale = score, 0
            print(f'  checkpoint saved (epoch {epoch + 1})', flush=True)
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f'early stopping after {epoch + 1} epochs')
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    results = {}
    for name, frame in (('test', test_frame), ('ood_rsh', ood_frame)):
        labels, probabilities = predict(model, _loader(frame, sequences, args.batch_size), device)
        results[name] = lesion_metrics(labels, probabilities)
        for k, v in results[name].items():
            run.summary[f'{name}/{k}'] = v
        print(f'{name}: ' + ' '.join(f'{k}={v:.3f}' for k, v in results[name].items()))
    Path(f'odelia_mst_fold{args.fold}_results.json').write_text(json.dumps(results, indent=2))
    run.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('cache')
    for name in ('eval-released', 'train'):
        p = sub.add_parser(name)
        p.add_argument('--fold', type=int, choices=range(5), default=0)
        p.add_argument('--device', default='cuda')
        p.add_argument('--batch-size', type=int, default=1)
    train_parser = sub.choices['train']
    train_parser.add_argument('--sequences', nargs='+', default=list(SEQUENCES), choices=SEQUENCES)
    train_parser.add_argument('--accumulate', type=int, default=8,
                              help='1 x 8 = the paper\'s batch size 8 (batch 2 needs more than 12 GB)')
    train_parser.add_argument('--lr', type=float, default=1e-5)
    train_parser.add_argument('--max-epochs', type=int, default=50)
    train_parser.add_argument('--patience', type=int, default=10)
    train_parser.add_argument('--select-on', choices=('auc', 'loss'), default='auc',
                              help='validation macro AUC (default) or validation loss')
    train_parser.add_argument('--seed', type=int, default=42)
    train_parser.add_argument('--run-name', default=None)
    args = parser.parse_args(argv)
    if args.command == 'cache':
        build_cache()
    elif args.command == 'eval-released':
        eval_released(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
