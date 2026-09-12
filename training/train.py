import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import cfg
from data.dataset import (
    BreaDMDataset, BreadmTransform, BusbraTransform, MiasTransform,
    SingleModalityBranchDataset, TripleImageDataset,
)
from models.backbone import create_backbone
from models.fusion_model import FusionLateModel
from utils.metrics import classification_metrics

# branch order used throughout combined (multi-modality) training
MAMMOGRAPHY_BRANCH, ULTRASOUND_BRANCH, MRI_BRANCH = 0, 1, 2
BRANCH_CHANNELS = (3, 3, 9)
ULTRASOUND_DATASET_CHOICES = ('busbra', 'busi', 'busc', 'breast')


def _make_loader(dataset, batch_size, num_workers=0, shuffle=False, sampler=None):
    """DataLoader with worker processes so PIL preprocessing (median filter,
    contrast stretch, mask-ROI crop, augmentation, resize) overlaps with the GPU
    forward/backward pass instead of blocking it every batch."""
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=num_workers > 0,
        persistent_workers=num_workers > 0,
    )


class SingleBackboneClassifier(nn.Module):
    def __init__(self, backbone_name, pretrained, embedding_dim, num_classes, input_channels=3,
                 pretrained_weights=None):
        super().__init__()
        self.backbone, feature_dim = create_backbone(
            backbone_name, pretrained=pretrained, remove_head=True,
            pretrained_weights=pretrained_weights,
        )
        if backbone_name.lower().startswith('resnet'):
            first_layer = self.backbone.conv1
            if first_layer.in_channels != input_channels:
                replacement = nn.Conv2d(
                    input_channels, first_layer.out_channels, first_layer.kernel_size,
                    first_layer.stride, first_layer.padding, bias=False
                )
                with torch.no_grad():
                    replacement.weight.copy_(first_layer.weight.mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1))
                self.backbone.conv1 = replacement
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim), nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x1, x2=None, x3=None, presence_mask=None):
        features = self.backbone(x1)
        if features.dim() == 4:
            features = nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        else:
            features = features.flatten(1)
        return self.classifier(self.projection(features)), None, None


def train_one_epoch(model, loader, opt, device, modality_dropout_prob=0.0, criterion=None):
    model.train()
    total_loss = 0.0
    criterion = criterion or nn.CrossEntropyLoss()
    for batch in tqdm(loader):
        x1,x2,x3,y,pres = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        y = y.to(device)
        pres = pres.to(device)

        # modality dropout augmentation
        if modality_dropout_prob > 0:
            mask = torch.bernoulli(torch.ones_like(pres.float()) * (1 - modality_dropout_prob)).bool()
            pres = pres & mask
            # zero-out inputs where dropped
            x1 = x1 * pres[:,0].float().view(-1,1,1,1)
            x2 = x2 * pres[:,1].float().view(-1,1,1,1)
            x3 = x3 * pres[:,2].float().view(-1,1,1,1)

        opt.zero_grad()
        fused, branch_logits, weights = model(x1,x2,x3,pres)
        loss = criterion(fused, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, device, criterion=None):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    # per-branch breakdown: samples where exactly that branch is present
    branch_correct = [0, 0, 0]
    branch_total = [0, 0, 0]
    criterion = criterion or nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            x1,x2,x3,y,pres = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            y = y.to(device)
            pres = pres.to(device)
            fused, branch_logits, weights = model(x1,x2,x3,pres)
            total_loss += criterion(fused, y).item() * y.size(0)
            preds = fused.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
            correct_mask = preds == y
            for i in range(3):
                only_branch_i = pres[:, i] & (pres.sum(dim=1) == 1)
                branch_total[i] += only_branch_i.sum().item()
                branch_correct[i] += (correct_mask & only_branch_i).sum().item()
    branch_accuracy = [
        (branch_correct[i] / branch_total[i]) if branch_total[i] else None
        for i in range(3)
    ]
    return total_loss / total, correct / total, branch_accuracy


def collect_predictions(model, loader, device):
    """Run a single-modality model over a loader and return (probs, targets)
    tensors, for any downstream metric computation without retraining."""
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            x1, x2, x3, y, pres = batch
            x1, x2, x3, pres = x1.to(device), x2.to(device), x3.to(device), pres.to(device)
            fused, _, _ = model(x1, x2, x3, pres)
            all_probs.append(torch.softmax(fused, dim=1).cpu())
            all_targets.append(y)
    return torch.cat(all_probs), torch.cat(all_targets)


def evaluate_full_metrics(model, loader, device):
    """Binary classification metrics (accuracy/sensitivity/specificity/precision/F1/AUC)
    matching the indices in the BUS-BRA paper's Table 3/4, for a single-modality model.
    """
    probs, targets = collect_predictions(model, loader, device)
    return classification_metrics(probs, targets)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _busbra_metadata(root):
    """Load and filter bus_data.csv, resolving image/mask paths. Keeps the paper's
    own K5B/K10B (5-/10-fold pathology CV) fold-id columns for official-split use.
    """
    root = Path(root)
    metadata_path = root / 'bus_data.csv'
    images_dir = root / 'Images'
    masks_dir = root / 'Masks'
    metadata = pd.read_csv(metadata_path)
    required = {'ID', 'Case', 'Pathology'}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f'BUSBRA metadata is missing columns: {sorted(missing)}')

    metadata = metadata[metadata['Pathology'].isin(['benign', 'malignant'])].copy()
    metadata['label'] = metadata['Pathology'].map({'benign': 0, 'malignant': 1})
    metadata['image'] = metadata['ID'].map(lambda image_id: images_dir / f'{image_id}.png')
    metadata['mask'] = metadata['ID'].map(lambda image_id: masks_dir / f'mask_{image_id.removeprefix("bus_")}.png')
    metadata = metadata[metadata['image'].map(Path.exists) & metadata['mask'].map(Path.exists)].copy()
    if metadata.empty:
        raise ValueError(f'No BUSBRA images found under {images_dir}')
    return metadata


def _busbra_frame(root, max_samples=None, seed=42, validation_fraction=0.1):
    metadata = _busbra_metadata(root)

    # Keep both views from a case in the same split to prevent leakage.
    cases = metadata['Case'].drop_duplicates().to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(cases)
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    validation_cases = max(1, int(len(cases) * validation_fraction))
    split_at = len(cases) - validation_cases
    train_cases = set(cases[:split_at])
    train = metadata[metadata['Case'].isin(train_cases)]
    validation = metadata[~metadata['Case'].isin(train_cases)]
    if max_samples:
        train = train.head(max_samples)
        validation = validation.head(max(1, max_samples // 5))
    return train, validation


def _busbra_kfold_frame(root, num_folds, test_fold, seed=42, holdout_fraction=0.2):
    """Reproduce the BUS-BRA paper's evaluation protocol (Section 2.3, Table 3): the
    dataset's own case-level K5B/K10B fold assignment picks the test fold, and the
    remaining folds are further split case-level 80/20 into train/validation (the
    paper's "hold-out" step used to select the best-val-loss epoch).
    """
    metadata = _busbra_metadata(root)
    fold_col = {5: 'K5B', 10: 'K10B'}.get(num_folds)
    if fold_col is None:
        raise ValueError('num_folds must be 5 or 10 to match the BUS-BRA paper folds')
    if not 1 <= test_fold <= num_folds:
        raise ValueError(f'test_fold must be in [1, {num_folds}]')

    test_cases = set(metadata.loc[metadata[fold_col] == test_fold, 'Case'])
    pool_cases = np.array(sorted(set(metadata['Case']) - test_cases))
    rng = np.random.default_rng(seed)
    rng.shuffle(pool_cases)
    validation_count = max(1, int(len(pool_cases) * holdout_fraction))
    validation_cases = set(pool_cases[:validation_count])
    train_cases = set(pool_cases[validation_count:])

    train = metadata[metadata['Case'].isin(train_cases)]
    validation = metadata[metadata['Case'].isin(validation_cases)]
    test = metadata[metadata['Case'].isin(test_cases)]
    return train, validation, test


def _make_dataset(frame, transform):
    values = {
        'img1': frame['image'].map(str),
        'img2': frame['image'].map(str),
        'img3': frame['image'].map(str),
    }
    if 'mask' in frame:
        values['mask'] = frame['mask'].map(str)
    if 'crop_box' in frame:
        values['crop_box'] = frame['crop_box']
    values['label'] = frame['label']
    manifest = pd.DataFrame(values)
    dataset = TripleImageDataset.__new__(TripleImageDataset)
    dataset.df = manifest.reset_index(drop=True)
    dataset.transform1 = transform
    dataset.transform2 = transform
    dataset.transform3 = transform
    dataset.label_map = None
    return dataset


def _mias_metadata(root):
    root = Path(root)
    info_path = root / 'Info.txt'
    image_dir = root
    rows = []
    for line in info_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        fields = line.split()
        if len(fields) < 7 or not fields[0].startswith('mdb') or fields[3] not in {'B', 'M'}:
            continue
        image_id = fields[0]
        x, y, radius = map(int, fields[4:7])
        image_path = image_dir / f'{image_id}.pgm'
        if not image_path.exists():
            continue
        center_y = 1024 - y
        half_size = max(32, int(radius * 1.5))
        rows.append({
            'image': image_path,
            'label': 0 if fields[3] == 'B' else 1,
            'patient': (int(image_id[3:]) + 1) // 2,
            'crop_box': (
                max(0, x - half_size), max(0, center_y - half_size),
                min(1024, x + half_size), min(1024, center_y + half_size),
            ),
        })
    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise ValueError(f'No annotated MIAS images found under {root}')
    return metadata


def _mias_frame(root, seed=42, validation_fraction=0.2):
    metadata = _mias_metadata(root)
    patients = metadata['patient'].drop_duplicates().to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    validation_count = max(1, int(len(patients) * validation_fraction))
    validation_patients = set(patients[-validation_count:])
    return metadata[~metadata['patient'].isin(validation_patients)], metadata[metadata['patient'].isin(validation_patients)]


def _busi_metadata(root):
    """Load the official Al-Dhabyani et al. 2020 BUSI release layout: benign/
    malignant/normal folders, each image paired with an `<image>_mask.png` (extra
    lesion masks like `_mask_1.png` are ignored; every image has a plain mask)."""
    root = Path(root)
    rows = []
    for class_name, label in (('benign', 0), ('malignant', 1)):
        class_dir = root / class_name
        for image_path in sorted(class_dir.glob('*.png')):
            if 'mask' in image_path.stem:
                continue
            mask_path = class_dir / f'{image_path.stem}_mask.png'
            if not mask_path.exists():
                continue
            rows.append({'image': image_path, 'mask': mask_path, 'label': label})
    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise ValueError(f'No paired BUSI images found under {root}')
    return metadata


def _busi_frame(root, seed=42, validation_fraction=0.2, max_samples=None):
    metadata = _busi_metadata(root)
    rng = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []
    for label, group in metadata.groupby('label'):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        validation_count = max(1, int(len(indices) * validation_fraction))
        validation_parts.append(metadata.loc[indices[-validation_count:]])
        train_parts.append(metadata.loc[indices[:-validation_count]])
    train = pd.concat(train_parts).sample(frac=1, random_state=seed)
    validation = pd.concat(validation_parts).sample(frac=1, random_state=seed)
    if max_samples:
        train = train.head(max_samples)
        validation = validation.head(max(1, max_samples // 5))
    return train, validation


def _busc_metadata(root):
    root = Path(root) / 'originals'
    rows = []
    for class_name, label in (('benign', 0), ('malignant', 1)):
        for image_path in sorted((root / class_name).glob('*.bmp')):
            rows.append({'image': image_path, 'label': label})
    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise ValueError(f'No BUSC images found under {root}')
    return metadata


def _busc_frame(root, seed=42, validation_fraction=0.2, max_samples=None):
    metadata = _busc_metadata(root)
    rng = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []
    for label, group in metadata.groupby('label'):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        validation_count = max(1, int(len(indices) * validation_fraction))
        validation_parts.append(metadata.loc[indices[-validation_count:]])
        train_parts.append(metadata.loc[indices[:-validation_count]])
    train = pd.concat(train_parts).sample(frac=1, random_state=seed)
    validation = pd.concat(validation_parts).sample(frac=1, random_state=seed)
    if max_samples:
        train = train.head(max_samples)
        validation = validation.head(max(1, max_samples // 5))
    return train, validation


def _breast_metadata(root):
    root = Path(root)
    metadata_path = next(root.glob('*.xlsx'))
    image_dir = root / 'BrEaST-Lesions_USG-images_and_masks'
    metadata = pd.read_excel(metadata_path, sheet_name=0)
    metadata = metadata[metadata['Classification'].isin(['benign', 'malignant'])].copy()
    metadata['image'] = metadata['Image_filename'].map(lambda name: image_dir / str(name))
    metadata['mask'] = metadata['Mask_tumor_filename'].map(lambda name: image_dir / str(name))
    metadata['label'] = metadata['Classification'].map({'benign': 0, 'malignant': 1})
    metadata = metadata[metadata['image'].map(Path.exists) & metadata['mask'].map(Path.exists)].copy()
    if metadata.empty:
        raise ValueError(f'No paired BrEaST images found under {image_dir}')
    return metadata


def _breast_frame(root, seed=42, validation_fraction=0.2):
    metadata = _breast_metadata(root)
    rng = np.random.default_rng(seed)
    indices = metadata.index.to_numpy(copy=True)
    rng.shuffle(indices)
    validation_count = max(1, int(len(indices) * validation_fraction))
    validation_indices = set(indices[-validation_count:])
    return metadata[~metadata.index.isin(validation_indices)], metadata[metadata.index.isin(validation_indices)]


def _assign_stratified_group_folds(metadata, group_col, num_folds, seed):
    """Assign each group (or each row, if group_col is None) to one of num_folds
    folds, stratified by class label at the group level, with a fixed seed."""
    groups = pd.Series(metadata.index, index=metadata.index) if group_col is None else metadata[group_col]
    group_labels = metadata.groupby(groups)['label'].first()
    rng = np.random.default_rng(seed)
    fold_of_group = {}
    for label in sorted(group_labels.unique()):
        ids = group_labels[group_labels == label].index.to_numpy(copy=True)
        rng.shuffle(ids)
        for i, group_id in enumerate(ids):
            fold_of_group[group_id] = (i % num_folds) + 1
    return groups.map(fold_of_group)


def _group_kfold_frame(metadata, group_col, num_folds, test_fold, seed, holdout_fraction=0.2):
    """Generic k-fold split for datasets without an official fold assignment: build
    our own stratified group folds (patient-level where a group column is given,
    e.g. MIAS's 'patient' or BrEaST's 'CaseID'; row-level otherwise), then apply the
    same test-fold + 80/20 train/val hold-out protocol used for BUS-BRA."""
    if not 1 <= test_fold <= num_folds:
        raise ValueError(f'test_fold must be in [1, {num_folds}]')
    fold_ids = _assign_stratified_group_folds(metadata, group_col, num_folds, seed)
    groups = pd.Series(metadata.index, index=metadata.index) if group_col is None else metadata[group_col]

    test_groups = set(groups[fold_ids == test_fold])
    pool_groups = np.array(sorted(set(groups) - test_groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(pool_groups)
    validation_count = max(1, int(len(pool_groups) * holdout_fraction))
    validation_groups = set(pool_groups[:validation_count])
    train_groups = set(pool_groups[validation_count:])

    train = metadata[groups.isin(train_groups)]
    validation = metadata[groups.isin(validation_groups)]
    test = metadata[groups.isin(test_groups)]
    return train, validation, test


def _ultrasound_frame(name, args):
    if name == 'busbra':
        root = Path('datasets/ultrasound/busbra/BUSBRA')
        frame = _busbra_frame(root, args.max_samples, args.seed, args.validation_fraction)
        transform_cls = BusbraTransform
    elif name == 'busi':
        root = Path('datasets/ultrasound/Dataset_BUSI/Dataset_BUSI_with_GT')
        frame = _busi_frame(root, args.seed, args.validation_fraction, args.max_samples)
        transform_cls = BusbraTransform
    elif name == 'busc':
        root = Path('datasets/ultrasound/us-dataset')
        frame = _busc_frame(root, args.seed, args.validation_fraction, args.max_samples)
        transform_cls = MiasTransform
    elif name == 'breast':
        root = Path('datasets/ultrasound/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023')
        frame = _breast_frame(root, args.seed, args.validation_fraction)
        transform_cls = BusbraTransform
    else:
        raise ValueError(f'Unknown ultrasound dataset {name}')
    train_frame, validation_frame = frame
    train_transform = transform_cls(size=args.image_size, augment=True)
    validation_transform = transform_cls(size=args.image_size, augment=False)
    return train_frame, validation_frame, train_transform, validation_transform


def _branch_dataset_source(train_ds, val_ds, labels, branch_index):
    return {
        'train': SingleModalityBranchDataset(train_ds, branch_index, BRANCH_CHANNELS),
        'val': SingleModalityBranchDataset(val_ds, branch_index, BRANCH_CHANNELS),
        'labels': list(labels),
    }


def _build_combined_sources(args):
    """Build one SingleModalityBranchDataset source per underlying dataset, grouped by branch."""
    sources_by_branch = {MAMMOGRAPHY_BRANCH: [], ULTRASOUND_BRANCH: [], MRI_BRANCH: []}

    # mammography branch: MIAS
    mias_root = Path('datasets/mammography/mias/all-mias')
    train_frame, validation_frame = _mias_frame(mias_root, args.seed, args.validation_fraction)
    train_transform = MiasTransform(size=args.image_size, augment=True)
    validation_transform = MiasTransform(size=args.image_size, augment=False)
    train_ds = _make_dataset(train_frame, train_transform)
    val_ds = _make_dataset(validation_frame, validation_transform)
    sources_by_branch[MAMMOGRAPHY_BRANCH].append(
        _branch_dataset_source(train_ds, val_ds, train_frame['label'], MAMMOGRAPHY_BRANCH)
    )

    # ultrasound branch: one or more of busbra/busi/busc/breast, concatenated
    for name in args.ultrasound_datasets:
        train_frame, validation_frame, train_transform, validation_transform = _ultrasound_frame(name, args)
        train_ds = _make_dataset(train_frame, train_transform)
        val_ds = _make_dataset(validation_frame, validation_transform)
        sources_by_branch[ULTRASOUND_BRANCH].append(
            _branch_dataset_source(train_ds, val_ds, train_frame['label'], ULTRASOUND_BRANCH)
        )

    # MRI branch: BreaDM
    breamdm_root = Path('datasets/MRI/BreaDM/cls/img9Se')
    train_transform = BreadmTransform(size=args.image_size, augment=True)
    validation_transform = BreadmTransform(size=args.image_size, augment=False)
    train_ds = BreaDMDataset(breamdm_root, 'train', train_transform)
    val_ds = BreaDMDataset(breamdm_root, 'val', validation_transform)
    labels = [label for _, label in train_ds.samples]
    sources_by_branch[MRI_BRANCH].append(
        _branch_dataset_source(train_ds, val_ds, labels, MRI_BRANCH)
    )
    return sources_by_branch


def _build_combined_datasets(args):
    """Concatenate all per-modality sources and compute weights that balance across
    branches, then across sources within a branch, then across classes within a source.
    """
    sources_by_branch = _build_combined_sources(args)
    num_branches = sum(1 for sources in sources_by_branch.values() if sources)

    train_parts = []
    val_parts = []
    train_weights = []
    for sources in sources_by_branch.values():
        num_sources = len(sources)
        if num_sources == 0:
            continue
        for source in sources:
            labels = source['labels']
            label_counts = Counter(labels)
            num_classes_present = len(label_counts)
            for label in labels:
                weight = (1.0 / num_branches) * (1.0 / num_sources) * (1.0 / num_classes_present) * (1.0 / label_counts[label])
                train_weights.append(weight)
            train_parts.append(source['train'])
            val_parts.append(source['val'])

    train_dataset = ConcatDataset(train_parts)
    validation_dataset = ConcatDataset(val_parts)
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    return train_dataset, validation_dataset, sampler


def _warm_start_branch(model, branch_index, checkpoint_path):
    """Best-effort load of a single-modality checkpoint's backbone/projection/classifier
    into one branch of a FusionLateModel."""
    if checkpoint_path is None:
        return
    state = torch.load(checkpoint_path, map_location='cpu')
    for prefix, module in (
        ('backbone.', model.backbones[branch_index]),
        ('projection.', model.projections[branch_index]),
        ('classifier.', model.classifiers[branch_index]),
    ):
        sub_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not sub_state:
            continue
        try:
            module.load_state_dict(sub_state, strict=True)
            print(f'warm-started branch {branch_index} {prefix.rstrip(".")} from {checkpoint_path}')
        except RuntimeError as error:
            print(f'skipping warm-start for branch {branch_index} {prefix.rstrip(".")}: {error}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train the three-branch fusion model.')
    parser.add_argument('--dataset', choices=['busbra', 'busi', 'busc', 'breast', 'mias', 'breamdm', 'combined'], default='busbra')
    parser.add_argument('--dataset-root', type=Path, default=None)
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument(
        '--mammography-backbone', default=None,
        help='Backbone for the mammography (mias) branch in --dataset combined. Defaults to --backbone.'
    )
    parser.add_argument(
        '--ultrasound-backbone', default=None,
        help='Backbone for the ultrasound branch in --dataset combined. Defaults to --backbone.'
    )
    parser.add_argument(
        '--mri-backbone', default=None,
        help='Backbone for the MRI (breamdm) branch in --dataset combined. Defaults to --backbone.'
    )
    parser.add_argument('--epochs', type=int, default=cfg.epochs)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--embedding-dim', type=int, default=cfg.embedding_dim)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument(
        '--pretrained-weights', type=Path, default=None,
        help='Local checkpoint (e.g. RadImageNet) to load instead of ImageNet weights. '
             'Used for --single-mode and as a shared default for all three fusion branches.'
    )
    parser.add_argument('--mammography-pretrained-weights', type=Path, default=None)
    parser.add_argument('--ultrasound-pretrained-weights', type=Path, default=None)
    parser.add_argument('--mri-pretrained-weights', type=Path, default=None)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--wandb-project', default='mmfm-busbra')
    parser.add_argument('--wandb-run-name', default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--single-mode', action='store_true')
    parser.add_argument('--checkpoint-path', type=Path, default=None)
    parser.add_argument(
        '--ultrasound-datasets', nargs='+', choices=ULTRASOUND_DATASET_CHOICES,
        default=list(ULTRASOUND_DATASET_CHOICES),
        help='Which ultrasound datasets to concatenate into the ultrasound branch for --dataset combined.'
    )
    parser.add_argument('--mammography-init-checkpoint', type=Path, default=None)
    parser.add_argument('--ultrasound-init-checkpoint', type=Path, default=None)
    parser.add_argument('--mri-init-checkpoint', type=Path, default=None)
    parser.add_argument(
        '--paper-match', action='store_true',
        help='Match the BUS-BRA paper training recipe: SGD(momentum=0.9, lr=1e-3), '
             'weighted cross-entropy, no early stopping, and contrast-stretched inputs.'
    )
    parser.add_argument(
        '--busbra-kfold', type=int, choices=[5, 10], default=None,
        help='Use the BUS-BRA dataset\'s own case-level K5B/K10B fold assignment '
             '(Gomez-Flores et al. 2024) instead of an ad-hoc random split.'
    )
    parser.add_argument(
        '--busbra-test-fold', type=int, default=None,
        help='Run only this fold (1-indexed) of --busbra-kfold. Omit to run all folds '
             'sequentially and report the mean/std, like the paper\'s Table 3.'
    )
    parser.add_argument(
        '--kfold', type=int, choices=[5, 10], default=None,
        help='Cross-validate busi/busc/breast/mias with our own stratified group '
             'k-fold (no official fold assignment exists for these datasets, unlike '
             'BUS-BRA), grouped by patient/case where available.'
    )
    parser.add_argument(
        '--test-fold', type=int, default=None,
        help='Run only this fold (1-indexed) of --kfold. Omit to run all folds '
             'sequentially and report the mean/std.'
    )
    args = parser.parse_args(argv)

    _set_seed(args.seed)
    requested_device = args.device
    if requested_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but no GPU is available. Use --device cpu explicitly for a CPU run.')
    device = torch.device(requested_device)

    if args.dataset in DATASET_KFOLD_SPECS and args.kfold is not None:
        return _run_generic_kfold(args, device)
    elif args.dataset == 'mias':
        dataset_root = args.dataset_root or Path('datasets/mammography/mias/all-mias')
        train_frame, validation_frame = _mias_frame(dataset_root, args.seed, args.validation_fraction)
        train_transform = MiasTransform(size=args.image_size, augment=True)
        validation_transform = MiasTransform(size=args.image_size, augment=False)
    elif args.dataset == 'busi':
        dataset_root = args.dataset_root or Path('datasets/ultrasound/Dataset_BUSI/Dataset_BUSI_with_GT')
        train_frame, validation_frame = _busi_frame(
            dataset_root, args.seed, args.validation_fraction, args.max_samples
        )
        train_transform = BusbraTransform(size=args.image_size, augment=True)
        validation_transform = BusbraTransform(size=args.image_size, augment=False)
    elif args.dataset == 'busc':
        dataset_root = args.dataset_root or Path('datasets/ultrasound/us-dataset')
        train_frame, validation_frame = _busc_frame(
            dataset_root, args.seed, args.validation_fraction, args.max_samples
        )
        train_transform = MiasTransform(size=args.image_size, augment=True)
        validation_transform = MiasTransform(size=args.image_size, augment=False)
    elif args.dataset == 'breast':
        dataset_root = args.dataset_root or Path('datasets/ultrasound/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023')
        train_frame, validation_frame = _breast_frame(
            dataset_root, args.seed, args.validation_fraction
        )
        train_transform = BusbraTransform(size=args.image_size, augment=True)
        validation_transform = BusbraTransform(size=args.image_size, augment=False)
    elif args.dataset == 'breamdm':
        dataset_root = args.dataset_root or Path('datasets/MRI/BreaDM/cls/img9Se')
        train_transform = BreadmTransform(size=args.image_size, augment=True)
        validation_transform = BreadmTransform(size=args.image_size, augment=False)
        train_dataset = BreaDMDataset(dataset_root, 'train', train_transform)
        validation_dataset = BreaDMDataset(dataset_root, 'val', validation_transform)
        if args.checkpoint_path is None:
            args.checkpoint_path = Path(f'{args.dataset}_single_{args.backbone}_best.pth')
        loader = _make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
        val_loader = _make_loader(validation_dataset, args.batch_size, args.num_workers)
        print(f'dataset={args.dataset} device={device} train_samples={len(train_dataset)} validation_samples={len(validation_dataset)}')
        return _train_model(args, device, loader, val_loader)
    elif args.dataset == 'combined':
        train_dataset, validation_dataset, sampler = _build_combined_datasets(args)
        if args.checkpoint_path is None:
            args.checkpoint_path = Path(f'combined_fusion_{args.backbone}_best.pth')
        loader = _make_loader(train_dataset, args.batch_size, args.num_workers, sampler=sampler)
        val_loader = _make_loader(validation_dataset, args.batch_size, args.num_workers)
        print(f'dataset=combined device={device} train_samples={len(train_dataset)} validation_samples={len(validation_dataset)}')
        return _train_model(args, device, loader, val_loader)
    elif args.dataset == 'busbra' and args.busbra_kfold is not None:
        return _run_busbra_kfold(args, device)
    else:
        dataset_root = args.dataset_root or Path('datasets/ultrasound/busbra/BUSBRA')
        train_frame, validation_frame = _busbra_frame(
            dataset_root, args.max_samples, args.seed, args.validation_fraction
        )
        train_transform = BusbraTransform(size=args.image_size, augment=True, contrast_stretch=args.paper_match)
        validation_transform = BusbraTransform(size=args.image_size, augment=False, contrast_stretch=args.paper_match)
    if args.checkpoint_path is None:
        args.checkpoint_path = Path(f'{args.dataset}_single_{args.backbone}_best.pth')
    train_dataset = _make_dataset(train_frame, train_transform)
    validation_dataset = _make_dataset(validation_frame, validation_transform)
    loader = _make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    val_loader = _make_loader(validation_dataset, args.batch_size, args.num_workers)
    print(f'dataset={args.dataset} device={device} train_samples={len(train_dataset)} validation_samples={len(validation_dataset)}')

    class_weights = _class_weights(train_frame['label'], cfg.num_classes) if args.paper_match else None
    return _train_model(args, device, loader, val_loader, class_weights=class_weights)


def _run_busbra_kfold(args, device):
    """Run the BUS-BRA official k-fold protocol: one train/val/test split per fold
    (test fold from K5B/K10B, val is an 80/20 hold-out of the remaining folds), then
    report mean +/- std across folds like the paper's Table 3."""
    if not args.single_mode:
        raise ValueError('--busbra-kfold requires --single-mode (the paper benchmarks a single ResNet backbone, not fusion)')
    dataset_root = args.dataset_root or Path('datasets/ultrasound/busbra/BUSBRA')
    base_checkpoint = args.checkpoint_path or Path(f'busbra_single_{args.backbone}_best.pth')
    base_run_name = args.wandb_run_name
    folds = [args.busbra_test_fold] if args.busbra_test_fold else list(range(1, args.busbra_kfold + 1))

    fold_test_metrics = []
    for fold in folds:
        train_frame, validation_frame, test_frame = _busbra_kfold_frame(
            dataset_root, args.busbra_kfold, fold, args.seed, holdout_fraction=0.2
        )
        train_transform = BusbraTransform(size=args.image_size, augment=True, contrast_stretch=args.paper_match)
        eval_transform = BusbraTransform(size=args.image_size, augment=False, contrast_stretch=args.paper_match)
        train_dataset = _make_dataset(train_frame, train_transform)
        validation_dataset = _make_dataset(validation_frame, eval_transform)
        test_dataset = _make_dataset(test_frame, eval_transform)
        loader = _make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
        val_loader = _make_loader(validation_dataset, args.batch_size, args.num_workers)
        test_loader = _make_loader(test_dataset, args.batch_size, args.num_workers)
        print(f'--- BUS-BRA {args.busbra_kfold}-fold, fold {fold}/{args.busbra_kfold} --- '
              f'train={len(train_dataset)} val={len(validation_dataset)} test={len(test_dataset)}')

        args.checkpoint_path = base_checkpoint.with_stem(f'{base_checkpoint.stem}_fold{fold}')
        args.wandb_run_name = f'{base_run_name}-fold{fold}' if base_run_name else f'busbra-{args.busbra_kfold}fold-{fold}'
        class_weights = _class_weights(train_frame['label'], cfg.num_classes) if args.paper_match else None
        result = _train_model(args, device, loader, val_loader, test_loader=test_loader, class_weights=class_weights)
        fold_test_metrics.append(result['test_metrics'])

    metric_names = fold_test_metrics[0].keys()
    summary = {
        name: {
            'mean': float(np.mean([m[name] for m in fold_test_metrics])),
            'std': float(np.std([m[name] for m in fold_test_metrics])),
        }
        for name in metric_names
    }
    print(f'=== BUS-BRA {args.busbra_kfold}-fold summary (test-fold metrics, mean +/- std) ===')
    for name, stats in summary.items():
        print(f'{name}: {stats["mean"]:.4f} +/- {stats["std"]:.4f}')

    summary_path = Path(f'busbra_{args.busbra_kfold}fold_{args.backbone}_results.json')
    with open(summary_path, 'w') as f:
        json.dump({'per_fold': fold_test_metrics, 'summary': summary}, f, indent=2)
    print(f'saved {summary_path}')
    return summary


# datasets with no official fold assignment: cross-validate with our own stratified
# group k-fold, grouped by patient/case where the dataset provides one
DATASET_KFOLD_SPECS = {
    'busi': {
        'metadata_fn': _busi_metadata,
        'default_root': Path('datasets/ultrasound/Dataset_BUSI/Dataset_BUSI_with_GT'),
        'group_col': None,
        'transform_cls': BusbraTransform,
    },
    'busc': {
        'metadata_fn': _busc_metadata,
        'default_root': Path('datasets/ultrasound/us-dataset'),
        'group_col': None,
        'transform_cls': MiasTransform,
    },
    'breast': {
        'metadata_fn': _breast_metadata,
        'default_root': Path('datasets/ultrasound/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023'),
        'group_col': 'CaseID',
        'transform_cls': BusbraTransform,
    },
    'mias': {
        'metadata_fn': _mias_metadata,
        'default_root': Path('datasets/mammography/mias/all-mias'),
        'group_col': 'patient',
        'transform_cls': MiasTransform,
    },
}


def _run_generic_kfold(args, device):
    """Cross-validate busi/busc/breast/mias with our own stratified group k-fold
    (none of them ship an official fold assignment like BUS-BRA's K5B/K10B)."""
    if not args.single_mode:
        raise ValueError('--kfold requires --single-mode (a single ResNet backbone, not fusion)')
    spec = DATASET_KFOLD_SPECS[args.dataset]
    dataset_root = args.dataset_root or spec['default_root']
    metadata = spec['metadata_fn'](dataset_root)
    transform_cls = spec['transform_cls']
    supports_contrast_stretch = transform_cls is BusbraTransform

    base_checkpoint = args.checkpoint_path or Path(f'{args.dataset}_single_{args.backbone}_best.pth')
    base_run_name = args.wandb_run_name
    folds = [args.test_fold] if args.test_fold else list(range(1, args.kfold + 1))

    fold_test_metrics = []
    for fold in folds:
        train_frame, validation_frame, test_frame = _group_kfold_frame(
            metadata, spec['group_col'], args.kfold, fold, args.seed, holdout_fraction=0.2
        )
        transform_kwargs = {'contrast_stretch': args.paper_match} if supports_contrast_stretch else {}
        train_transform = transform_cls(size=args.image_size, augment=True, **transform_kwargs)
        eval_transform = transform_cls(size=args.image_size, augment=False, **transform_kwargs)
        train_dataset = _make_dataset(train_frame, train_transform)
        validation_dataset = _make_dataset(validation_frame, eval_transform)
        test_dataset = _make_dataset(test_frame, eval_transform)
        loader = _make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
        val_loader = _make_loader(validation_dataset, args.batch_size, args.num_workers)
        test_loader = _make_loader(test_dataset, args.batch_size, args.num_workers)
        print(f'--- {args.dataset} {args.kfold}-fold, fold {fold}/{args.kfold} --- '
              f'train={len(train_dataset)} val={len(validation_dataset)} test={len(test_dataset)}')

        args.checkpoint_path = base_checkpoint.with_stem(f'{base_checkpoint.stem}_fold{fold}')
        args.wandb_run_name = f'{base_run_name}-fold{fold}' if base_run_name else f'{args.dataset}-{args.kfold}fold-{fold}'
        class_weights = _class_weights(train_frame['label'], cfg.num_classes) if args.paper_match else None
        result = _train_model(args, device, loader, val_loader, test_loader=test_loader, class_weights=class_weights)
        fold_test_metrics.append(result['test_metrics'])

    metric_names = fold_test_metrics[0].keys()
    summary = {
        name: {
            'mean': float(np.mean([m[name] for m in fold_test_metrics])),
            'std': float(np.std([m[name] for m in fold_test_metrics])),
        }
        for name in metric_names
    }
    print(f'=== {args.dataset} {args.kfold}-fold summary (test-fold metrics, mean +/- std) ===')
    for name, stats in summary.items():
        print(f'{name}: {stats["mean"]:.4f} +/- {stats["std"]:.4f}')

    summary_path = Path(f'{args.dataset}_{args.kfold}fold_{args.backbone}_results.json')
    with open(summary_path, 'w') as f:
        json.dump({'per_fold': fold_test_metrics, 'summary': summary}, f, indent=2)
    print(f'saved {summary_path}')
    return summary


def _class_weights(labels, num_classes):
    """Inverse-frequency class weights, matching the BUS-BRA paper's weighted
    cross-entropy for its ~9:19 malignant:benign imbalance."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _train_model(args, device, loader, val_loader, test_loader=None, class_weights=None):

    import wandb
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={**vars(args), 'device': str(device), 'train_samples': len(loader.dataset), 'validation_samples': len(val_loader.dataset)},
    )

    if args.single_mode:
        model = SingleBackboneClassifier(
            backbone_name=args.backbone,
            pretrained=args.pretrained,
            embedding_dim=args.embedding_dim,
            num_classes=cfg.num_classes,
            input_channels=9 if args.dataset == 'breamdm' else 3,
            pretrained_weights=args.pretrained_weights,
        )
    else:
        branch_weights = (
            args.mammography_pretrained_weights or args.pretrained_weights,
            args.ultrasound_pretrained_weights or args.pretrained_weights,
            args.mri_pretrained_weights or args.pretrained_weights,
        )
        branch_backbones = (
            args.mammography_backbone or args.backbone,
            args.ultrasound_backbone or args.backbone,
            args.mri_backbone or args.backbone,
        )
        model = FusionLateModel(
            backbone_names=branch_backbones,
            pretrained=args.pretrained,
            embedding_dim=args.embedding_dim,
            num_classes=cfg.num_classes,
            fusion_mode=cfg.fusion_mode,
            input_channels=BRANCH_CHANNELS,
            pretrained_weights=branch_weights,
        )
        if args.dataset == 'combined':
            _warm_start_branch(model, MAMMOGRAPHY_BRANCH, args.mammography_init_checkpoint)
            _warm_start_branch(model, ULTRASOUND_BRANCH, args.ultrasound_init_checkpoint)
            _warm_start_branch(model, MRI_BRANCH, args.mri_init_checkpoint)
    model.to(device)

    paper_match = getattr(args, 'paper_match', False)
    if paper_match:
        opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        opt = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    best_validation_loss = float('inf')
    epochs_without_improvement = 0
    # the paper trains a fixed 100-epoch budget with no early stopping, only
    # snapshotting the lowest-val-loss epoch; --paper-match reproduces that
    patience = args.epochs if paper_match else args.patience

    # combined samples carry exactly one modality each, so dropping it leaves no signal at all
    for epoch in range(args.epochs):
        dropout_prob = 0.0 if (args.single_mode or args.dataset == 'combined') else cfg.modality_dropout_prob
        loss = train_one_epoch(model, loader, opt, device, modality_dropout_prob=dropout_prob, criterion=criterion)
        validation_loss, acc, branch_accuracy = validate(model, val_loader, device, criterion=criterion)
        branch_names = ('mammography', 'ultrasound', 'mri')
        branch_summary = ' '.join(
            f'{name}_acc={value:.4f}' if value is not None else f'{name}_acc=n/a'
            for name, value in zip(branch_names, branch_accuracy)
        )
        print(f'Epoch {epoch+1}/{args.epochs} loss={loss:.4f} val_acc={acc:.4f} {branch_summary}')
        log_payload = {'epoch': epoch + 1, 'train/loss': loss, 'validation/loss': validation_loss, 'validation/accuracy': acc}
        for name, value in zip(branch_names, branch_accuracy):
            if value is not None:
                log_payload[f'validation/{name}_accuracy'] = value
        run.log(log_payload)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break

    result = {'best_validation_loss': best_validation_loss}
    if args.single_mode:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        validation_metrics = evaluate_full_metrics(model, val_loader, device)
        for name, value in validation_metrics.items():
            run.summary[f'validation/{name}'] = value
        print('Final validation metrics (best checkpoint): ' +
              ' '.join(f'{k}={v:.4f}' for k, v in validation_metrics.items()))
        result['validation_metrics'] = validation_metrics
        if test_loader is not None:
            test_metrics = evaluate_full_metrics(model, test_loader, device)
            for name, value in test_metrics.items():
                run.summary[f'test/{name}'] = value
            print('Held-out test-fold metrics: ' +
                  ' '.join(f'{k}={v:.4f}' for k, v in test_metrics.items()))
            result['test_metrics'] = test_metrics

    run.save(str(args.checkpoint_path))
    run.summary['best_validation_loss'] = best_validation_loss
    run.finish()
    print(f'saved {args.checkpoint_path}')
    return result


if __name__ == '__main__':
    main()
