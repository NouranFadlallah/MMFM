import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
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

# branch order used throughout combined (multi-modality) training
MAMMOGRAPHY_BRANCH, ULTRASOUND_BRANCH, MRI_BRANCH = 0, 1, 2
BRANCH_CHANNELS = (3, 3, 9)
ULTRASOUND_DATASET_CHOICES = ('busbra', 'busi', 'busc', 'breast')


class SingleBackboneClassifier(nn.Module):
    def __init__(self, backbone_name, pretrained, embedding_dim, num_classes, input_channels=3):
        super().__init__()
        self.backbone, feature_dim = create_backbone(
            backbone_name, pretrained=pretrained, remove_head=True
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


def train_one_epoch(model, loader, opt, device, modality_dropout_prob=0.0):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
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


def validate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    # per-branch breakdown: samples where exactly that branch is present
    branch_correct = [0, 0, 0]
    branch_total = [0, 0, 0]
    criterion = nn.CrossEntropyLoss()
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


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _busbra_frame(root, max_samples=None, seed=42, validation_fraction=0.1):
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


def _mias_frame(root, seed=42, validation_fraction=0.2):
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
    patients = metadata['patient'].drop_duplicates().to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    validation_count = max(1, int(len(patients) * validation_fraction))
    validation_patients = set(patients[-validation_count:])
    return metadata[~metadata['patient'].isin(validation_patients)], metadata[metadata['patient'].isin(validation_patients)]


def _busi_frame(root, seed=42, validation_fraction=0.2, max_samples=None):
    root = Path(root)
    metadata_path = root / 'DatasetB.xlsx'
    image_dir = root / 'original'
    mask_dir = root / 'GT'
    metadata = pd.read_excel(metadata_path)
    metadata = metadata[metadata['Type'].isin(['Benign', 'Malignant'])].copy()
    metadata['image'] = metadata['Image'].map(lambda image_id: image_dir / f'{int(image_id):06d}.png')
    metadata['mask'] = metadata['Image'].map(lambda image_id: mask_dir / f'{int(image_id):06d}.png')
    metadata['label'] = metadata['Type'].map({'Benign': 0, 'Malignant': 1})
    metadata = metadata[metadata['image'].map(Path.exists) & metadata['mask'].map(Path.exists)].copy()
    if metadata.empty:
        raise ValueError(f'No paired BUSI images found under {root}')

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


def _busc_frame(root, seed=42, validation_fraction=0.2, max_samples=None):
    root = Path(root) / 'originals'
    rows = []
    for class_name, label in (('benign', 0), ('malignant', 1)):
        for image_path in sorted((root / class_name).glob('*.bmp')):
            rows.append({'image': image_path, 'label': label})
    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise ValueError(f'No BUSC images found under {root}')

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


def _breast_frame(root, seed=42, validation_fraction=0.2):
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

    rng = np.random.default_rng(seed)
    indices = metadata.index.to_numpy(copy=True)
    rng.shuffle(indices)
    validation_count = max(1, int(len(indices) * validation_fraction))
    validation_indices = set(indices[-validation_count:])
    return metadata[~metadata.index.isin(validation_indices)], metadata[metadata.index.isin(validation_indices)]


def _ultrasound_frame(name, args):
    if name == 'busbra':
        root = Path('datasets/ultrasound/busbra/BUSBRA')
        frame = _busbra_frame(root, args.max_samples, args.seed, args.validation_fraction)
        transform_cls = BusbraTransform
    elif name == 'busi':
        root = Path('datasets/ultrasound/BUS/BUS')
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
    parser.add_argument('--epochs', type=int, default=cfg.epochs)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--embedding-dim', type=int, default=cfg.embedding_dim)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained', action='store_true')
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
    args = parser.parse_args(argv)

    _set_seed(args.seed)
    requested_device = args.device
    if requested_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but no GPU is available. Use --device cpu explicitly for a CPU run.')
    device = torch.device(requested_device)

    if args.dataset == 'mias':
        dataset_root = args.dataset_root or Path('datasets/mammography/mias/all-mias')
        train_frame, validation_frame = _mias_frame(dataset_root, args.seed, args.validation_fraction)
        train_transform = MiasTransform(size=args.image_size, augment=True)
        validation_transform = MiasTransform(size=args.image_size, augment=False)
    elif args.dataset == 'busi':
        dataset_root = args.dataset_root or Path('datasets/ultrasound/BUS/BUS')
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
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
        print(f'dataset={args.dataset} device={device} train_samples={len(train_dataset)} validation_samples={len(validation_dataset)}')
        return _train_model(args, device, loader, val_loader)
    elif args.dataset == 'combined':
        train_dataset, validation_dataset, sampler = _build_combined_datasets(args)
        if args.checkpoint_path is None:
            args.checkpoint_path = Path(f'combined_fusion_{args.backbone}_best.pth')
        loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
        val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
        print(f'dataset=combined device={device} train_samples={len(train_dataset)} validation_samples={len(validation_dataset)}')
        return _train_model(args, device, loader, val_loader)
    else:
        dataset_root = args.dataset_root or Path('datasets/ultrasound/busbra/BUSBRA')
        train_frame, validation_frame = _busbra_frame(
            dataset_root, args.max_samples, args.seed, args.validation_fraction
        )
        train_transform = BusbraTransform(size=args.image_size, augment=True)
        validation_transform = BusbraTransform(size=args.image_size, augment=False)
    if args.checkpoint_path is None:
        args.checkpoint_path = Path(f'{args.dataset}_single_{args.backbone}_best.pth')
    train_dataset = _make_dataset(train_frame, train_transform)
    validation_dataset = _make_dataset(validation_frame, validation_transform)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    print(f'dataset={args.dataset} device={device} train_samples={len(train_dataset)} validation_samples={len(validation_dataset)}')

    return _train_model(args, device, loader, val_loader)


def _train_model(args, device, loader, val_loader):

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
        )
    else:
        model = FusionLateModel(
            backbone_names=(args.backbone,) * 3,
            pretrained=args.pretrained,
            embedding_dim=args.embedding_dim,
            num_classes=cfg.num_classes,
            fusion_mode=cfg.fusion_mode,
            input_channels=BRANCH_CHANNELS,
        )
        if args.dataset == 'combined':
            _warm_start_branch(model, MAMMOGRAPHY_BRANCH, args.mammography_init_checkpoint)
            _warm_start_branch(model, ULTRASOUND_BRANCH, args.ultrasound_init_checkpoint)
            _warm_start_branch(model, MRI_BRANCH, args.mri_init_checkpoint)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    best_validation_loss = float('inf')
    epochs_without_improvement = 0

    # combined samples carry exactly one modality each, so dropping it leaves no signal at all
    for epoch in range(args.epochs):
        dropout_prob = 0.0 if (args.single_mode or args.dataset == 'combined') else cfg.modality_dropout_prob
        loss = train_one_epoch(model, loader, opt, device, modality_dropout_prob=dropout_prob)
        validation_loss, acc, branch_accuracy = validate(model, val_loader, device)
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
            if epochs_without_improvement >= args.patience:
                print(f'Early stopping after {epoch+1} epochs')
                break

    run.save(str(args.checkpoint_path))
    run.summary['best_validation_loss'] = best_validation_loss
    run.finish()
    print(f'saved {args.checkpoint_path}')


if __name__ == '__main__':
    main()
