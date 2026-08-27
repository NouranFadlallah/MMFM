import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import cfg
from data.dataset import BusbraTransform, TripleImageDataset
from models.backbone import create_backbone
from models.fusion_model import FusionLateModel


class SingleBackboneClassifier(nn.Module):
    def __init__(self, backbone_name, pretrained, embedding_dim, num_classes):
        super().__init__()
        self.backbone, feature_dim = create_backbone(
            backbone_name, pretrained=pretrained, remove_head=True
        )
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
    return total_loss / total, correct / total


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
    manifest = frame.assign(
        img1=frame['image'].map(str),
        img2=frame['image'].map(str),
        img3=frame['image'].map(str),
        mask=frame['mask'].map(str),
    )[['img1', 'img2', 'img3', 'mask', 'label']]
    dataset = TripleImageDataset.__new__(TripleImageDataset)
    dataset.df = manifest.reset_index(drop=True)
    dataset.transform1 = transform
    dataset.transform2 = transform
    dataset.transform3 = transform
    dataset.label_map = None
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train the three-branch fusion model.')
    parser.add_argument('--dataset-root', type=Path, default=Path('data/busbra/BUSBRA'))
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--epochs', type=int, default=cfg.epochs)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--embedding-dim', type=int, default=cfg.embedding_dim)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--wandb-project', default='mmfm-busbra')
    parser.add_argument('--wandb-run-name', default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--single-mode', action='store_true')
    args = parser.parse_args(argv)

    _set_seed(args.seed)
    requested_device = args.device or cfg.device
    device = torch.device(requested_device if requested_device == 'cuda' and torch.cuda.is_available() else 'cpu')

    train_transform = BusbraTransform(size=args.image_size, augment=True)
    validation_transform = BusbraTransform(size=args.image_size, augment=False)
    train_frame, validation_frame = _busbra_frame(
        args.dataset_root, args.max_samples, args.seed, args.validation_fraction
    )
    train_dataset = _make_dataset(train_frame, train_transform)
    validation_dataset = _make_dataset(validation_frame, validation_transform)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    print(f'device={device} train_samples={len(train_dataset)} validation_samples={len(validation_dataset)}')

    import wandb
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={**vars(args), 'device': str(device), 'train_samples': len(train_dataset), 'validation_samples': len(validation_dataset)},
    )

    if args.single_mode:
        model = SingleBackboneClassifier(
            backbone_name=args.backbone,
            pretrained=args.pretrained,
            embedding_dim=args.embedding_dim,
            num_classes=cfg.num_classes,
        )
    else:
        model = FusionLateModel(
            backbone_names=(args.backbone,) * 3,
            pretrained=args.pretrained,
            embedding_dim=args.embedding_dim,
            num_classes=cfg.num_classes,
            fusion_mode=cfg.fusion_mode,
        )
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    best_validation_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, opt, device, modality_dropout_prob=cfg.modality_dropout_prob)
        validation_loss, acc = validate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{args.epochs} loss={loss:.4f} val_acc={acc:.4f}')
        run.log({'epoch': epoch + 1, 'train/loss': loss, 'validation/loss': validation_loss, 'validation/accuracy': acc})
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'fusion_checkpoint.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f'Early stopping after {epoch+1} epochs')
                break

    run.save('fusion_checkpoint.pth')
    run.summary['best_validation_loss'] = best_validation_loss
    run.finish()
    print('saved fusion_checkpoint.pth')


if __name__ == '__main__':
    main()
