"""Write per-sample test predictions for the existing BUS-BRA official-fold
checkpoints (runs/warm/busbra_fold{k}.pth: --paper-match, ResNet-18, K5B folds),
in the same NPZ format training/train.py writes for k-fold specialists, so
scripts/compare_fusion_specialists.py can use them without retraining."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg  # noqa: E402
from data.dataset import BusbraTransform  # noqa: E402
from training.train import (SingleBackboneClassifier, _busbra_kfold_frame, _make_dataset,  # noqa: E402
                            _make_loader, collect_predictions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/warm/busbra_fold{fold}.pth')
    parser.add_argument('--out', default='runs/specialists/busbra_fold{fold}_test_predictions.npz')
    parser.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold in args.folds:
        _, _, test_frame = _busbra_kfold_frame(Path('datasets/ultrasound/busbra/BUSBRA'), 5, fold, args.seed,
                                               holdout_fraction=0.2)
        dataset = _make_dataset(test_frame, BusbraTransform(size=224, augment=False, contrast_stretch=True))
        model = SingleBackboneClassifier(backbone_name='resnet18', pretrained=False,
                                         embedding_dim=cfg.embedding_dim, num_classes=cfg.num_classes)
        model.load_state_dict(torch.load(args.checkpoint.format(fold=fold), map_location='cpu'))
        probs, targets = collect_predictions(model.to(device), _make_loader(dataset, 32, 4), device)
        out = Path(args.out.format(fold=fold))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, probs=probs[:, 1].numpy(), labels=targets.numpy(), image=np.array(dataset.df['img1'].tolist()))
        from sklearn.metrics import roc_auc_score
        print(f'fold {fold}: n={len(targets)} auc={roc_auc_score(targets.numpy(), probs[:, 1].numpy()):.4f} -> {out}')


if __name__ == '__main__':
    main()
