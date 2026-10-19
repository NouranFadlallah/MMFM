"""AMBL benign-vs-malignant baseline: ResNet-18 on the 9-channel lesion crops
against a lesion-size-only baseline, with 5-fold patient-level cross validation.

The official 70/15/15 split leaves 18 test lesions (13 patients), where one
lesion moves the AUC by about 0.05, so every patient is instead tested once
across five stratified group folds. Per fold: one fold is test, the next is
validation (checkpoint = best validation lesion AUC), the rest train.

  slice score    softmax malignant probability of one crop
  lesion score   mean slice score over the lesion's crops
  patient score  most suspicious lesion (max), patient malignant if any lesion is
  size baseline  lesion volume in ml from the audit (analysis/ambl/lesions_resolved.csv)

Reports per-fold and pooled out-of-fold AUCs with 1,000-sample bootstrap 95%
confidence intervals, and a paired bootstrap of the model-minus-size AUC.

Usage:
    python3 scripts/ambl_baseline.py              # 5 folds, 30 epochs each
    python3 scripts/ambl_baseline.py --folds 1 --epochs 2 --no-wandb   # smoke test
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import BreadmTransform  # noqa: E402
from training.train import SingleBackboneClassifier  # noqa: E402

CROPS = Path('datasets/MRI/AMBL/cls/img9Se')
RESOLVED = Path('analysis/ambl/lesions_resolved.csv')


def crop_frame(root=CROPS):
    rows = []
    for path in sorted(root.glob('*/*/*/p-*.npy')):
        lesion = path.parent.name                      # <patient>_lesion<k>
        patient, segment = lesion.rsplit('_lesion', 1)
        rows.append({'path': str(path), 'label': int(path.parents[1].name == 'Malignant'),
                     'lesion': lesion, 'patient': patient, 'segment': int(segment)})
    frame = pd.DataFrame(rows)
    sizes = pd.read_csv(RESOLVED)[['group', 'segment', 'volume_ml']].rename(columns={'group': 'patient'})
    return frame.merge(sizes, on=['patient', 'segment'], how='left')


def patient_folds(frame, n_folds, seed):
    """Fold per patient, stratified on whether the patient has a malignant lesion."""
    rng = np.random.default_rng(seed)
    patients = frame.groupby('patient')['label'].max()
    fold_of = {}
    for _, group in patients.groupby(patients):
        ids = group.index.to_numpy(copy=True)
        rng.shuffle(ids)
        for k, patient in enumerate(ids):
            fold_of[patient] = k % n_folds
    return frame['patient'].map(fold_of)


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, frame, augment):
        self.frame = frame.reset_index(drop=True)
        channels = np.load(self.frame.iloc[0]['path'], mmap_mode='r').shape[-1]
        self.transform = BreadmTransform(size=224, augment=augment, channels=channels)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return self.transform(np.load(row['path'])), int(row['label'])


@torch.no_grad()
def slice_scores(model, frame, device):
    model.eval()
    loader = torch.utils.data.DataLoader(CropDataset(frame, augment=False), batch_size=64, num_workers=2)
    scores = []
    for images, _ in loader:
        logits = model(images.to(device))[0]
        scores.append(F.softmax(logits.float(), dim=1)[:, 1].cpu())
    return frame.assign(score=torch.cat(scores).numpy())


def lesion_table(scored):
    return scored.groupby('lesion').agg(label=('label', 'first'), patient=('patient', 'first'),
                                        score=('score', 'mean'), volume_ml=('volume_ml', 'first')).reset_index()


def auc(labels, scores):
    from sklearn.metrics import roc_auc_score

    labels, scores = np.asarray(labels), np.asarray(scores)
    return float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else float('nan')


def patient_table(lesions, column):
    return lesions.groupby('patient').agg(label=('label', 'max'), score=(column, 'max')).reset_index()


def bootstrap(lesions, n=1000, seed=0):
    """95% CIs for model AUC, size AUC and their difference, resampling patients."""
    rng = np.random.default_rng(seed)
    patients = lesions['patient'].unique()
    by_patient = {p: g for p, g in lesions.groupby('patient')}
    model, size, diff = [], [], []
    for _ in range(n):
        sample = pd.concat([by_patient[p] for p in rng.choice(patients, len(patients), replace=True)])
        a, b = auc(sample['label'], sample['score']), auc(sample['label'], sample['volume_ml'])
        if not (np.isnan(a) or np.isnan(b)):
            model.append(a), size.append(b), diff.append(a - b)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {'model_auc_ci': ci(model), 'size_auc_ci': ci(size), 'difference_ci': ci(diff),
            'p_model_not_better': float(np.mean(np.array(diff) <= 0))}


def train_fold(train, val, args, device, run, fold):
    torch.manual_seed(args.seed + fold)
    random.seed(args.seed + fold)
    channels = np.load(train.iloc[0]['path'], mmap_mode='r').shape[-1]   # 9, or 10 with a mask channel
    model = SingleBackboneClassifier('resnet18', pretrained=True, embedding_dim=512, num_classes=2,
                                     input_channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = torch.utils.data.DataLoader(CropDataset(train, augment=True), batch_size=args.batch_size,
                                         shuffle=True, num_workers=4, drop_last=True)
    best, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, labels in loader:
            loss = F.cross_entropy(model(images.to(device))[0], labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        val_lesions = lesion_table(slice_scores(model, val, device))
        val_auc = auc(val_lesions['label'], val_lesions['score'])
        if run:
            run.log({f'fold{fold}/train_loss': total / len(loader.dataset), f'fold{fold}/val_lesion_auc': val_auc})
        if val_auc > best:
            best, best_state = val_auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--folds', type=int, default=None, help='run only the first N folds (smoke test)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('runs/run-4/ambl_cv_results.json'))
    parser.add_argument('--crops', type=Path, default=CROPS, help='img9Se root (e.g. the 64 mm window crops)')
    parser.add_argument('--run-name', default='ambl_resnet18_cv5')
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    frame = crop_frame(args.crops)
    frame['fold'] = patient_folds(frame, args.n_folds, args.seed)
    print(f'{frame.lesion.nunique()} lesions, {frame.patient.nunique()} patients, {len(frame)} crops, '
          f'{args.n_folds} patient-level folds')
    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project='mmfm-ambl', name=args.run_name, config=vars(args))

    per_fold, out_of_fold = [], []
    for fold in range(args.folds or args.n_folds):
        val_fold = (fold + 1) % args.n_folds
        test, val = frame[frame.fold == fold], frame[frame.fold == val_fold]
        train = frame[~frame.fold.isin([fold, val_fold])]
        model, best_val = train_fold(train, val, args, device, run, fold)
        lesions = lesion_table(slice_scores(model, test, device)).assign(fold=fold)
        result = {
            'fold': fold, 'test_lesions': len(lesions), 'test_patients': lesions.patient.nunique(),
            'best_val_lesion_auc': best_val,
            'lesion_auc': auc(lesions['label'], lesions['score']),
            'size_lesion_auc': auc(lesions['label'], lesions['volume_ml']),
            'patient_auc': auc(*patient_table(lesions, 'score')[['label', 'score']].T.values),
            'size_patient_auc': auc(*patient_table(lesions, 'volume_ml')[['label', 'score']].T.values),
        }
        per_fold.append(result)
        out_of_fold.append(lesions)
        print(' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in result.items()), flush=True)

    pooled = pd.concat(out_of_fold)
    sens_spec = lambda labels, predicted: (float(predicted[labels == 1].mean()), float((~predicted[labels == 0]).mean()))  # noqa: E731
    sensitivity, specificity = sens_spec(pooled['label'].values, pooled['score'].values >= 0.5)
    summary = {
        'lesions': int(len(pooled)), 'patients': int(pooled.patient.nunique()),
        'pooled_lesion_auc': auc(pooled['label'], pooled['score']),
        'pooled_size_lesion_auc': auc(pooled['label'], pooled['volume_ml']),
        'pooled_patient_auc': auc(*patient_table(pooled, 'score')[['label', 'score']].T.values),
        'pooled_size_patient_auc': auc(*patient_table(pooled, 'volume_ml')[['label', 'score']].T.values),
        'lesion_sensitivity_at_0.5': sensitivity, 'lesion_specificity_at_0.5': specificity,
        **{f'mean_{k}': float(np.nanmean([r[k] for r in per_fold])) for k in ('lesion_auc', 'size_lesion_auc', 'patient_auc')},
        **{f'std_{k}': float(np.nanstd([r[k] for r in per_fold], ddof=1)) if len(per_fold) > 1 else float('nan')
           for k in ('lesion_auc', 'size_lesion_auc', 'patient_auc')},
        **bootstrap(pooled),
    }
    print(json.dumps(summary, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'config': {k: str(v) for k, v in vars(args).items()}, 'per_fold': per_fold,
                                    'summary': summary}, indent=2))
    pooled.to_csv(args.out.with_name(args.out.stem + '_lesions.csv'), index=False)
    if run:
        run.summary.update(summary)
        run.finish()


if __name__ == '__main__':
    main()
