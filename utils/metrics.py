import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score,
)


def accuracy(preds, targets):
    return (preds.argmax(dim=1) == targets).float().mean().item()


def classification_metrics(probs, targets, positive_class=1):
    """Full binary metric set (accuracy, sensitivity, specificity, precision, F1,
    AUC), matching the indices reported in the BUS-BRA paper (Gomez-Flores et al.,
    2024, Table 3/4). `probs` is [N, num_classes] softmax output; `targets` is [N].
    """
    probs = probs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    preds = probs.argmax(axis=1)

    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) else float('nan')

    metrics = {
        'accuracy': accuracy_score(targets, preds),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision_score(targets, preds, pos_label=positive_class, zero_division=0),
        'f1': f1_score(targets, preds, pos_label=positive_class, zero_division=0),
    }
    if len(set(targets.tolist())) > 1:
        metrics['auc'] = roc_auc_score(targets, probs[:, positive_class])
    else:
        metrics['auc'] = float('nan')
    return metrics
