import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
    precision_score, roc_auc_score,
)

CLASS_NAMES = ('benign', 'malignant')


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


def per_class_confusion(probs, targets):
    """Confusion-matrix counts, per-class (benign/malignant) precision,
    recall, and F1, plus a false-positive-vs-false-negative bias verdict.

    False positive here means predicting malignant on an actually-benign
    case (over-diagnosis); false negative means missing an actually-
    malignant case (under-diagnosis) — clinically the costlier direction.
    `probs` is [N, num_classes] softmax output; `targets` is [N].
    """
    probs = probs.detach().cpu().numpy() if torch.is_tensor(probs) else np.asarray(probs)
    targets = targets.detach().cpu().numpy() if torch.is_tensor(targets) else np.asarray(targets)
    preds = probs.argmax(axis=1)

    tn, fp, fn, tp = (int(x) for x in confusion_matrix(targets, preds, labels=[0, 1]).ravel())
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        targets, preds, labels=[0, 1], zero_division=0
    )

    fp_rate = fp / (fp + tn) if (fp + tn) else float('nan')
    fn_rate = fn / (fn + tp) if (fn + tp) else float('nan')
    if np.isnan(fp_rate) or np.isnan(fn_rate) or fp_rate == fn_rate:
        bias = 'balanced'
    elif fp_rate > fn_rate:
        bias = 'false_positive_leaning'
    else:
        bias = 'false_negative_leaning'

    return {
        'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'per_class': {
            name: {
                'precision': float(precisions[i]),
                'recall': float(recalls[i]),
                'f1': float(f1s[i]),
                'support': int(supports[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        'false_positive_rate': float(fp_rate),
        'false_negative_rate': float(fn_rate),
        'bias': bias,
    }
