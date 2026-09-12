import pytest
import torch

from utils.metrics import accuracy, per_class_confusion


def test_accuracy_all_correct():
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    targets = torch.tensor([1, 0, 1])
    assert accuracy(preds, targets) == pytest.approx(1.0)


def test_accuracy_partial_match():
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.9, 0.1]])
    targets = torch.tensor([1, 0, 1])
    assert accuracy(preds, targets) == pytest.approx(2 / 3)


def test_per_class_confusion_detects_false_positive_lean():
    # 2 benign (0) misclassified as malignant, 0 malignant missed
    probs = torch.tensor([
        [0.2, 0.8], [0.3, 0.7],  # benign predicted malignant (2 false positives)
        [0.9, 0.1], [0.9, 0.1],  # benign predicted benign (2 true negatives)
        [0.1, 0.9], [0.1, 0.9],  # malignant predicted malignant (2 true positives)
    ])
    targets = torch.tensor([0, 0, 0, 0, 1, 1])
    result = per_class_confusion(probs, targets)
    assert result['confusion'] == {'tn': 2, 'fp': 2, 'fn': 0, 'tp': 2}
    assert result['bias'] == 'false_positive_leaning'
    assert result['false_positive_rate'] == pytest.approx(0.5)
    assert result['false_negative_rate'] == pytest.approx(0.0)
    assert result['per_class']['benign']['recall'] == pytest.approx(0.5)
    assert result['per_class']['malignant']['recall'] == pytest.approx(1.0)


def test_per_class_confusion_balanced_when_no_errors():
    probs = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
    targets = torch.tensor([0, 0, 1, 1])
    result = per_class_confusion(probs, targets)
    assert result['bias'] == 'balanced'
    assert result['confusion'] == {'tn': 2, 'fp': 0, 'fn': 0, 'tp': 2}
