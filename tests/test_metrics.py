import pytest
import torch

from utils.metrics import accuracy


def test_accuracy_all_correct():
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    targets = torch.tensor([1, 0, 1])
    assert accuracy(preds, targets) == pytest.approx(1.0)


def test_accuracy_partial_match():
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.9, 0.1]])
    targets = torch.tensor([1, 0, 1])
    assert accuracy(preds, targets) == pytest.approx(2 / 3)
