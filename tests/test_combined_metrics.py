import numpy as np
import pytest

from training.train import source_metrics


def test_source_metrics_splits_in_order_and_skips_single_class_auc():
    # mias (mammography): perfectly ranked; ispy2 (mri): malignant only; odelia (mri): inverted
    probs = np.array([0.1, 0.9, 0.8, 0.7, 0.9, 0.2])
    labels = np.array([0, 1, 1, 1, 0, 1])
    segments = [("mias", 0, 2), ("ispy2", 2, 2), ("odelia", 2, 2)]

    result = source_metrics(probs, labels, segments)

    assert result["sources"]["mias"] == {"n": 2, "accuracy": 1.0, "auc": 1.0}
    assert result["sources"]["ispy2"]["auc"] is None and result["sources"]["ispy2"]["accuracy"] == 1.0
    assert result["sources"]["odelia"]["auc"] == 0.0
    assert set(result["branches"]) == {"mammography", "mri"}
    # the malignant-only ispy2 pool is left out of the MRI branch AUC
    assert result["branches"]["mri"]["n"] == 2
    assert result["branches"]["mri"]["auc"] == 0.0
    assert result["mean_branch_auc"] == pytest.approx(0.5)


def test_source_metrics_without_any_auc():
    result = source_metrics(np.array([0.6, 0.7]), np.array([1, 1]), [("ispy2", 2, 2)])

    assert result["branches"] == {} and result["mean_branch_auc"] is None
