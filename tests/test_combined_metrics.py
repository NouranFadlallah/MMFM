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


def test_freeze_backbones_keeps_weights_and_batchnorm_stats_but_trains_heads():
    import copy

    import torch

    from models.fusion_model import FusionLateModel
    from training.train import _freeze_backbones, train_one_epoch

    model = FusionLateModel(backbone_names=("resnet18",) * 3, pretrained=False, embedding_dim=8,
                            num_classes=2, input_channels=(3, 3, 9))
    _freeze_backbones(model)
    before = copy.deepcopy(model.state_dict())
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.randn(4, 3, 32, 32), torch.zeros(4, 3, 32, 32), torch.zeros(4, 9, 32, 32),
        torch.tensor([0, 1, 0, 1]), torch.tensor([[True, False, False]] * 4)), batch_size=4)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)

    train_one_epoch(model, loader, opt, "cpu", criterion=torch.nn.CrossEntropyLoss())

    after = model.state_dict()
    backbone_keys = [k for k in before if k.startswith("backbones.")]
    assert any("running_mean" in k for k in backbone_keys)
    assert all(torch.equal(before[k], after[k]) for k in backbone_keys)
    assert not torch.equal(before["projections.0.0.weight"], after["projections.0.0.weight"])
