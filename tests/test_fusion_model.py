import pytest
import torch

from models.fusion_model import FusionLateModel


def _make_model(**kwargs):
    defaults = dict(
        backbone_names=("simple", "simple", "simple"),
        pretrained=False,
        embedding_dim=8,
        num_classes=2,
    )
    defaults.update(kwargs)
    return FusionLateModel(**defaults)


def test_forward_all_present_weights_sum_to_one():
    model = _make_model()
    x1 = torch.randn(4, 3, 32, 32)
    x2 = torch.randn(4, 3, 32, 32)
    x3 = torch.randn(4, 3, 32, 32)
    presence = torch.ones(4, 3, dtype=torch.bool)

    fused, branch_logits, weights = model(x1, x2, x3, presence)

    assert fused.shape == (4, 2)
    assert len(branch_logits) == 3
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)
    # logit_scalars start at zero, so an all-present batch gets uniform weights
    assert torch.allclose(weights, torch.full_like(weights, 1 / 3), atol=1e-4)


def test_forward_masks_out_absent_branch_via_presence_mask():
    model = _make_model()
    x1 = torch.randn(2, 3, 32, 32)
    x2 = torch.randn(2, 3, 32, 32)
    x3 = torch.randn(2, 3, 32, 32)
    presence = torch.tensor([[True, False, True], [True, False, True]])

    _, _, weights = model(x1, x2, x3, presence)

    assert torch.allclose(weights[:, 1], torch.zeros(2), atol=1e-6)
    assert torch.allclose(weights[:, 0], weights[:, 2], atol=1e-4)


def test_forward_with_none_branch_and_no_presence_mask_does_not_crash():
    # Regression test: forward() used to raise AttributeError here because it
    # read presence_mask.shape[0] before a default mask had been created.
    model = _make_model()
    x1 = torch.randn(3, 3, 32, 32)
    x3 = torch.randn(3, 3, 32, 32)

    fused, branch_logits, weights = model(x1, None, x3)

    assert fused.shape == (3, 2)
    assert torch.allclose(weights[:, 1], torch.zeros(3), atol=1e-6)
    assert torch.allclose(weights[:, 0], weights[:, 2], atol=1e-4)


def test_gating_fusion_mode_not_implemented():
    model = _make_model(fusion_mode="gating")
    x1 = torch.randn(1, 3, 32, 32)
    x2 = torch.randn(1, 3, 32, 32)
    x3 = torch.randn(1, 3, 32, 32)

    with pytest.raises(NotImplementedError):
        model(x1, x2, x3, torch.ones(1, 3, dtype=torch.bool))


def test_per_branch_input_channels_are_respected():
    model = _make_model(
        backbone_names=("resnet18", "resnet18", "resnet18"),
        input_channels=(3, 3, 9),
    )
    x1 = torch.randn(2, 3, 64, 64)
    x2 = torch.randn(2, 3, 64, 64)
    x3 = torch.randn(2, 9, 64, 64)
    presence = torch.ones(2, 3, dtype=torch.bool)

    fused, _, _ = model(x1, x2, x3, presence)

    assert fused.shape == (2, 2)
    assert model.backbones[2].conv1.in_channels == 9
