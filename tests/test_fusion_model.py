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


def test_absent_samples_do_not_touch_branch_batchnorm():
    """A branch's backbone runs only on samples that carry that modality, so zero-filled
    placeholders never enter its BatchNorm statistics (combined batches are mostly absent
    from any one branch)."""
    torch.manual_seed(0)
    # resnet18 (untrained) because the fallback 'simple' backbone has no BatchNorm
    model = FusionLateModel(backbone_names=('resnet18', 'resnet18', 'resnet18'), pretrained=False, embedding_dim=8)
    real = torch.randn(4, 3, 32, 32) + 3.0
    batch = torch.cat([real, torch.zeros(12, 3, 32, 32)])
    presence = torch.zeros(16, 3, dtype=torch.bool)
    presence[:4, 1] = True  # only the first four samples are ultrasound
    presence[4:, 0] = True

    reference = FusionLateModel(backbone_names=('resnet18', 'resnet18', 'resnet18'), pretrained=False, embedding_dim=8)
    reference.load_state_dict(model.state_dict())
    model.train(), reference.train()
    model(batch, batch, batch, presence)
    reference(real, real, real, torch.tensor([[False, True, False]] * 4))

    for (name, a), (_, b) in zip(model.backbones[1].named_buffers(), reference.backbones[1].named_buffers()):
        assert torch.allclose(a, b), f'ultrasound-branch buffer {name} was changed by absent samples'


def test_single_present_branch_output_equals_its_logits():
    torch.manual_seed(0)
    model = FusionLateModel(backbone_names=('simple', 'simple', 'simple'), pretrained=False, embedding_dim=8).eval()
    x = torch.randn(6, 3, 32, 32)
    presence = torch.zeros(6, 3, dtype=torch.bool)
    presence[torch.arange(6), torch.tensor([0, 1, 2, 0, 1, 2])] = True
    fused, branch_logits, _ = model(x, x, x, presence)
    for row, branch in enumerate([0, 1, 2, 0, 1, 2]):
        assert torch.allclose(fused[row], branch_logits[branch][row], atol=1e-6)
        for other in {0, 1, 2} - {branch}:
            assert torch.all(branch_logits[other][row] == 0)


def test_gated_attention_weights_mask_absent_branches_and_depend_on_the_sample():
    torch.manual_seed(0)
    model = _make_model(fusion_mode="gated_attention").eval()
    x = torch.randn(4, 3, 32, 32)
    presence = torch.tensor([[True, True, True], [True, True, True], [True, False, True], [False, True, False]])
    fused, branch_logits, weights = model(x, x.flip(0), x * 2, presence)

    assert fused.shape == (4, 2)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)
    assert weights[2, 1] < 1e-6 and weights[3, 0] < 1e-6 and weights[3, 2] < 1e-6
    assert torch.isclose(weights[3, 1], torch.tensor(1.0))
    # unlike masked_scalar, two all-present samples with different images get different weights
    assert not torch.allclose(weights[0], weights[1])


def test_gated_attention_learns_only_from_multi_modality_samples():
    """With one modality per sample the attention scorer gets no gradient (softmax over a
    single branch is constant), so it can only be trained on multi-modality samples."""
    torch.manual_seed(0)
    x = torch.randn(4, 3, 32, 32)
    for presence, expect_gradient in ((torch.eye(3, dtype=torch.bool)[[0, 1, 2, 0]], False),
                                      (torch.ones(4, 3, dtype=torch.bool), True)):
        model = _make_model(fusion_mode="gated_attention")
        fused, _, _ = model(x, x, x, presence)
        torch.nn.functional.cross_entropy(fused, torch.tensor([0, 1, 0, 1])).backward()
        gradient = model.attention_w.weight.grad.abs().sum().item()
        assert (gradient > 1e-8) == expect_gradient


def test_aux_branch_loss_uses_only_present_samples():
    from training.train import _aux_branch_loss

    criterion = torch.nn.CrossEntropyLoss()
    y = torch.tensor([0, 1, 1])
    presence = torch.tensor([[True, False, False], [False, True, False], [False, True, False]])
    branch_logits = [torch.tensor([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                     torch.tensor([[9.0, -9.0], [0.0, 5.0], [0.0, 5.0]]),
                     torch.zeros(3, 2)]
    expected = (criterion(branch_logits[0][:1], y[:1]) + criterion(branch_logits[1][1:], y[1:])) / 2
    assert torch.isclose(_aux_branch_loss(branch_logits, y, presence, criterion), expected)
