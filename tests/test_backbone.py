import torch

from models.backbone import create_backbone


def test_resnet18_default_keeps_three_input_channels():
    model, feat_dim = create_backbone("resnet18", pretrained=False)
    assert feat_dim == 512
    assert model.conv1.in_channels == 3


def test_resnet18_adapts_first_conv_to_nine_channels():
    model, feat_dim = create_backbone("resnet18", pretrained=False, input_channels=9)
    assert feat_dim == 512
    assert model.conv1.in_channels == 9
    assert model.conv1.out_channels == 64
    # replacement weight is the pretrained-average repeated across every new channel
    assert torch.allclose(model.conv1.weight[:, 0], model.conv1.weight[:, 4])

    output = model(torch.randn(2, 9, 64, 64))
    assert output.shape[0] == 2


def test_resnet18_remove_head_replaces_fc_with_identity():
    model, _ = create_backbone("resnet18", pretrained=False, remove_head=True)
    output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 512)


def test_unknown_backbone_name_falls_back_to_simple_cnn():
    model, feat_dim = create_backbone("not-a-real-backbone", pretrained=False)
    assert feat_dim == 64
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 64, 1, 1)
