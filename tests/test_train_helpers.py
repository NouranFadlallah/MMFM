from pathlib import Path

import pandas as pd
import torch

from data.dataset import TripleImageDataset
from models.fusion_model import FusionLateModel
from training.train import SingleBackboneClassifier, _make_dataset, _warm_start_branch


def test_make_dataset_replicates_image_across_all_three_branches():
    frame = pd.DataFrame(
        {
            "image": [Path("/a/one.png"), Path("/a/two.png")],
            "label": [0, 1],
        }
    )

    dataset = _make_dataset(frame, transform=None)

    assert isinstance(dataset, TripleImageDataset)
    assert list(dataset.df["img1"]) == list(dataset.df["img2"]) == list(dataset.df["img3"])
    assert list(dataset.df["img1"]) == ["/a/one.png", "/a/two.png"]
    assert list(dataset.df["label"]) == [0, 1]


def test_make_dataset_carries_mask_and_crop_box_columns_when_present():
    frame = pd.DataFrame(
        {
            "image": [Path("/a/one.png")],
            "mask": [Path("/a/one_mask.png")],
            "crop_box": [(0, 0, 10, 10)],
            "label": [1],
        }
    )

    dataset = _make_dataset(frame, transform=None)

    assert dataset.df.loc[0, "mask"] == "/a/one_mask.png"
    assert dataset.df.loc[0, "crop_box"] == (0, 0, 10, 10)


def test_single_backbone_classifier_adapts_to_nine_input_channels():
    model = SingleBackboneClassifier(
        backbone_name="resnet18", pretrained=False, embedding_dim=8, num_classes=2, input_channels=9
    )

    assert model.backbone.conv1.in_channels == 9
    logits, branch_logits, weights = model(torch.randn(2, 9, 32, 32))
    assert logits.shape == (2, 2)
    assert branch_logits is None and weights is None


def test_warm_start_branch_copies_matching_weights_into_fusion_model(tmp_path):
    source = SingleBackboneClassifier(
        backbone_name="resnet18", pretrained=False, embedding_dim=8, num_classes=2, input_channels=3
    )
    checkpoint_path = tmp_path / "single_best.pth"
    torch.save(source.state_dict(), checkpoint_path)

    fusion_model = FusionLateModel(
        backbone_names=("resnet18", "resnet18", "resnet18"),
        pretrained=False,
        embedding_dim=8,
        num_classes=2,
        input_channels=(3, 3, 9),
    )

    _warm_start_branch(fusion_model, 0, checkpoint_path)

    assert torch.equal(fusion_model.backbones[0].conv1.weight, source.backbone.conv1.weight)
    assert torch.equal(fusion_model.classifiers[0].weight, source.classifier.weight)


def test_warm_start_branch_skips_on_channel_mismatch_without_raising(tmp_path, capsys):
    source = SingleBackboneClassifier(
        backbone_name="resnet18", pretrained=False, embedding_dim=8, num_classes=2, input_channels=3
    )
    checkpoint_path = tmp_path / "single_best.pth"
    torch.save(source.state_dict(), checkpoint_path)

    fusion_model = FusionLateModel(
        backbone_names=("resnet18", "resnet18", "resnet18"),
        pretrained=False,
        embedding_dim=8,
        num_classes=2,
        input_channels=(3, 3, 9),
    )
    original_conv1 = fusion_model.backbones[2].conv1.weight.clone()

    # branch 2 expects 9 input channels; the checkpoint's backbone was trained for 3.
    _warm_start_branch(fusion_model, 2, checkpoint_path)

    assert torch.equal(fusion_model.backbones[2].conv1.weight, original_conv1)
    assert "skipping warm-start" in capsys.readouterr().out


def test_warm_start_branch_is_a_no_op_when_no_checkpoint_given():
    fusion_model = FusionLateModel(
        backbone_names=("simple", "simple", "simple"), pretrained=False, embedding_dim=8, num_classes=2
    )
    original = fusion_model.backbones[0].features[0].weight.clone()

    _warm_start_branch(fusion_model, 0, None)

    assert torch.equal(fusion_model.backbones[0].features[0].weight, original)
