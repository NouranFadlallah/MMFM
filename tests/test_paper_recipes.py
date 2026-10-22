from argparse import Namespace

import pytest
import torch
import torch.optim as optim

from training.train import (
    PAPER_RECIPES, SingleBackboneClassifier, _apply_paper_recipe, _epoch_stage, _make_optimizer,
    _set_trainable_scope, _use_class_weights,
)


def _args(dataset, paper_match=True, **overrides):
    defaults = dict(dataset=dataset, paper_match=paper_match, backbone="resnet18", pretrained=False,
                    image_size=224, epochs=10, batch_size=8)
    return Namespace(**{**defaults, **overrides})


def test_datasets_without_their_own_recipe_keep_the_busbra_sgd_recipe():
    args = _args("busi", epochs=100, batch_size=32)
    _apply_paper_recipe(args)

    assert (args.backbone, args.epochs, args.batch_size) == ("resnet18", 100, 32)
    assert _use_class_weights(args)
    optimizer = _make_optimizer(PAPER_RECIPES["default"], [torch_param()], 1e-3)
    assert isinstance(optimizer, optim.SGD) and optimizer.defaults["momentum"] == 0.9


def test_bus_cot_recipe_overrides_cli_args_to_the_paper_values():
    args = _args("bus_cot")
    _apply_paper_recipe(args)

    assert (args.backbone, args.pretrained, args.epochs, args.batch_size) == ("resnet50", True, 5, 16)
    assert not _use_class_weights(args)
    optimizer = _make_optimizer(PAPER_RECIPES["bus_cot"], [torch_param()], 5e-5)
    assert isinstance(optimizer, optim.AdamW) and optimizer.defaults["weight_decay"] == 0.1


def test_recipe_is_ignored_without_paper_match():
    args = _args("bus_cot", paper_match=False)
    _apply_paper_recipe(args)

    assert (args.backbone, args.epochs) == ("resnet18", 10)
    assert not _use_class_weights(args)


def test_cbis_ddsm_stages_sum_to_shen_200_epochs():
    args = _args("cbis_ddsm")
    _apply_paper_recipe(args)
    stages = PAPER_RECIPES["cbis_ddsm"]["stages"]

    assert args.epochs == 200
    assert _epoch_stage(stages, 0) == (1e-3, "head")
    assert _epoch_stage(stages, 3) == (1e-4, "top")
    assert _epoch_stage(stages, 12) == (1e-4, "top")
    assert _epoch_stage(stages, 13) == (1e-5, "all")
    assert _epoch_stage(stages, 199) == (1e-5, "all")


@pytest.mark.parametrize("scope", ["head", "top", "all"])
def test_trainable_scope_freezes_the_right_modules(scope):
    model = SingleBackboneClassifier(backbone_name="resnet18", pretrained=False, embedding_dim=8, num_classes=2)
    _set_trainable_scope(model, scope)

    def trainable(module):
        return all(p.requires_grad for p in module.parameters())

    assert trainable(model.classifier) and trainable(model.projection)
    assert trainable(model.backbone.layer4) == (scope in ("top", "all"))
    assert trainable(model.backbone.conv1) == (scope == "all")


def torch_param():
    return torch.nn.Parameter(torch.zeros(1))
