"""CycleGAN variants for ultrasound domain translation / enhancement, ported
natively from the thesis preprocessing notebook's three setups:

  cyclegan  junyanz/pytorch-CycleGAN-and-pix2pix defaults: ResNet-9 generator,
            70x70 PatchGAN, LSGAN, cycle 10, identity 0.5, lr 2e-4, image pool 50.
            Module layout matches that repo's, so its latest_net_G_A.pth loads.
  ncia      USenhance 2023 "NCIA" submission (gkdl3000/USenhance-submission):
            8-block generator whose bottleneck is conditioned on a pooled
            auxiliary feature, with a 50-way pseudo-label head (k-means
            clusters, refreshed every 10 epochs), VGG19 perceptual loss,
            LSGAN, gradient clipping.
  poas      USenhance 2023 "POAS" multi-task idea (YoonChiHo/USenhance23_POAS
            --is_mtl): a classifier on the generator bottleneck trained with
            each image's own label. In the original the classifier was never
            optimized or saved; here it is.

load_generator() returns an inference-ready generator from either an MMFM
checkpoint (scripts/train_cyclegan.py) or a raw generator state dict from the
original repos.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ('cyclegan', 'ncia', 'poas')


# ----------------------------------------------------------------- junyanz ResNet generator

class _JunyanzResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, bias=True), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, bias=True), nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """junyanz ResnetGenerator (ngf=64, 2 downsamplings, n_blocks residual
    blocks, instance norm, reflection padding, tanh), same `model.N` keys."""

    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=9):
        super().__init__()
        self.input_nc = input_nc
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7, bias=True), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        for i in range(2):
            mult = 2 ** i
            layers += [nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=True),
                       nn.InstanceNorm2d(ngf * mult * 2), nn.ReLU(True)]
        layers += [_JunyanzResnetBlock(ngf * 4) for _ in range(n_blocks)]
        for i in range(2):
            mult = 2 ** (2 - i)
            layers += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1, bias=True),
                       nn.InstanceNorm2d(ngf * mult // 2), nn.ReLU(True)]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)
        self.bottleneck_index = 10 + n_blocks - 1   # output of the last residual block

    def forward(self, x):
        return self.model(x)

    def encode(self, x):
        """Bottleneck features (after the residual blocks), for the POAS head."""
        return self.model[:self.bottleneck_index + 1](x)


# ----------------------------------------------------------------- NCIA generator

class _ZeroPadResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ZeroPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(inplace=True),
            nn.ZeroPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class _BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                                            nn.InstanceNorm2d(planes))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + identity)


class _FeatureExtractorWithInfo(nn.Module):
    def __init__(self, blocks_per_group, dim):
        super().__init__()
        for i in range(1, 5):
            setattr(self, f'model_body_block{i}', nn.Sequential(*[_ZeroPadResidualBlock(dim) for _ in range(blocks_per_group)]))

    def forward(self, x, info):
        for i in range(1, 5):
            x = getattr(self, f'model_body_block{i}')(x) + info
        return x


class NCIAGenerator(nn.Module):
    """USenhance NCIA generator: encoder -> 8 residual blocks, each group of 2
    offset by a pooled auxiliary feature -> decoder with additive skips;
    returns (image, 50-way auxiliary logits)."""

    def __init__(self, input_nc=1, output_nc=1, n_residual_blocks=8, num_classes=50):
        super().__init__()
        self.input_nc = input_nc
        self.model_head1 = nn.Sequential(nn.ZeroPad2d(3), nn.Conv2d(input_nc, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        self.model_head2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True))
        self.model_head3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.InstanceNorm2d(256), nn.ReLU(inplace=True))
        self.feature_extractor = _FeatureExtractorWithInfo(n_residual_blocks // 4, 256)
        self.model_body2 = nn.Sequential(*[_BasicBlock(256, 256, stride) for _ in range(3) for stride in (1, 2)])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.model_tail1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                                         nn.InstanceNorm2d(128), nn.ReLU(inplace=True))
        self.model_tail2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                         nn.InstanceNorm2d(64), nn.ReLU(inplace=True))
        self.model_tail3 = nn.Sequential(nn.ZeroPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh())

    def _info(self, sfeat):
        return self.avgpool(self.model_body2(sfeat))

    def forward(self, x, return_aux=False):
        x1 = self.model_head1(x)
        x2 = self.model_head2(x1)
        sfeat = self.model_head3(x2)
        info = self._info(sfeat)
        aux = self.fc(torch.flatten(info, 1))
        feat = self.feature_extractor(sfeat, info.expand_as(sfeat))
        out = self.model_tail3(self.model_tail2(self.model_tail1(feat) + x2) + x1)
        return (out, aux) if return_aux else out

    @torch.no_grad()
    def info_feature(self, x):
        """Pooled 256-d auxiliary feature that the k-means pseudo-labels cluster."""
        return torch.flatten(self._info(self.model_head3(self.model_head2(self.model_head1(x)))), 1)


# ----------------------------------------------------------------- discriminator, losses

class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN (junyanz NLayerDiscriminator, n_layers=3; NCIA uses the same shape)."""

    def __init__(self, input_nc=1, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers + 1):
            previous, mult = mult, min(2 ** n, 8)
            stride = 2 if n < n_layers else 1
            layers += [nn.Conv2d(ndf * previous, ndf * mult, 4, stride=stride, padding=1, bias=True),
                       nn.InstanceNorm2d(ndf * mult), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(ndf * mult, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VGGPerceptualLoss(nn.Module):
    """NCIA's PerceptualLoss: frozen ImageNet VGG19 features cut after conv
    layers 2, 7, 16, 25, 34 (pre-ReLU), weighted 0.1, 0.1, 1, 1, 1, L1.
    Grayscale inputs are repeated to 3 channels; like the NCIA trainer
    (use_input_norm=False), inputs stay in [-1, 1]."""

    def __init__(self, layers=(2, 7, 16, 25, 34), weights=(0.1, 0.1, 1.0, 1.0, 1.0)):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        bounds = (-1,) + tuple(layers)
        self.slices = nn.ModuleList(nn.Sequential(*features[bounds[i] + 1:bounds[i + 1] + 1]) for i in range(len(layers)))
        self.weights = weights
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x, target):
        x, target = x.expand(-1, 3, -1, -1), target.detach().expand(-1, 3, -1, -1)
        loss = 0.0
        for weight, block in zip(self.weights, self.slices):
            x, target = block(x), block(target)
            loss = loss + weight * F.l1_loss(x, target)
        return loss


class ImagePool:
    """History of generated images for the discriminator (Shrivastava et al.):
    half the time return a stored fake and keep the new one instead."""

    def __init__(self, size=50):
        self.size = size
        self.images = []

    def query(self, images):
        if self.size == 0:
            return images
        out = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(image)
                out.append(image)
            elif random.random() > 0.5:
                index = random.randrange(self.size)
                out.append(self.images[index].clone())
                self.images[index] = image
            else:
                out.append(image)
        return torch.cat(out)


class BottleneckClassifier(nn.Module):
    """POAS multi-task head: flattened generator bottleneck -> class logits."""

    def __init__(self, bottleneck_shape, num_classes):
        super().__init__()
        channels, height, width = bottleneck_shape
        self.fc = nn.Linear(channels * height * width, num_classes)

    def forward(self, features):
        return self.fc(torch.flatten(features, 1))


# ----------------------------------------------------------------- construction / loading

def build_generator(variant, input_nc=1, output_nc=1):
    if variant == 'ncia':
        return NCIAGenerator(input_nc, output_nc)
    return ResnetGenerator(input_nc, output_nc)


def _strip_module_prefix(state_dict):
    return {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}


def load_generator(checkpoint, direction='A2B', map_location='cpu'):
    """(generator in eval mode, variant) from an MMFM checkpoint
    ({'variant', 'input_nc', 'G_A2B', 'G_B2A'}) or a bare generator state dict
    saved by the junyanz or NCIA repos (variant inferred from its keys)."""
    state = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and 'variant' in state:
        variant, input_nc = state['variant'], state.get('input_nc', 1)
        weights = state[f'G_{direction}']
    else:
        weights = _strip_module_prefix(state)
        variant = 'ncia' if any(k.startswith('model_head1') for k in weights) else 'cyclegan'
        first_conv = 'model_head1.1.weight' if variant == 'ncia' else 'model.1.weight'
        input_nc = weights[first_conv].shape[1]
    output_key = 'model_tail3.1.weight' if variant == 'ncia' else max(
        (k for k in weights if k.startswith('model.') and k.endswith('.weight')), key=lambda k: int(k.split('.')[1]))
    generator = build_generator(variant, input_nc, weights[output_key].shape[0])
    generator.load_state_dict(weights, strict=True)
    return generator.eval().to(map_location), variant
