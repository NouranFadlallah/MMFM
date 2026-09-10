import torch
import torch.nn as nn


def _adapt_first_conv(model, name, input_channels):
    """Replace a resnet's conv1 to accept input_channels, averaging pretrained weights."""
    if not name.startswith('resnet') or input_channels == 3:
        return
    first_layer = model.conv1
    if first_layer.in_channels == input_channels:
        return
    replacement = nn.Conv2d(
        input_channels, first_layer.out_channels, first_layer.kernel_size,
        first_layer.stride, first_layer.padding, bias=False
    )
    with torch.no_grad():
        replacement.weight.copy_(
            first_layer.weight.mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1)
        )
    model.conv1 = replacement


def _load_local_weights(model, path):
    """Best-effort load of a local state_dict checkpoint (e.g. RadImageNet weights)
    onto a freshly constructed torchvision model, before any head/channel surgery."""
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[create_backbone] loaded local weights from {path}: '
          f'{len(missing)} missing, {len(unexpected)} unexpected keys')


def create_backbone(name: str, pretrained: bool = True, remove_head: bool = True,
                     input_channels: int = 3, pretrained_weights: str = None):
    """Create a backbone feature extractor. Returns (module, feature_dim).

    Tries torchvision first, then timm. If neither is available, returns a small
    fallback CNN. The returned module should accept an image tensor and return
    either a [B, C, H, W] feature map or a [B, D] feature vector. The feature_dim
    describes the final channel / vector dimension that can be consumed by a
    Linear projection.

    pretrained_weights: optional path to a local state_dict checkpoint (e.g. a
    RadImageNet checkpoint) to load in place of ImageNet weights. When set,
    `pretrained` is ignored for the ImageNet-download path.
    """
    name = name.lower()

    tv_models = None
    timm = None
    try:
        from torchvision import models as tv_models
    except Exception:
        tv_models = None
    try:
        import timm
    except Exception:
        timm = None

    # Try torchvision models
    if tv_models is not None:
        if hasattr(tv_models, name):
            extra_kwargs = {'aux_logits': False} if name == 'googlenet' else {}
            m = getattr(tv_models, name)(pretrained=pretrained and not pretrained_weights, **extra_kwargs)
            if pretrained_weights:
                _load_local_weights(m, pretrained_weights)
            # infer feature dim and try to remove classification head
            if name.startswith('resnet'):
                feat_dim = m.fc.in_features
                if remove_head:
                    m.fc = nn.Identity()
            elif name.startswith('vgg'):
                feat_dim = m.classifier[-1].in_features
                if remove_head:
                    m.classifier = nn.Identity()
            elif name.startswith('densenet'):
                feat_dim = m.classifier.in_features
                if remove_head:
                    m.classifier = nn.Identity()
            elif name.startswith('efficientnet'):
                feat_dim = m.classifier[-1].in_features
                if remove_head:
                    m.classifier = nn.Identity()
            elif name == 'googlenet':
                feat_dim = m.fc.in_features
                if remove_head:
                    m.fc = nn.Identity()
            else:
                # best-effort: try to inspect common attributes
                feat_dim = getattr(m, 'num_features', None) or getattr(m, 'fc', None)
                if isinstance(feat_dim, nn.Module):
                    feat_dim = getattr(m.fc, 'in_features', 512)
                if feat_dim is None:
                    feat_dim = 512
            _adapt_first_conv(m, name, input_channels)
            return m, feat_dim

    # Try timm
    if timm is not None:
        try:
            # create_model with num_classes=0 often yields a feature vector
            m = timm.create_model(
                name, pretrained=pretrained and not pretrained_weights, num_classes=0,
                checkpoint_path=pretrained_weights or '',
            )
            feat_dim = getattr(m, 'num_features', None) or getattr(m, 'embed_dim', None) or 512
            return m, feat_dim
        except Exception:
            pass

    # Fallback simple CNN backbone
    class SimpleBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )

        def forward(self, x):
            x = self.features(x)
            return x

    m = SimpleBackbone()
    feat_dim = 64
    return m, feat_dim
