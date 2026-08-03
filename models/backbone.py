import torch
import torch.nn as nn


def create_backbone(name: str, pretrained: bool = True, remove_head: bool = True):
    """Create a backbone feature extractor. Returns (module, feature_dim).

    Tries torchvision first, then timm. If neither is available, returns a small
    fallback CNN. The returned module should accept an image tensor and return
    either a [B, C, H, W] feature map or a [B, D] feature vector. The feature_dim
    describes the final channel / vector dimension that can be consumed by a
    Linear projection.
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
            m = getattr(tv_models, name)(pretrained=pretrained)
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
            else:
                # best-effort: try to inspect common attributes
                feat_dim = getattr(m, 'num_features', None) or getattr(m, 'fc', None)
                if isinstance(feat_dim, nn.Module):
                    feat_dim = getattr(m.fc, 'in_features', 512)
                if feat_dim is None:
                    feat_dim = 512
            return m, feat_dim

    # Try timm
    if timm is not None:
        try:
            # create_model with num_classes=0 often yields a feature vector
            m = timm.create_model(name, pretrained=pretrained, num_classes=0)
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
