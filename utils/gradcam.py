"""Grad-CAM for the single-modality SingleBackboneClassifier models in this
repo. Works for any resnet*/efficientnet* backbone create_backbone() can
build, by hooking the last spatial (pre-pooling) feature map instead of
relying on the backbone's own forward (which already pools internally once
its classification head is removed).
"""
import numpy as np
import torch
import torch.nn.functional as F


def target_layer_for_raw_backbone(backbone_module, backbone_name):
    """Return the last conv-stage module to hook for Grad-CAM, given the raw
    backbone module create_backbone() built (no SingleBackboneClassifier/
    FusionLateModel wrapping) and the backbone name used to build it. Shared
    by target_layer_for_backbone (single-modality models, model.backbone)
    and FusionLateModel's per-branch model.backbones[i] (fusion models)."""
    name = backbone_name.lower()
    if name.startswith('resnet'):
        return backbone_module.layer4
    if name.startswith('efficientnet'):
        return backbone_module.features
    if name.startswith('densenet'):
        return backbone_module.features
    if name.startswith('vgg'):
        return backbone_module.features
    raise ValueError(f'No known Grad-CAM target layer for backbone {backbone_name!r}')


def target_layer_for_backbone(model, backbone_name):
    """Return the last conv-stage module to hook for Grad-CAM, given a
    SingleBackboneClassifier and the backbone name used to build it."""
    return target_layer_for_raw_backbone(model.backbone, backbone_name)


class GradCAM:
    """Standard Grad-CAM (Selvaraju et al., 2017): weight each channel of the
    target layer's activation map by the global-average-pooled gradient of
    the chosen class score, ReLU the weighted sum, then upsample to the
    input resolution.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x1, x2=None, x3=None, presence_mask=None, target_class=None):
        """x1 is the single real-modality input, shape [B, C, H, W]. Returns
        (cam [B, H, W] in [0, 1], predicted/used class [B], softmax probs [B, num_classes])."""
        self.model.zero_grad(set_to_none=True)
        logits, _, _ = self.model(x1, x2, x3, presence_mask)
        probs = torch.softmax(logits, dim=1).detach()
        if target_class is None:
            target_class = logits.argmax(dim=1)
        score = logits.gather(1, target_class.view(-1, 1)).squeeze(1)
        score.sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))
        cam = F.interpolate(
            cam.unsqueeze(1), size=x1.shape[-2:], mode='bilinear', align_corners=False
        ).squeeze(1)
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach().cpu().numpy(), target_class.detach().cpu().numpy(), probs.cpu().numpy()


class ViTGradCAM:
    """Grad-CAM adapted for a ViT block whose output is a token sequence
    [B, N, D] (e.g. a timm/open_clip ViT block, or a SigLIP encoder layer)
    rather than a CNN's spatial [B, C, H, W] feature map. Drops
    `num_prefix_tokens` (1 for a CLS token, 0 for towers with none) before
    reshaping the remaining N patch tokens into `grid_size` = (H, W), then
    treats D as the Grad-CAM channel dimension exactly like GradCAM above.

    Unlike GradCAM, this doesn't own the forward/backward call, since the two
    models it's used for here (BiomedCLIPClassifier, a plain classifier; and
    MedGemma, a generative chat model scored at the answer token) have very
    different calling conventions. The caller runs its own forward pass,
    picks a scalar `score` per example, and passes it to `compute()`.
    """

    def __init__(self, target_block, num_prefix_tokens, grid_size):
        self.activations = None
        self.gradients = None
        self.num_prefix_tokens = num_prefix_tokens
        self.grid_size = grid_size
        target_block.register_forward_hook(self._save_activation)
        target_block.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _to_grid(self, tokens):
        tokens = tokens[:, self.num_prefix_tokens:, :]
        b, n, d = tokens.shape
        h, w = self.grid_size
        if n != h * w:
            raise ValueError(f'{n} patch tokens after dropping prefix != grid {h}x{w}')
        return tokens.permute(0, 2, 1).reshape(b, d, h, w)

    def compute(self, score, image_hw):
        """score: a [B] tensor already selected for the target class/token
        (e.g. a classifier logit, or a generated token's logit), with
        .backward() not yet called. image_hw: (H, W) to upsample the CAM to.
        Returns a [B, H, W] numpy array in [0, 1]."""
        score.sum().backward()
        activations = self._to_grid(self.activations)
        gradients = self._to_grid(self.gradients)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations.detach()).sum(dim=1))
        cam = F.interpolate(cam.unsqueeze(1), size=image_hw, mode='bilinear', align_corners=False).squeeze(1)
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach().cpu().numpy()


def overlay_heatmap(image_chw, cam_hw, alpha=0.45):
    """image_chw: float tensor/array in [0,1], shape [3,H,W] or [1,H,W].
    cam_hw: [H,W] in [0,1]. Returns a uint8 RGB array [H,W,3] with a jet-like
    heatmap overlaid, using only numpy (no cv2/matplotlib dependency)."""
    image = np.asarray(image_chw)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = np.clip(image, 0.0, 1.0)

    heatmap = _simple_jet(cam_hw)
    overlay = np.clip((1 - alpha) * image + alpha * heatmap, 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def _simple_jet(x):
    """Minimal blue-to-red colormap for x in [0,1], shape [H,W] -> [H,W,3]."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)
