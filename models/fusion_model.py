import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import create_backbone


class FusionLateModel(nn.Module):
    def __init__(self,
                 backbone_names=('resnet18','resnet18','resnet18'),
                 pretrained=True,
                 embedding_dim=512,
                 num_classes=2,
                 fusion_mode='masked_scalar',
                 use_auxiliary=False,
                 input_channels=(3, 3, 3),
                 pretrained_weights=None,
                 attention_dim=128):
        """Late fusion model with three independent backbones.

        fusion_mode:
            'masked_scalar': fused logits = per-branch logits weighted by one learnable
                scalar per branch, softmax-normalised over the present branches. The
                weights are the same for every sample.
            'gated_attention': gated attention pooling of the branch embeddings
                (Ilse et al. 2018): each present branch gets a per-sample score
                w^T (tanh(V h) * sigmoid(U h)), scores are softmax-normalised over the
                present branches, and a shared classifier reads the weighted sum of
                embeddings. Per-branch logits are still returned (auxiliary losses).
            'gating': reserved, not implemented.
        attention_dim: hidden size of the gated attention scorer.
        input_channels: per-branch input channel count, e.g. (3, 3, 9) for a
            9-channel MRI branch.
        pretrained_weights: optional local checkpoint path (e.g. RadImageNet),
            or a 3-tuple of per-branch paths for mammography/ultrasound/mri.
        """
        super().__init__()
        assert len(backbone_names) == 3
        if pretrained_weights is None or isinstance(pretrained_weights, str):
            pretrained_weights = (pretrained_weights,) * 3
        assert len(pretrained_weights) == 3
        self.num_branches = 3
        self.backbones = nn.ModuleList()
        self.feature_dims = []
        for name, channels, weights_path in zip(backbone_names, input_channels, pretrained_weights):
            m, feat_dim = create_backbone(
                name, pretrained=pretrained, remove_head=True,
                input_channels=channels, pretrained_weights=weights_path,
            )
            self.backbones.append(m)
            self.feature_dims.append(feat_dim)

        # projection heads to embedding_dim
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(fd, embedding_dim), nn.ReLU(inplace=True))
            for fd in self.feature_dims
        ])

        # per-branch classifier heads
        self.classifiers = nn.ModuleList([
            nn.Linear(embedding_dim, num_classes) for _ in range(self.num_branches)
        ])

        # learnable scalar logits for fusion
        self.logit_scalars = nn.Parameter(torch.zeros(self.num_branches))
        self.fusion_mode = fusion_mode
        if fusion_mode == 'gated_attention':
            self.attention_v = nn.Linear(embedding_dim, attention_dim)
            self.attention_u = nn.Linear(embedding_dim, attention_dim)
            self.attention_w = nn.Linear(attention_dim, 1)
            self.fused_classifier = nn.Linear(embedding_dim, num_classes)
        self.use_auxiliary = use_auxiliary

    def forward(self, x1, x2, x3, presence_mask=None):
        """x1,x2,x3: tensors of shapes [B,C,H,W] or None for missing
        presence_mask: optional bool tensor [B,3] indicating availability
        Returns: fused_logits, branch_logits_list, weights
        """
        xs = [x1, x2, x3]
        branch_logits = []
        embeddings = []
        device = xs[0].device if xs[0] is not None else (xs[1].device if xs[1] is not None else xs[2].device)
        batch_size = next(x.shape[0] for x in xs if x is not None)

        if presence_mask is None:
            presence_mask = torch.tensor(
                [[x is not None for x in xs]] * batch_size, dtype=torch.bool, device=device
            )

        for i, x in enumerate(xs):
            # absent samples get zero embeddings/logits (masked out of the fusion below).
            # Run the backbone only on present samples: zero-filled placeholders would
            # otherwise enter BatchNorm's batch statistics in training mode, and in a
            # combined batch most samples are absent from any given branch.
            emb = torch.zeros((batch_size, self.projections[i][0].out_features), device=device)
            logits = torch.zeros((batch_size, self.classifiers[i].out_features), device=device)
            present = presence_mask[:, i].nonzero(as_tuple=True)[0] if x is not None else None
            if present is not None and len(present) > 0:
                feat = self.backbones[i](x[present])
                if feat.dim() == 4:
                    feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                else:
                    feat = feat.flatten(1)
                present_emb = self.projections[i](feat)
                present_logits = self.classifiers[i](present_emb)
                emb = emb.to(present_emb.dtype).index_copy(0, present, present_emb)
                logits = logits.to(present_logits.dtype).index_copy(0, present, present_logits)
            embeddings.append(emb)
            branch_logits.append(logits)

        # compute weights
        pres = presence_mask.float()
        large_neg = -1e9

        if self.fusion_mode == 'masked_scalar':
            # expand scalar per-batch and mask
            scalars = self.logit_scalars.unsqueeze(0).expand(pres.shape[0], -1)
            # mask absent branches
            masked = scalars * pres + (1.0 - pres) * large_neg
            weights = F.softmax(masked, dim=1)
            # weighted sum of logits
            fused = 0
            for i in range(self.num_branches):
                fused = fused + weights[:, i].unsqueeze(1) * branch_logits[i]
        elif self.fusion_mode == 'gated_attention':
            stacked = torch.stack(embeddings, dim=1)  # [B, branches, embedding_dim]
            gated = torch.tanh(self.attention_v(stacked)) * torch.sigmoid(self.attention_u(stacked))
            scores = self.attention_w(gated).squeeze(-1)  # [B, branches]
            scores = scores.masked_fill(~presence_mask.bool(), large_neg)
            weights = F.softmax(scores, dim=1)
            fused = self.fused_classifier((weights.unsqueeze(-1) * stacked).sum(dim=1))
        else:
            raise NotImplementedError('Gating mode not implemented')

        return fused, branch_logits, weights
