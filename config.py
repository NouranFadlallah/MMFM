from dataclasses import dataclass


@dataclass
class Config:
    backbone_names = ('resnet18','resnet18','resnet18')
    pretrained = True
    embedding_dim = 512
    num_classes = 2
    fusion_mode = 'masked_scalar'
    batch_size = 8
    lr = 1e-4
    epochs = 10
    modality_dropout_prob = 0.1
    device = 'cuda'  # change to cpu if needed


cfg = Config()
