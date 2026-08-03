import os
from typing import Optional
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


MISSING_TOKEN = 'MISSING'


class TripleImageDataset(Dataset):
    """Dataset expecting a CSV with columns: img1,img2,img3,label

    Missing image can be represented as empty string or 'MISSING'.
    Returns (img1,img2,img3,label,presence_mask)
    """
    def __init__(self, csv_path, transform1=None, transform2=None, transform3=None, label_map=None):
        df = pd.read_csv(csv_path)
        self.df = df
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.label_map = label_map

    def _load_image(self, path, transform):
        if not path or (isinstance(path, str) and path.strip().upper() == MISSING_TOKEN):
            return None
        if not os.path.exists(path):
            return None
        im = Image.open(path).convert('RGB')
        if transform:
            im = transform(im)
        return im

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p1 = str(row['img1']) if 'img1' in row else ''
        p2 = str(row['img2']) if 'img2' in row else ''
        p3 = str(row['img3']) if 'img3' in row else ''
        l = row['label'] if 'label' in row else 0
        if self.label_map:
            l = self.label_map[l]

        img1 = self._load_image(p1, self.transform1)
        img2 = self._load_image(p2, self.transform2)
        img3 = self._load_image(p3, self.transform3)

        presence = [1 if x is not None else 0 for x in (img1, img2, img3)]

        # replace None with zeros tensors of appropriate shape
        # determine a default shape from transforms if possible
        def to_tensor_or_zero(x, transform):
            if x is None:
                # try to get size from transform
                if transform and hasattr(transform, 'transforms'):
                    # try a common Resize + ToTensor
                    # fallback to (3,224,224)
                    return torch.zeros(3,224,224)
                else:
                    return torch.zeros(3,224,224)
            else:
                return x

        img1 = to_tensor_or_zero(img1, self.transform1)
        img2 = to_tensor_or_zero(img2, self.transform2)
        img3 = to_tensor_or_zero(img3, self.transform3)

        presence_mask = torch.tensor(presence, dtype=torch.bool)

        return img1, img2, img3, int(l), presence_mask
