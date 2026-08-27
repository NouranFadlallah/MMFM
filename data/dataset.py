import os
import random
from typing import Optional
import pandas as pd
from PIL import Image, ImageFilter
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


MISSING_TOKEN = 'MISSING'


class BusbraTransform:
    """Paper-inspired preprocessing for BUS-BRA grayscale images and masks."""
    def __init__(self, size=224, augment=False, crop_margin=0.05):
        self.size = (size, size)
        self.augment = augment
        self.crop_margin = crop_margin

    def __call__(self, image, mask=None):
        image = image.convert('L').filter(ImageFilter.MedianFilter(size=3))
        if mask is not None:
            mask = mask.convert('L').filter(ImageFilter.MinFilter(size=3)).filter(
                ImageFilter.MaxFilter(size=3)
            )
            mask_array = np.asarray(mask) > 0
            coordinates = np.argwhere(mask_array)
            if coordinates.size:
                top, left = coordinates.min(axis=0)
                bottom, right = coordinates.max(axis=0) + 1
                margin_y = max(1, int((bottom - top) * self.crop_margin))
                margin_x = max(1, int((right - left) * self.crop_margin))
                top = max(0, top - margin_y)
                left = max(0, left - margin_x)
                bottom = min(image.height, bottom + margin_y)
                right = min(image.width, right + margin_x)
                image = image.crop((left, top, right, bottom))

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            angle = random.uniform(-10.0, 10.0)
            scale = random.uniform(1.0, 1.1)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
            width, height = image.size
            crop_width = max(1, int(width / scale))
            crop_height = max(1, int(height / scale))
            left = max(0, (width - crop_width) // 2)
            top = max(0, (height - crop_height) // 2)
            image = image.crop((left, top, left + crop_width, top + crop_height))

        image = image.resize(self.size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)


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

    def _load_image(self, path, transform, mask_path=None):
        if not path or (isinstance(path, str) and path.strip().upper() == MISSING_TOKEN):
            return None
        if not os.path.exists(path):
            return None
        im = Image.open(path)
        if transform:
            mask = Image.open(mask_path) if mask_path and os.path.exists(mask_path) else None
            im = transform(im, mask) if mask is not None else transform(im)
        else:
            im = im.convert('RGB')
        return im

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p1 = str(row['img1']) if 'img1' in row else ''
        p2 = str(row['img2']) if 'img2' in row else ''
        p3 = str(row['img3']) if 'img3' in row else ''
        mask = str(row['mask']) if 'mask' in row else ''
        l = row['label'] if 'label' in row else 0
        if self.label_map:
            l = self.label_map[l]

        img1 = self._load_image(p1, self.transform1, mask)
        img2 = self._load_image(p2, self.transform2, mask)
        img3 = self._load_image(p3, self.transform3, mask)

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
