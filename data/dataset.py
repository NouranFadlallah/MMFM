import os
import random
from pathlib import Path
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
    def __init__(self, size=224, augment=False, crop_margin=0.05, contrast_stretch=False):
        self.size = (size, size)
        self.augment = augment
        self.crop_margin = crop_margin
        self.contrast_stretch = contrast_stretch

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
        array = np.asarray(image, dtype=np.float32)
        if self.contrast_stretch:
            # 5th/95th percentile min-max normalization, per Gomez-Flores et al. 2024
            low, high = np.percentile(array, [5, 95])
            if high > low:
                array = np.clip((array - low) / (high - low), 0.0, 1.0)
            else:
                array = array / 255.0
        else:
            array = array / 255.0
        array = array.astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)


class MiasTransform:
    """Preprocess MIAS lesion patches using the coordinates in Info.txt."""
    def __init__(self, size=224, augment=False, crop_scale=1.5):
        self.size = (size, size)
        self.augment = augment
        self.crop_scale = crop_scale

    def __call__(self, image, crop_box=None):
        image = image.convert('L')
        if crop_box is not None:
            left, top, right, bottom = crop_box
            image = image.crop((left, top, right, bottom))
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = image.rotate(random.uniform(-10.0, 10.0), resample=Image.Resampling.BILINEAR)
        image = image.resize(self.size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)


class BreadmTransform:
    """Prepare a BreaDM img9Se array while preserving its nine slice channels."""
    def __init__(self, size=224, augment=False):
        self.size = (size, size)
        self.augment = augment

    def __call__(self, array):
        if array.ndim != 3 or array.shape[-1] != 9:
            raise ValueError(f'Expected BreaDM array with shape [H,W,9], got {array.shape}')
        array = np.asarray(array, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False
        ).squeeze(0)
        if self.augment:
            if random.random() < 0.5:
                tensor = tensor.flip(-1)
            if random.random() < 0.5:
                tensor = tensor.flip(-2)
        return tensor


class BreaDMDataset(Dataset):
    """Dataset for BreaDM's official class/split directory layout."""
    def __init__(self, root, split, transform=None):
        root = os.fspath(root)
        self.samples = []
        for class_name, label in (('Benign', 0), ('Malignant', 1)):
            for path in sorted(Path(root, split, class_name).rglob('*.npy')):
                self.samples.append((path, label))
        if not self.samples:
            raise ValueError(f'No BreaDM samples found under {root}/{split}')
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        array = np.load(path)
        image = self.transform(array) if self.transform else torch.from_numpy(array).permute(2, 0, 1).float()
        present = torch.ones(3, dtype=torch.bool)
        return image, image.clone(), image.clone(), label, present


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

    def _load_image(self, path, transform, mask_path=None, crop_box=None):
        if not path or (isinstance(path, str) and path.strip().upper() == MISSING_TOKEN):
            return None
        if not os.path.exists(path):
            return None
        im = Image.open(path)
        if transform:
            mask = Image.open(mask_path) if mask_path and os.path.exists(mask_path) else None
            if mask is not None:
                im = transform(im, mask)
            elif crop_box is not None:
                im = transform(im, crop_box)
            else:
                im = transform(im)
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
        crop_box = row['crop_box'] if 'crop_box' in row else None
        l = row['label'] if 'label' in row else 0
        if self.label_map:
            l = self.label_map[l]

        img1 = self._load_image(p1, self.transform1, mask, crop_box)
        img2 = self._load_image(p2, self.transform2, mask, crop_box)
        img3 = self._load_image(p3, self.transform3, mask, crop_box)

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


class SingleModalityBranchDataset(Dataset):
    """Wraps a single-modality dataset and places its image into one branch slot.

    base_dataset must yield (img1, img2, img3, label, presence) with img1==img2==img3
    (as produced by _make_dataset/TripleImageDataset) or (img, img, img, label, present)
    as produced by BreaDMDataset. Only branch_index is populated; the other two
    branches are zero-filled with their own channel counts so they stack correctly
    with real samples from other branches in a combined dataset.
    """
    def __init__(self, base_dataset, branch_index, channels_per_branch=(3, 3, 9)):
        assert branch_index in (0, 1, 2)
        self.base_dataset = base_dataset
        self.branch_index = branch_index
        self.channels_per_branch = channels_per_branch

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        images = self.base_dataset[index]
        label = images[3]
        image = images[self.branch_index]
        size = image.shape[-2:]
        branch_images = []
        presence = [False, False, False]
        for i, channels in enumerate(self.channels_per_branch):
            if i == self.branch_index:
                branch_images.append(image)
                presence[i] = True
            else:
                branch_images.append(torch.zeros(channels, *size))
        return (*branch_images, int(label), torch.tensor(presence, dtype=torch.bool))
