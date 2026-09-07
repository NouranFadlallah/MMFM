import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.dataset import (
    BreaDMDataset,
    BreadmTransform,
    MiasTransform,
    SingleModalityBranchDataset,
    TripleImageDataset,
)


def test_mias_transform_crops_and_normalizes():
    array = np.zeros((40, 40), dtype=np.uint8)
    array[10:30, 10:30] = 64
    image = Image.fromarray(array, mode="L")

    output = MiasTransform(size=8, augment=False)(image, crop_box=(10, 10, 30, 30))

    assert output.shape == (3, 8, 8)
    assert output.dtype == torch.float32
    assert torch.allclose(output, torch.full_like(output, 64 / 255))


def test_breadm_transform_preserves_nine_channels_and_resizes():
    array = (np.random.rand(20, 24, 9) * 255).astype(np.uint8)

    output = BreadmTransform(size=12, augment=False)(array)

    assert output.shape == (9, 12, 12)
    assert output.dtype == torch.float32


def test_breadm_transform_rejects_wrong_channel_count():
    array = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        BreadmTransform(size=8)(array)


def _identity_transform(image):
    array = np.asarray(image.resize((4, 4)), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)


def test_triple_image_dataset_marks_missing_modalities(tmp_path):
    present_path = tmp_path / "present.png"
    Image.new("L", (10, 10), 200).save(present_path)

    manifest = pd.DataFrame(
        [
            {"img1": str(present_path), "img2": str(present_path), "img3": str(present_path), "label": 0},
            {"img1": str(present_path), "img2": "MISSING", "img3": str(present_path), "label": 1},
            {"img1": str(present_path), "img2": str(tmp_path / "does_not_exist.png"), "img3": str(present_path), "label": 1},
        ]
    )
    csv_path = tmp_path / "manifest.csv"
    manifest.to_csv(csv_path, index=False)

    dataset = TripleImageDataset(
        csv_path, transform1=_identity_transform, transform2=_identity_transform, transform3=_identity_transform
    )

    img1, img2, img3, label, presence = dataset[0]
    assert presence.tolist() == [True, True, True]
    assert img2.shape == (3, 4, 4)

    _, img2_missing, _, label1, presence1 = dataset[1]
    assert presence1.tolist() == [True, False, True]
    assert torch.equal(img2_missing, torch.zeros(3, 224, 224))
    assert label1 == 1

    _, img2_absent, _, _, presence2 = dataset[2]
    assert presence2.tolist() == [True, False, True]
    assert torch.equal(img2_absent, torch.zeros(3, 224, 224))


class _FakeTripleDataset(Dataset):
    """Minimal stand-in for a base dataset yielding (img1,img2,img3,label,presence)."""

    def __init__(self, image, label):
        self.image = image
        self.label = label

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.image, self.image.clone(), self.image.clone(), self.label, torch.ones(3, dtype=torch.bool)


def test_single_modality_branch_dataset_only_fills_its_branch():
    image = torch.rand(3, 16, 16)
    base = _FakeTripleDataset(image, label=1)

    wrapped = SingleModalityBranchDataset(base, branch_index=1, channels_per_branch=(3, 3, 9))

    img1, img2, img3, label, presence = wrapped[0]
    assert label == 1
    assert presence.tolist() == [False, True, False]
    assert torch.equal(img2, image)
    assert torch.equal(img1, torch.zeros(3, 16, 16))
    assert img3.shape == (9, 16, 16)
    assert torch.equal(img3, torch.zeros(9, 16, 16))


def test_breadm_dataset_loads_labeled_npy_files(tmp_path):
    for split, class_name, label in (("train", "Benign", 0), ("train", "Malignant", 1)):
        class_dir = tmp_path / split / class_name
        class_dir.mkdir(parents=True)
        np.save(class_dir / "sample.npy", np.zeros((8, 8, 9), dtype=np.uint8))

    dataset = BreaDMDataset(tmp_path, "train", transform=BreadmTransform(size=8, augment=False))

    assert len(dataset) == 2
    labels = sorted(label for _, _, _, label, _ in (dataset[i] for i in range(len(dataset))))
    assert labels == [0, 1]
    image, image2, image3, label, presence = dataset[0]
    assert image.shape == (9, 8, 8)
    assert torch.equal(image, image2) and torch.equal(image, image3)
    assert presence.tolist() == [True, True, True]


def test_breadm_dataset_raises_when_no_samples_found(tmp_path):
    (tmp_path / "train" / "Benign").mkdir(parents=True)
    with pytest.raises(ValueError):
        BreaDMDataset(tmp_path, "train")
