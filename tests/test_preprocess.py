import csv
from pathlib import Path

from PIL import Image
import numpy as np
import torch

from data.dataset import BusbraTransform
from data.preprocess import build_manifest


def test_busbra_transform_crops_mask_and_normalizes():
    image = Image.fromarray(np.full((20, 30), 128, dtype=np.uint8), mode="L")
    mask_array = np.zeros((20, 30), dtype=np.uint8)
    mask_array[5:15, 10:20] = 255
    mask = Image.fromarray(mask_array, mode="L")

    output = BusbraTransform(size=16)(image, mask)

    assert output.shape == (3, 16, 16)
    assert output.dtype == torch.float32
    assert torch.allclose(output, torch.full_like(output, 128 / 255))


def test_build_manifest_creates_rows_and_processed_images(tmp_path):
    source_dir = tmp_path / "source"
    benign_dir = source_dir / "benign"
    malignant_dir = source_dir / "malignant"
    benign_dir.mkdir(parents=True)
    malignant_dir.mkdir(parents=True)

    benign_image = benign_dir / "sample.jpg"
    malignant_image = malignant_dir / "sample.jpg"
    Image.new("RGB", (16, 16), (0, 0, 0)).save(benign_image)
    Image.new("RGB", (16, 16), (255, 255, 255)).save(malignant_image)

    output_dir = tmp_path / "processed"
    output_csv = tmp_path / "manifest.csv"

    rows = build_manifest(
        source_dir,
        output_csv,
        output_dir,
        target_size=(32, 32),
        label_map={"benign": 0, "malignant": 1},
    )

    assert output_csv.exists()
    with output_csv.open(newline="", encoding="utf-8") as handle:
        data = list(csv.DictReader(handle))

    assert len(data) == 2
    assert set(data[0].keys()) == {"img1", "img2", "img3", "label"}
    assert data[0]["label"] == "0"
    assert data[1]["label"] == "1"
    assert all(Path(row["img1"]).exists() for row in data)
    assert all(Path(row["img2"]).exists() for row in data)
    assert all(Path(row["img3"]).exists() for row in data)
    assert len(rows) == 2
