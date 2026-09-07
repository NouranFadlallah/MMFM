import csv
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from PIL import Image


MISSING_TOKEN = "MISSING"


def _safe_list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    supported_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".nii", ".nii.gz"}
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and (p.suffix.lower() in supported_suffixes or p.name.endswith(".nii.gz"))
    )


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_array_from_path(image_path: Path):
    suffix = image_path.suffix.lower()
    if suffix == ".dcm":
        try:
            import pydicom
        except ImportError as exc:
            raise RuntimeError("pydicom is required for DICOM files") from exc
        ds = pydicom.dcmread(image_path)
        arr = ds.pixel_array
        return arr

    if suffix in {".nii", ".gz"} or image_path.name.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise RuntimeError("nibabel is required for NIfTI files") from exc
        img = nib.load(image_path)
        arr = img.get_fdata()
        return arr

    img = Image.open(image_path).convert("RGB")
    return img


def _resize_and_save(image_path: Path, output_path: Path, target_size: Tuple[int, int]) -> Path:
    data = _load_array_from_path(image_path)

    if isinstance(data, Image.Image):
        img = data.resize(target_size, Image.Resampling.BILINEAR)
    else:
        if data.ndim == 3:
            data = data[..., 0]
        if data.ndim == 2:
            import numpy as np

            arr = np.asarray(data)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            img = Image.fromarray(arr.astype("uint8"))
        else:
            raise ValueError(f"Unsupported array shape for {image_path}")
        img = img.resize(target_size, Image.Resampling.BILINEAR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def _write_manifest(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["img1", "img2", "img3", "label"]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(
    source_dir: str | os.PathLike,
    output_csv: str | os.PathLike,
    output_dir: str | os.PathLike,
    target_size: Tuple[int, int] = (224, 224),
    label_map: Optional[Dict[str, int]] = None,
    class_dirs: Optional[Iterable[str]] = None,
) -> list[dict]:
    """Create a manifest CSV compatible with TripleImageDataset.

    The function scans a source directory for image files, copies/resizes them into
    an output folder, and writes a CSV with columns img1, img2, img3, label.

    For a single-image dataset, the same processed image is used for img1/img2/img3.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_csv = Path(output_csv)
    _ensure_output_dir(output_dir)

    if class_dirs is None:
        class_dirs = ["benign", "malignant"]

    rows = []
    for class_name in class_dirs:
        class_dir = source_dir / class_name
        if not class_dir.exists():
            continue

        images = _safe_list_images(class_dir)
        for image_path in images:
            rel_name = image_path.stem + image_path.suffix
            if image_path.name.endswith(".nii.gz"):
                rel_name = image_path.name.replace(".nii.gz", ".png")
            out_path = output_dir / class_name / rel_name
            _resize_and_save(image_path, out_path, target_size)

            label = label_map[class_name] if label_map is not None else 0
            rows.append(
                {
                    "img1": str(out_path),
                    "img2": str(out_path),
                    "img3": str(out_path),
                    "label": int(label),
                }
            )

    if not rows:
        raise ValueError(f"No images found under {source_dir}")

    _write_manifest(rows, output_csv)
    return rows


def build_manifest_from_folder(
    source_dir: str | os.PathLike,
    output_csv: str | os.PathLike,
    output_dir: str | os.PathLike,
    target_size: Tuple[int, int] = (224, 224),
    label_map: Optional[Dict[str, int]] = None,
) -> list[dict]:
    """Convenience wrapper for binary class folders."""
    return build_manifest(
        source_dir=source_dir,
        output_csv=output_csv,
        output_dir=output_dir,
        target_size=target_size,
        label_map=label_map,
        class_dirs=list(label_map.keys()) if label_map is not None else ["benign", "malignant"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a CSV manifest for image-based training")
    parser.add_argument("source_dir", help="Root folder containing class directories")
    parser.add_argument("output_csv", help="Path to output manifest CSV")
    parser.add_argument("output_dir", help="Directory for resized images")
    parser.add_argument("--target-size", nargs=2, type=int, default=[224, 224], help="Resize target (width height)")
    parser.add_argument("--label-map", nargs="*", default=[], help="Class label pairs like benign=0 malignant=1")
    args = parser.parse_args()

    label_map = {}
    for item in args.label_map:
        name, value = item.split("=", 1)
        label_map[name] = int(value)

    build_manifest(
        source_dir=args.source_dir,
        output_csv=args.output_csv,
        output_dir=args.output_dir,
        target_size=(args.target_size[0], args.target_size[1]),
        label_map=label_map or None,
    )
    print(f"Created manifest at {args.output_csv}")
