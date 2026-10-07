"""Sample records and the loaders that turn them into pixel arrays.

A `Sample` is a plain, picklable description of one image as it exists on disk
(the "native" form) and, where training reads something different, the form
training actually sees (the "prepared" form, e.g. a cached 8-bit PNG crop of a
16-bit DICOM). Loading is dispatched on `kind` strings so samples can be shipped
to worker processes.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class Sample:
    sample_id: str
    dataset: str
    modality: str                  # 'ultrasound' | 'mammography' | 'mri'
    label_name: str                # dataset's own class name ('benign', 'Normal', 'Actionable', ...)
    label: Optional[int]           # 0 benign / 1 malignant; None when not a training label
    used: bool                     # included by the training loader
    group: Optional[str]           # patient / case id used for grouping splits
    split: Optional[str]           # default hold-out split the training code assigns
    fold: Optional[int]            # k-fold assignment (official or this repo's), if any
    native_kind: str               # 'image' | 'dicom' | 'npy'
    native_path: str
    native_frame: Optional[int] = None
    prepared_kind: Optional[str] = None
    prepared_path: Optional[str] = None
    mask_path: Optional[str] = None
    box: Optional[tuple] = None    # (x0, y0, x1, y1) in native pixel coordinates
    extra: dict = field(default_factory=dict)


@dataclass
class Loaded:
    """Pixel data plus the facts needed to interpret it."""
    gray: np.ndarray               # 2D float32, native intensity units
    rgb: Optional[np.ndarray]      # HxWx3 uint8 when the source is color, else None
    bits: int                      # stored bit depth (8, 12, 16, ...)
    channels: Optional[np.ndarray] = None   # HxWxC for multi-channel MRI arrays
    header: Optional[dict] = None


def _load_image(path):
    image = Image.open(path)
    mode = image.mode
    if mode in ('I;16', 'I;16B', 'I;16L', 'I'):
        array = np.asarray(image, dtype=np.float32)
        return Loaded(gray=array, rgb=None, bits=16)
    if mode in ('RGB', 'RGBA', 'P', 'CMYK', 'LA'):
        rgb = np.asarray(image.convert('RGB'))
        channel_spread = np.ptp(rgb.astype(np.int16), axis=2)
        is_color = bool((channel_spread > 8).any())
        gray = np.asarray(image.convert('L'), dtype=np.float32)
        return Loaded(gray=gray, rgb=rgb if is_color else None, bits=8)
    return Loaded(gray=np.asarray(image.convert('L'), dtype=np.float32), rgb=None, bits=8)


def dicom_header(ds):
    """Flatten a pydicom dataset's non-pixel, non-sequence elements to {keyword: str}."""
    header = {}
    for element in ds:
        if element.tag == (0x7FE0, 0x0010) or element.VR in ('SQ', 'OB', 'OW', 'OF', 'UN'):
            continue
        keyword = element.keyword or str(element.tag)
        value = str(element.value)
        header[keyword] = value if len(value) <= 256 else value[:256]
    return header


def _load_dicom(path, frame=None):
    import pydicom
    from pydicom.pixels import apply_voi_lut, pixel_array

    ds = pydicom.dcmread(path, stop_before_pixels=True)
    frames = int(getattr(ds, 'NumberOfFrames', 1) or 1)
    if frame is None and frames > 1:
        frame = frames // 2  # decode only the middle frame of a multi-frame volume
    array = pixel_array(path, index=frame) if frame is not None else pixel_array(path)
    bits = int(getattr(ds, 'BitsStored', 16) or 16)
    array = apply_voi_lut(array, ds).astype(np.float32)
    if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        array = array.max() - array
    return Loaded(gray=array, rgb=None, bits=bits, header=dicom_header(ds))


def _load_npy(path):
    array = np.load(path)
    if array.ndim == 3:
        return Loaded(gray=array.astype(np.float32).mean(axis=2), rgb=None,
                      bits=8 * array.dtype.itemsize, channels=array)
    return Loaded(gray=array.astype(np.float32), rgb=None, bits=8 * array.dtype.itemsize)


def load(kind, path, frame=None):
    if kind == 'image':
        return _load_image(path)
    if kind == 'dicom':
        return _load_dicom(path, frame)
    if kind == 'npy':
        return _load_npy(path)
    raise ValueError(f'unknown sample kind {kind!r}')


def load_native(sample):
    return load(sample.native_kind, sample.native_path, sample.native_frame)


def load_prepared(sample):
    if sample.prepared_kind is None:
        return None
    return load(sample.prepared_kind, sample.prepared_path)


def load_mask(sample, shape):
    """Binary lesion mask in native coordinates: from the mask file if present,
    else rasterized from the box, else None."""
    if sample.mask_path and Path(sample.mask_path).exists():
        mask = np.asarray(Image.open(sample.mask_path).convert('L')) > 127
        return mask
    if sample.box is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in sample.box)
        mask = np.zeros(shape, dtype=bool)
        mask[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
        return mask
    return None
