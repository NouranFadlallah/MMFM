"""Per-sample worker: load one image, compute every per-image metric, return a
flat row plus its DICOM header (if any) and a grayscale thumbnail."""
import os
from pathlib import Path

import numpy as np
from PIL import Image

from audit import artifacts, metrics
from audit.sample import load_mask, load_native, load_prepared

PREPARED_KEYS = ('height', 'width', 'bits_stored', 'brightness', 'contrast', 'entropy',
                 'dynamic_range', 'sharpness_lapvar', 'clip_low_frac', 'clip_high_frac')


def _base_row(sample):
    row = {
        'sample_id': sample.sample_id, 'dataset': sample.dataset, 'modality': sample.modality,
        'label_name': sample.label_name, 'label': sample.label, 'used': sample.used,
        'group': sample.group, 'split': sample.split, 'fold': sample.fold,
        'native_path': sample.native_path, 'native_frame': sample.native_frame,
        'prepared_path': sample.prepared_path, 'mask_path': sample.mask_path,
        'has_mask': bool(sample.mask_path), 'has_box': sample.box is not None,
        'file_ext': Path(sample.native_path).suffix.lower(),
    }
    try:
        row['file_bytes'] = os.path.getsize(sample.native_path)
    except OSError:
        row['file_bytes'] = None
    for key, value in sample.extra.items():
        row[f'meta_{key}'] = value
    return row


def analyze_sample(sample, thumb_size):
    row = _base_row(sample)
    try:
        loaded = load_native(sample)
    except Exception as error:  # report unreadable files instead of aborting the run
        row['load_error'] = repr(error)
        return row, None, None
    gray = loaded.gray
    row['is_color'] = loaded.rgb is not None
    roi = metrics.content_roi(metrics.to_uint8(gray, loaded.bits), sample.modality)
    quality, img8 = metrics.quality_metrics(gray, loaded.bits, roi, sample.modality)
    row.update(quality)
    row['ahash'], row['dhash'], row['phash'] = metrics.ahash(img8), metrics.dhash(img8), metrics.phash(img8)

    if sample.modality == 'ultrasound':
        row.update(artifacts.ultrasound_artifacts(img8, loaded.rgb, roi))
    elif sample.modality == 'mammography':
        row.update(artifacts.mammography_artifacts(img8, roi, loaded.header))
    if loaded.channels is not None:
        row.update(artifacts.mri_channel_stats(loaded.channels))

    mask = load_mask(sample, gray.shape)
    if mask is not None:
        if mask.shape != gray.shape:
            row['mask_shape_mismatch'] = True
            mask = np.asarray(Image.fromarray(mask).resize(gray.shape[::-1], Image.NEAREST))
        row.update(metrics.lesion_metrics(mask, img8, roi))

    if sample.prepared_kind is not None:
        try:
            prepared = load_prepared(sample)
            prepared_roi = metrics.content_roi(metrics.to_uint8(prepared.gray, prepared.bits), sample.modality)
            prepared_quality, _ = metrics.quality_metrics(prepared.gray, prepared.bits, prepared_roi, sample.modality)
            row.update({f'prep_{k}': prepared_quality[k] for k in PREPARED_KEYS})
        except Exception as error:
            row['prep_load_error'] = repr(error)

    return row, loaded.header, metrics.thumbnail(img8, thumb_size)
