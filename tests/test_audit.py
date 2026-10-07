import io

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from audit import artifacts, duplicates, metrics
from audit.report import update_review_file


def _texture(seed, size=128):
    rng = np.random.default_rng(seed)
    image = rng.normal(100, 25, (size, size))
    image = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 2)
    return np.clip(image, 0, 255).astype(np.uint8)


def _frame(images, **columns):
    rows = []
    for k, image in enumerate(images):
        rows.append({
            'sample_id': f's{k}', 'dataset': 'toy', 'label_name': 'benign', 'label': 0, 'used': True,
            'group': f'g{k}', 'split': 'train', 'fold': 1,
            'ahash': metrics.ahash(image), 'dhash': metrics.dhash(image), 'phash': metrics.phash(image),
        })
    frame = pd.DataFrame(rows)
    for column, values in columns.items():
        frame[column] = values
    return frame


def test_to_uint8_uses_stored_bit_range_not_image_range():
    gray = np.array([[0, 2047, 4095]], dtype=np.float32)
    assert metrics.to_uint8(gray, 12).tolist() == [[0, 127, 255]]
    assert metrics.to_uint8(gray[:, :2], 12).max() == 127  # a dim 12-bit image stays dim


def test_near_duplicates_are_confirmed_and_leakage_reported():
    base = _texture(0)
    noisy = np.clip(base.astype(int) + np.random.default_rng(1).integers(-2, 3, base.shape), 0, 255).astype(np.uint8)
    other = _texture(2)
    images = [base, noisy, other]
    thumbs = np.stack([metrics.thumbnail(i, 128) for i in images])
    frame = _frame(images, split=['train', 'val', 'train'], label=[0, 1, 0])

    pairs, _ = duplicates.find_duplicates(frame, thumbs)

    assert len(pairs) == 1
    assert {pairs.iloc[0]['sample_id_a'], pairs.iloc[0]['sample_id_b']} == {'s0', 's1'}
    assert bool(pairs.iloc[0]['label_conflict'])
    leaks = duplicates.split_leakage(pairs)
    assert len(leaks) == 1 and leaks.iloc[0]['leak_type'] == 'holdout_split'


def test_same_group_duplicates_are_not_leakage_and_can_be_skipped():
    base = _texture(3)
    images = [base, base.copy()]
    thumbs = np.stack([metrics.thumbnail(i, 128) for i in images])
    frame = _frame(images, group=['p1', 'p1'], split=['train', 'val'])

    pairs, _ = duplicates.find_duplicates(frame, thumbs)
    assert len(pairs) == 1 and duplicates.split_leakage(pairs).empty
    skipped, _ = duplicates.find_duplicates(frame, thumbs, skip_same_group=True)
    assert skipped.empty


def test_lesion_metrics_measure_contrast_and_geometry():
    image = np.full((100, 100), 150, np.uint8)
    image = np.clip(image + np.random.default_rng(0).normal(0, 5, image.shape), 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[:100, :100]
    mask = (yy - 50) ** 2 + (xx - 30) ** 2 <= 10 ** 2
    image[mask] = 50
    out = metrics.lesion_metrics(mask, image, np.ones_like(mask))
    assert out['lesion_components'] == 1
    assert abs(out['lesion_cx'] - 0.30) < 0.02 and abs(out['lesion_cy'] - 0.50) < 0.02
    assert out['lesion_cnr'] > 10


def test_jpeg_blockiness_detects_compression():
    image = _texture(4, 256)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format='JPEG', quality=10)
    compressed = np.asarray(Image.open(buffer))
    assert metrics.jpeg_blockiness(compressed) > metrics.jpeg_blockiness(image) + 0.1


def test_caliper_crosses_and_text_are_detected_but_blobs_are_not():
    image = np.full((200, 300), 60, np.uint8)
    for cx, cy in ((80, 100), (220, 100)):  # a caliper pair
        cv2.line(image, (cx - 6, cy), (cx + 6, cy), 255, 1)
        cv2.line(image, (cx, cy - 6), (cx, cy + 6), 255, 1)
    cv2.circle(image, (150, 40), 5, 255, -1)  # bright blob: not a caliper
    cv2.putText(image, 'RT UOQ', (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    roi = np.ones(image.shape, bool)

    out = artifacts.ultrasound_artifacts(image, None, roi)

    assert out['caliper_count'] == 2 and out['caliper_flag']
    assert out['text_lines'] >= 1 and out['text_flag']


def test_plain_image_has_no_artifact_flags():
    out = artifacts.ultrasound_artifacts(_texture(5, 200), None, np.ones((200, 200), bool))
    assert not out['caliper_flag'] and not out['text_flag'] and not out['doppler_flag']


def test_doppler_flag_needs_large_colored_blobs():
    rgb = np.stack([_texture(6, 200)] * 3, axis=-1)
    rgb[50:100, 50:100] = (220, 30, 30)
    out = artifacts.ultrasound_artifacts(rgb[..., 0], rgb, np.ones((200, 200), bool))
    assert out['doppler_flag']


def test_review_file_keeps_existing_decisions(tmp_path):
    path = tmp_path / 'outliers_review.csv'
    pd.DataFrame({'sample_id': ['a'], 'reasons': ['x'], 'decision': ['exclude'], 'note': ['blank frame']}).to_csv(path, index=False)
    flagged = pd.DataFrame({'sample_id': ['a', 'b'], 'z_reasons': ['brightness(z=+4.0)', '']})

    update_review_file(path, flagged)

    review = pd.read_csv(path)
    assert review.set_index('sample_id').loc['a', 'decision'] == 'exclude'
    assert review.set_index('sample_id').loc['b', 'reasons'] == 'isolation_forest'
