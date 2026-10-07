"""Per-image quality, noise, frequency and lesion metrics, plus perceptual hashes.

Conventions:
- `gray` is a 2D float32 array in native intensity units; `bits` its stored bit
  depth. `to_uint8` maps the full stored range (not the per-image min/max) to
  0-255, so 8-bit metrics stay comparable across images of one dataset.
- `roi` is a boolean mask of the anatomical content region (breast for
  mammography, scan sector for ultrasound, everything for MRI); intensity
  statistics are computed inside it so black background doesn't dominate.
"""
import numpy as np
from scipy import ndimage

HASH_SIZE = 8


def to_uint8(gray, bits):
    full_scale = float(2 ** bits - 1) if bits > 8 else 255.0
    if bits > 8 and gray.max() > full_scale:
        full_scale = float(gray.max())
    return np.clip(gray / full_scale * 255.0, 0, 255).astype(np.uint8)


def largest_component(mask):
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def otsu_threshold(values):
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = weight1[-1] - weight1
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    total = (hist * centers).sum()
    mean2 = (total - np.cumsum(hist * centers)) / np.maximum(weight2, 1)
    between = weight1 * weight2 * (mean1 - mean2) ** 2
    return centers[int(np.argmax(between))]


def content_roi(img8, modality):
    """Breast mask (mammography), scan-region mask (ultrasound), or all pixels (MRI)."""
    if modality == 'mri' or img8.size == 0:
        return np.ones(img8.shape, dtype=bool)
    small = ndimage.uniform_filter(img8.astype(np.float32), size=5)
    if modality == 'mammography':
        threshold = max(otsu_threshold(small[small > 2]) * 0.5, 8) if (small > 2).any() else 8
    else:
        threshold = 8
    mask = small > threshold
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = largest_component(mask)
    mask = ndimage.binary_fill_holes(mask)
    if mask.sum() < 0.02 * mask.size:
        return np.ones(img8.shape, dtype=bool)
    return mask


def entropy(img8, roi):
    hist = np.bincount(img8[roi].ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def noise_sigma(img):
    """Immerkaer (1996) fast noise standard-deviation estimate."""
    if min(img.shape) < 3:
        return float('nan')
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = ndimage.convolve(img.astype(np.float32), kernel)[1:-1, 1:-1]
    h, w = img.shape
    return float(np.sqrt(np.pi / 2) * np.abs(response).sum() / (6 * (w - 2) * (h - 2)))


def spectrum_stats(img8):
    """High-frequency energy fraction and log-log spectral slope of a centered
    square crop (no resampling, which would itself alter the spectrum)."""
    side = min(img8.shape)
    side = min(side, 512)
    if side < 32:
        return float('nan'), float('nan')
    h, w = img8.shape
    top, left = (h - side) // 2, (w - side) // 2
    crop = img8[top:top + side, left:left + side].astype(np.float32)
    crop -= crop.mean()
    window = np.outer(np.hanning(side), np.hanning(side))
    power = np.abs(np.fft.fftshift(np.fft.fft2(crop * window))) ** 2
    yy, xx = np.indices(power.shape)
    radius = np.hypot(yy - side / 2, xx - side / 2) / side  # cycles/pixel
    radial = ndimage.mean(power, labels=np.clip((radius * side).astype(int), 0, side), index=np.arange(side // 2))
    freqs = np.arange(side // 2) / side
    total = radial[1:].sum()
    high = radial[freqs > 0.25].sum()
    band = (freqs > 0.02) & (freqs < 0.4) & (radial > 0)
    slope = float(np.polyfit(np.log(freqs[band]), np.log(radial[band]), 1)[0]) if band.sum() > 3 else float('nan')
    return float(high / total) if total > 0 else float('nan'), slope


def jpeg_blockiness(img8):
    """Ratio of mean absolute intensity steps across 8-px block boundaries to
    steps elsewhere; ~1 for uncompressed images, >1 with JPEG blocking."""
    img = img8.astype(np.float32)
    if min(img.shape) < 32:
        return float('nan')
    dx = np.abs(np.diff(img, axis=1))
    dy = np.abs(np.diff(img, axis=0))
    col_boundary = (np.arange(dx.shape[1]) % 8) == 7
    row_boundary = (np.arange(dy.shape[0]) % 8) == 7
    boundary = dx[:, col_boundary].mean() + dy[row_boundary, :].mean()
    inner = dx[:, ~col_boundary].mean() + dy[~row_boundary, :].mean()
    return float(boundary / inner) if inner > 0 else float('nan')


def speckle_snr(img, roi, block=16):
    """Median mean/std over homogeneous blocks inside the ROI (~1.91 for fully
    developed Rayleigh speckle on envelope data; deviations indicate filtering
    or log compression)."""
    h, w = img.shape
    ratios = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            if not roi[y:y + block, x:x + block].all():
                continue
            patch = img[y:y + block, x:x + block]
            std = patch.std()
            if std > 0 and patch.mean() > 10:
                ratios.append(patch.mean() / std)
    if len(ratios) < 5:
        return float('nan')
    ratios = np.sort(ratios)
    # homogeneous regions = the most uniform half of blocks
    return float(np.median(ratios[len(ratios) // 2:]))


def quality_metrics(gray, bits, roi, modality):
    """Section B + H metrics for one 2D image."""
    img8 = to_uint8(gray, bits)
    values = gray[roi]
    out = {
        'height': gray.shape[0],
        'width': gray.shape[1],
        'aspect_ratio': gray.shape[1] / gray.shape[0] if gray.shape[0] else float('nan'),
        'bits_stored': bits,
        'roi_fraction': float(roi.mean()),
        'brightness': float(img8[roi].mean()),
        'contrast': float(img8[roi].std(ddof=1)) if roi.sum() > 1 else float('nan'),
        'sharpness_lapvar': float(ndimage.laplace(img8.astype(np.float32))[roi].var()),
        'entropy': entropy(img8, roi),
        'native_min': float(values.min()) if values.size else float('nan'),
        'native_max': float(values.max()) if values.size else float('nan'),
        'dynamic_range': float(values.max() - values.min()) if values.size else float('nan'),
        'dynamic_range_frac': float((values.max() - values.min()) / (2 ** bits - 1)) if values.size else float('nan'),
        'clip_low_frac': float((img8[roi] == 0).mean()),
        'clip_high_frac': float((img8[roi] == 255).mean()),
        'noise_sigma': noise_sigma(img8),
    }
    out['snr'] = out['brightness'] / out['noise_sigma'] if out['noise_sigma'] and out['noise_sigma'] > 0 else float('nan')
    out['hf_energy_frac'], out['spectral_slope'] = spectrum_stats(img8)
    out['jpeg_blockiness'] = jpeg_blockiness(img8)
    if modality == 'ultrasound':
        out['speckle_snr'] = speckle_snr(img8.astype(np.float32), roi)
    return out, img8


def lesion_metrics(mask, img8, roi):
    """Section F/K: lesion geometry, annotation shape, and lesion-vs-surround CNR."""
    h, w = mask.shape
    area = int(mask.sum())
    out = {'lesion_present': area > 0}
    if area == 0:
        return out
    labels, components = ndimage.label(mask)
    ys, xs = np.nonzero(mask)
    out.update({
        'lesion_area_frac': area / float(h * w),
        'lesion_components': int(components),
        'lesion_bbox_w': int(xs.max() - xs.min() + 1),
        'lesion_bbox_h': int(ys.max() - ys.min() + 1),
        'lesion_cx': float(xs.mean() / w),
        'lesion_cy': float(ys.mean() / h),
        'lesion_outside_roi_frac': float((mask & ~roi).sum() / area),
    })
    out['lesion_bbox_fill'] = area / float(out['lesion_bbox_w'] * out['lesion_bbox_h'])
    perimeter = int((mask & ~ndimage.binary_erosion(mask)).sum())
    out['lesion_compactness'] = perimeter ** 2 / (4 * np.pi * area) if area else float('nan')
    # contrast-to-noise vs a ring of surrounding tissue ~ 25% of the lesion's size
    radius = max(3, int(0.25 * np.sqrt(area / np.pi)))
    ring = ndimage.binary_dilation(mask, iterations=radius) & ~mask & roi
    if ring.sum() > 10:
        lesion_mean, ring_mean, ring_std = img8[mask].mean(), img8[ring].mean(), img8[ring].std()
        out['lesion_mean'] = float(lesion_mean)
        out['ring_mean'] = float(ring_mean)
        out['lesion_cnr'] = float(abs(lesion_mean - ring_mean) / ring_std) if ring_std > 0 else float('nan')
    return out


def _resize(img8, size):
    from PIL import Image
    return np.asarray(Image.fromarray(img8).resize(size, Image.BILINEAR), dtype=np.float32)


def ahash(img8):
    small = _resize(img8, (HASH_SIZE, HASH_SIZE))
    return _pack(small > small.mean())


def dhash(img8):
    small = _resize(img8, (HASH_SIZE + 1, HASH_SIZE))
    return _pack(small[:, 1:] > small[:, :-1])


def phash(img8):
    from scipy.fft import dctn
    small = _resize(img8, (32, 32))
    low = dctn(small, norm='ortho')[:HASH_SIZE, :HASH_SIZE]
    return _pack(low > np.median(low.ravel()[1:]))


def _pack(bits):
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    # store as signed int64 so it round-trips through parquet
    return np.int64(np.uint64(value).astype(np.int64))


def thumbnail(img8, size):
    from PIL import Image
    return np.asarray(Image.fromarray(img8).resize((size, size), Image.BILINEAR), dtype=np.uint8)
