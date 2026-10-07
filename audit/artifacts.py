"""Section E: modality-specific artifact detectors.

These are heuristics meant to *rank and flag* images for manual review through
the contact sheets, not to be trusted blindly; thresholds are module constants
so they can be recalibrated after a first look.
"""
import cv2
import numpy as np
from scipy import ndimage

# ultrasound
COLOR_SATURATION = 0.30      # HSV S above which a pixel counts as "colored"
COLOR_VALUE = 0.20
DOPPLER_BLOB_MIN_FRAC = 0.002  # colored component area (fraction of image) to count as Doppler, not annotation
DOPPLER_FRAC_THRESHOLD = 0.01
PAPER_DOPPLER_THRESHOLD = 0.02
PAPER_MARKER_LINES = 10
CALIPER_MIN = 2
DOTTED_MIN_DOTS = 8

# mammography
LABEL_MIN_AREA, LABEL_MAX_AREA = 20, 20000


def ultrasound_artifacts(img8, rgb, roi):
    h, w = img8.shape
    n = float(h * w)
    out = {}

    # Doppler / colored overlays: HSV saturation, split by blob size
    if rgb is not None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        colored = (hsv[..., 1] > COLOR_SATURATION * 255) & (hsv[..., 2] > COLOR_VALUE * 255)
        labels, count = ndimage.label(colored)
        sizes = ndimage.sum(colored, labels, index=np.arange(1, count + 1)) if count else np.array([])
        big = np.isin(labels, np.nonzero(sizes >= DOPPLER_BLOB_MIN_FRAC * n)[0] + 1) if count else colored & False
        out['color_frac'] = float(colored.mean())
        out['color_blob_frac_in_roi'] = float((big & roi).sum() / max(roi.sum(), 1))
        out['color_small_components'] = int((sizes < DOPPLER_BLOB_MIN_FRAC * n).sum()) if count else 0
        # the paper's rule, kept for comparability; it also fires on bright gray pixels
        r, b = rgb[..., 0], rgb[..., 2]
        out['paper_doppler_score'] = float(((r > 150).sum() + (b > 150).sum()) / n)
    else:
        out.update({'color_frac': 0.0, 'color_blob_frac_in_roi': 0.0, 'color_small_components': 0,
                    'paper_doppler_score': float(((img8 > 150).sum() * 2) / n)})
    out['doppler_flag'] = out['color_blob_frac_in_roi'] > DOPPLER_FRAC_THRESHOLD
    out['paper_doppler_flag'] = out['paper_doppler_score'] > PAPER_DOPPLER_THRESHOLD

    # measurement markers: the paper's Canny + Hough line count
    edges = cv2.Canny(img8, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=max(10, int(0.05 * min(h, w))), maxLineGap=3)
    out['hough_lines'] = 0 if lines is None else int(len(lines))
    out['paper_marker_flag'] = out['hough_lines'] > PAPER_MARKER_LINES

    # bright overlay pixels: near-white, or saturated colored graphics
    bright = img8 > 200
    if rgb is not None:
        bright |= (cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1] > 100) & (img8 > 120)
    # only keep pixels clearly brighter than their surroundings (overlays are
    # drawn on top of tissue; bright tissue/speckle is locally smooth)
    local = ndimage.uniform_filter(img8.astype(np.float32), size=15)
    bright &= img8.astype(np.float32) > local + 30
    components = _components(bright)

    out['caliper_count'] = _caliper_count(components)
    out['caliper_flag'] = out['caliper_count'] >= CALIPER_MIN
    out['dotted_lines'] = _dotted_lines(components, img8.shape)
    out['marker_flag'] = out['caliper_flag'] or out['dotted_lines'] >= 1
    out['text_lines'] = _text_lines(components)
    out['text_flag'] = out['text_lines'] >= 1

    # scan-region geometry (sector vs linear) and split-screen layout
    if roi.any():
        ys, xs = np.nonzero(roi)
        bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        out['roi_bbox_fill'] = float(roi.sum() / bbox_area)
        hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
        out['roi_solidity'] = float(roi.sum() / max(cv2.contourArea(hull), 1))
    content = ndimage.uniform_filter(img8.astype(np.float32), size=5) > 8
    column_profile = content.mean(axis=0)
    middle = column_profile[int(0.3 * w):int(0.7 * w)]
    out['dual_panel_flag'] = bool(middle.size and middle.min() < 0.02
                                  and column_profile[:int(0.3 * w)].max() > 0.3
                                  and column_profile[int(0.7 * w):].max() > 0.3)
    return out


def _components(mask, min_area=3, max_area=2000):
    labels, count = ndimage.label(mask)
    found = []
    for index, region in enumerate(ndimage.find_objects(labels), 1):
        if region is None:
            continue
        sub = labels[region] == index
        area = int(sub.sum())
        if not min_area <= area <= max_area:
            continue
        found.append({
            'sub': sub, 'area': area,
            'h': region[0].stop - region[0].start, 'w': region[1].stop - region[1].start,
            'cy': (region[0].start + region[0].stop) / 2, 'x0': region[1].start, 'x1': region[1].stop,
        })
    return found


def _is_cross(c):
    """'+'-shaped caliper glyph: small, 1-2 px strokes, a nearly full-width
    horizontal bar and full-height vertical bar through the middle, and no
    other wide rows/columns (which would make it a blob, not a cross)."""
    h, w, sub = c['h'], c['w'], c['sub']
    if not (7 <= h <= 30 and 7 <= w <= 30 and 0.6 <= w / h <= 1.6):
        return False
    if c['area'] > 2.2 * (h + w):
        return False
    wide_rows = (sub.sum(axis=1) >= 0.7 * w).sum()
    tall_cols = (sub.sum(axis=0) >= 0.7 * h).sum()
    if not (1 <= wide_rows <= 3 and 1 <= tall_cols <= 3):
        return False
    bar_row = int(np.argmax(sub.sum(axis=1)))
    bar_col = int(np.argmax(sub.sum(axis=0)))
    return abs(bar_row - h / 2) <= 0.25 * h and abs(bar_col - w / 2) <= 0.25 * w


def _caliper_count(components):
    """Crosses that have a same-size partner: calipers are placed in pairs with
    identical glyphs, whereas speckle that happens to look like a '+' is not."""
    crosses = [c for c in components if _is_cross(c)]
    return sum(1 for a in crosses
               if any(b is not a and abs(a['h'] - b['h']) <= 4 and abs(a['w'] - b['w']) <= 4 for b in crosses))


def _dotted_lines(components, shape):
    """Measurement lines drawn as evenly spaced tiny dots: Hough on a mask of
    1-6 px bright components, requiring many dots along a long segment."""
    dots = np.zeros(shape, np.uint8)
    for c in components:
        if c['area'] <= 6 and c['h'] <= 3 and (c['x1'] - c['x0']) <= 3:
            dots[int(c['cy']), (c['x0'] + c['x1']) // 2] = 255
    if dots.sum() < 8 * 255:
        return 0
    lines = cv2.HoughLinesP(dots, 1, np.pi / 180, threshold=DOTTED_MIN_DOTS,
                            minLineLength=max(30, int(0.08 * min(shape))), maxLineGap=8)
    return 0 if lines is None else int(len(lines))


def _text_lines(components):
    """Count horizontal runs of >= 3 glyph-sized components sharing a baseline."""
    glyphs = [c for c in components
              if 6 <= c['h'] <= 24 and 2 <= c['w'] <= 20 and 0.15 <= c['area'] / (c['h'] * c['w']) <= 0.85]
    glyphs.sort(key=lambda c: c['x0'])
    parent = list(range(len(glyphs)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for a in range(len(glyphs)):
        for b in range(a + 1, len(glyphs)):
            ga, gb = glyphs[a], glyphs[b]
            if gb['x0'] - ga['x1'] > 1.2 * ga['h']:
                break
            if abs(ga['cy'] - gb['cy']) <= 0.3 * ga['h'] and abs(ga['h'] - gb['h']) <= 0.4 * ga['h']:
                parent[find(b)] = find(a)
    sizes = {}
    for k in range(len(glyphs)):
        sizes[find(k)] = sizes.get(find(k), 0) + 1
    return sum(1 for size in sizes.values() if size >= 3)


def mammography_artifacts(img8, roi, header=None):
    h, w = img8.shape
    out = {'breast_area_frac': float(roi.mean())}
    ys, xs = np.nonzero(roi)
    if xs.size:
        out['breast_side_est'] = 'L' if xs.mean() < w / 2 else 'R'  # side of the image the breast occupies
    # burnt-in labels / markers: bright components clearly outside the breast
    outside = (img8 > 200) & ~ndimage.binary_dilation(roi, iterations=10)
    labels, count = ndimage.label(outside)
    if count:
        areas = ndimage.sum(outside, labels, index=np.arange(1, count + 1))
        keep = (areas >= LABEL_MIN_AREA) & (areas <= LABEL_MAX_AREA)
        out['label_components'] = int(keep.sum())
        out['label_area_frac'] = float(areas[keep].sum() / (h * w))
    else:
        out['label_components'], out['label_area_frac'] = 0, 0.0
    out['burnt_label_flag'] = out['label_components'] > 0
    # implant-like: a large share of breast pixels at near-maximum intensity
    if roi.any():
        out['near_max_frac_in_breast'] = float((img8[roi] >= 245).mean())
        out['implant_suspect_flag'] = out['near_max_frac_in_breast'] > 0.05
    if header:
        out['photometric'] = header.get('PhotometricInterpretation')
        laterality = header.get('ImageLaterality') or header.get('Laterality')
        if laterality:
            out['header_laterality'] = laterality
    return out


def mri_channel_stats(channels):
    """Per-channel (DCE phase / sequence) means and an enhancement summary."""
    means = channels.reshape(-1, channels.shape[-1]).astype(np.float32).mean(axis=0)
    out = {f'ch{i}_mean': float(m) for i, m in enumerate(means)}
    if means[0] > 0:
        out['enhancement_ratio'] = float(means.max() / means[0])
        out['peak_channel'] = int(np.argmax(means))
    return out
