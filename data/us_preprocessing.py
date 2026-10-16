"""Optional ultrasound preprocessing steps, ported from the thesis preprocessing
notebook (Dataprocessing.ipynb) so they can be switched on per training run.

Every step maps a 2D uint8 grayscale array to a 2D uint8 array of the same
shape, so lesion masks and crop boxes stay valid. Steps are chained with
`build_pipeline` from a spec list such as

    ['intensity_window:low=1;high=99', 'hist_match:reference=ref_hist.npy',
     'line_removal:line_length=2;min_area=3']

and wrapped around an existing image transform with `PreprocessedTransform`,
which caches each processed image on disk (fuzzy c-means and CycleGAN steps are
far too slow to redo every epoch).

Steps:
  hist_eq           global histogram equalization (cv2.equalizeHist)
  hist_match        CDF matching to a reference image or a 256-bin histogram (.npy)
  hist_spec         fuzzy c-means + generalized-gamma histogram specification
  intensity_window  percentile window rescaled to 0-255
  perc_hm           intensity_window then hist_match (the notebook's "Perc-HM")
  line_removal      bright and dark thin lines: top-hat/black-hat + Otsu + inpaint
  cross_removal     bright caliper crosses only, filtered by component area
  cyclegan          a trained MMFM CycleGAN generator (models/cyclegan.py)
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

LEVELS = 256


# ----------------------------------------------------------------- histogram steps

def hist_eq(image):
    import cv2

    return cv2.equalizeHist(image)


def _cdf(histogram):
    cdf = np.cumsum(np.asarray(histogram, dtype=np.float64))
    return cdf / cdf[-1]


def histogram_of(images):
    """Pooled 256-bin histogram of one or more uint8 images (a hist_match reference)."""
    total = np.zeros(LEVELS, dtype=np.float64)
    for image in images:
        total += np.bincount(np.asarray(image, dtype=np.uint8).ravel(), minlength=LEVELS)
    return total


def _load_reference(reference):
    """A reference given as a 256-bin histogram (.npy) or as an image path."""
    if isinstance(reference, np.ndarray):
        return reference if reference.shape == (LEVELS,) else histogram_of([reference])
    path = Path(reference)
    if path.suffix == '.npy':
        return np.load(path)
    return histogram_of([np.asarray(Image.open(path).convert('L'))])


def hist_match(image, reference):
    """Map grey levels so the image's CDF follows the reference CDF (the
    single-channel equivalent of skimage.exposure.match_histograms). A pooled
    reference histogram over a whole training set is steadier than the
    notebook's single randomly chosen reference image."""
    # interpolate over occupied grey levels only, as skimage does: empty bins
    # make the CDF flat and the inverse ambiguous
    source_counts = np.bincount(image.ravel(), minlength=LEVELS)
    reference_counts = np.asarray(_load_reference(reference), dtype=np.float64)
    source_levels, reference_levels = np.flatnonzero(source_counts), np.flatnonzero(reference_counts)
    source_quantiles = _cdf(source_counts[source_levels])
    reference_quantiles = _cdf(reference_counts[reference_levels])
    lut = np.zeros(LEVELS)
    lut[source_levels] = np.interp(source_quantiles, reference_quantiles, reference_levels)
    return np.clip(np.round(lut), 0, LEVELS - 1).astype(np.uint8)[image]


def _fuzzy_cmeans_1d(levels, weights, clusters=3, fuzziness=2.0, tol=0.005, max_iter=1000, seed=0):
    """Fuzzy c-means on grey levels weighted by their pixel counts: the same
    objective as clustering every pixel (as skfuzzy.cmeans does in the notebook)
    at 256 points instead of H*W. Returns (centres, memberships[c, 256])."""
    rng = np.random.default_rng(seed)
    membership = rng.random((clusters, len(levels)))
    membership /= membership.sum(axis=0)
    for _ in range(max_iter):
        um = membership ** fuzziness * weights
        centres = um @ levels / np.maximum(um.sum(axis=1), 1e-12)
        distance = np.abs(levels[None, :] - centres[:, None]) + 1e-12
        updated = 1.0 / (distance ** (2.0 / (fuzziness - 1.0)))
        updated /= updated.sum(axis=0)
        converged = np.abs(updated - membership).max() < tol
        membership = updated
        if converged:
            break
    return centres, membership


# (s, m, sigma) generalized-gamma prototypes for hypo-, normal- and
# hyper-echogenic tissue: the notebook's "smoother, less speckle" settings
DEFAULT_PROTOTYPES = ((0.85, 1.5, 15.0), (0.95, 1.8, 60.0), (0.85, 2.5, 150.0))


def hist_spec(image, prototypes=DEFAULT_PROTOTYPES, fuzziness=2.0):
    """Fuzzy histogram specification: split grey levels into hypo / normal /
    hyper-echogenic clusters with fuzzy c-means, map each cluster's
    membership-weighted CDF onto a generalized-gamma CDF, and blend the mapped
    images by membership. Clusters are sorted by centre intensity so the first
    prototype always goes to the darkest cluster; the notebook paired
    prototypes with skfuzzy's arbitrary cluster order."""
    from scipy.special import gammainc

    levels = np.arange(LEVELS, dtype=np.float64)
    counts = np.bincount(image.ravel(), minlength=LEVELS).astype(np.float64)
    centres, membership = _fuzzy_cmeans_1d(levels, counts, clusters=len(prototypes), fuzziness=fuzziness)
    membership = membership[np.argsort(centres)]

    blended = np.zeros(LEVELS, dtype=np.float64)
    for cluster_membership, (s, m, sigma) in zip(membership, prototypes):
        target_cdf = (LEVELS - 1) * _cdf(counts * cluster_membership)
        reference_cdf = (LEVELS - 1) * gammainc(m, (levels / sigma) ** s)
        lut = np.abs(reference_cdf[None, :] - target_cdf[:, None]).argmin(axis=1)
        blended += lut * cluster_membership
    return np.clip(np.round(blended), 0, LEVELS - 1).astype(np.uint8)[image]


def intensity_window(image, low=0.5, high=99.5):
    """Rescale the [low, high] percentile window to 0-255 with clipping. The
    notebook's MONAI ScaleIntensityRanged(a=0..255 -> b=0..255, clip) is the
    identity on uint8 input, so a percentile window is used here instead."""
    lo, hi = np.percentile(image, [float(low), float(high)])
    if hi <= lo:
        return image.copy()
    scaled = (image.astype(np.float64) - lo) / (hi - lo)
    return np.clip(np.round(scaled * (LEVELS - 1)), 0, LEVELS - 1).astype(np.uint8)


def perc_hm(image, reference, low=0.5, high=99.5):
    return hist_match(intensity_window(image, low, high), reference)


# ----------------------------------------------------------------- annotation removal

def _component_filter(mask, min_area=0, max_area=None):
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = (areas >= min_area) & (areas <= (max_area if max_area is not None else np.inf))
    return np.isin(labels, np.flatnonzero(keep) + 1)


def _otsu_mask(response):
    import cv2

    if response.max() == 0:
        return np.zeros(response.shape, dtype=bool)
    _, mask = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask > 0


def _inpaint(image, mask, radius):
    import cv2

    mask = mask.astype(np.uint8) * 255
    if not mask.any():
        return image.copy()
    return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)


def line_removal(image, line_length=4, min_area=5, radius=3):
    """Thin horizontal/vertical lines, bright (top-hat) and dark (black-hat),
    Otsu-thresholded, small components dropped, closed + dilated, inpainted
    (the notebook's remove_lines; the combined pipeline used line_length=2,
    min_area=3).

    Caution: on speckled B-mode images Otsu on the top-hat response flags most
    of the texture, so this inpaints 15-91% of the pixels on BUS-BRA and
    BrEaST samples (checked 2026-10-16) at either setting. Preview it before
    training with it; cross_removal or a marker-removal CycleGAN alters far less."""
    import cv2

    line_length, min_area = int(line_length), int(min_area)
    elements = (cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, 1)),
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_length)))
    mask = np.zeros(image.shape, dtype=bool)
    for op in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
        for element in elements:
            mask |= _otsu_mask(cv2.morphologyEx(image, op, element))
    mask = _component_filter(mask, min_area=min_area).astype(np.uint8)
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse), ellipse)
    return _inpaint(image, mask > 0, int(radius))


def cross_removal(image, line_length=2, min_area=8, max_area=120, min_peak=40, radius=3):
    """Bright caliper crosses only (top-hat), keeping components between
    min_area and max_area pixels so anatomy is not smudged; skipped when the
    top-hat response is too dim to threshold reliably (the notebook's
    refined remove_white_cross)."""
    import cv2

    line_length = int(line_length)
    elements = (cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, 1)),
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_length)))
    mask = np.zeros(image.shape, dtype=bool)
    for element in elements:
        response = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, element)
        if response.max() >= int(min_peak):
            mask |= _otsu_mask(response)
    mask = _component_filter(mask, min_area=int(min_area), max_area=int(max_area))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, disk), disk)
    return _inpaint(image, mask > 0, int(radius))


# ----------------------------------------------------------------- synthetic markers

def add_synthetic_markers(image, mask=None, rng=None):
    """Draw scanner-style annotations onto a clean image: settings text in a
    random corner (p=0.7), caliper crosses joined by a line (p=0.6), a
    measurement box (p=0.5) and an arrow (p=0.3), placed around the lesion
    centroid when a mask is given. Pairs of (marked, clean) images train a
    marker-removal generator."""
    import cv2

    rng = rng or np.random.default_rng()
    marked = image.copy()
    height, width = marked.shape[:2]
    if mask is not None and np.asarray(mask).any():
        moments = cv2.moments(np.asarray(mask, dtype=np.uint8))
        target = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    else:
        target = (width // 2, height // 2)

    def jitter(low, high):
        return int(rng.integers(low, high + 1))

    if rng.random() > 0.3:
        font = cv2.FONT_HERSHEY_TRIPLEX
        settings = [f'G: {jitter(65, 85)}%', f'Freq: {rng.choice([10.0, 12.0, 15.0])} MHz',
                    f'DR: {jitter(55, 75)}', 'Breast / L12-5']
        corner = jitter(0, 3)
        for i, text in enumerate(settings):
            text_width = cv2.getTextSize(text, font, 0.6, 1)[0][0]
            x = 15 if corner in (0, 2) else width - text_width - 15
            y = 30 + i * 25 if corner in (0, 1) else height - (len(settings) - i) * 25
            cv2.putText(marked, text, (x, y), font, 0.6, 255, 1, cv2.LINE_AA)
    if rng.random() > 0.4:
        p1 = (target[0] + jitter(-20, 20), target[1] + jitter(-20, 20))
        p2 = (p1[0] + jitter(-50, 50), p1[1] + jitter(-50, 50))
        cv2.drawMarker(marked, p1, 255, cv2.MARKER_CROSS, 12, 1)
        cv2.drawMarker(marked, p2, 255, cv2.MARKER_CROSS, 12, 1)
        cv2.line(marked, p1, p2, 255, 1)
    if rng.random() > 0.5:
        box_w, box_h = jitter(40, 80), jitter(30, 60)
        top_left = (target[0] - box_w // 2, target[1] - box_h // 2)
        bottom_right = (top_left[0] + box_w, top_left[1] + box_h)
        cv2.rectangle(marked, top_left, bottom_right, 255, 1, cv2.LINE_4)
        for corner_point in (top_left, bottom_right, (top_left[0], bottom_right[1]), (bottom_right[0], top_left[1])):
            cv2.drawMarker(marked, corner_point, 255, cv2.MARKER_TILTED_CROSS, 5, 1)
    if rng.random() > 0.7:
        start = (target[0] + jitter(30, 60), target[1] + jitter(30, 60))
        cv2.arrowedLine(marked, start, target, 255, 1, tipLength=0.2)
    return marked


# ----------------------------------------------------------------- CycleGAN step

_GENERATORS = {}


def cyclegan(image, checkpoint, direction='A2B', device='cpu', size=256, flip_tta=None):
    """Translate with a trained MMFM CycleGAN generator: resize to the training
    size, map to [-1, 1], run (averaging with the horizontally flipped input
    for the NCIA variant, as its inference script does), resize back to the
    original shape."""
    import torch
    from models.cyclegan import load_generator

    key = (str(checkpoint), direction, device)
    if key not in _GENERATORS:
        _GENERATORS[key] = load_generator(checkpoint, direction, map_location=device)
    generator, variant = _GENERATORS[key]
    flip_tta = variant == 'ncia' if flip_tta is None else str(flip_tta).lower() in ('1', 'true')
    size = int(size)
    resized = np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32)
    # junyanz's defaults train 3-channel generators even on grayscale data
    tensor = torch.from_numpy(resized / 127.5 - 1.0)[None, None].expand(1, generator.input_nc, -1, -1).to(device)
    with torch.no_grad():
        output = generator(tensor)
        if flip_tta:
            output = (output + generator(tensor.flip(-1)).flip(-1)) / 2
    output = ((output[0].mean(dim=0).clamp(-1, 1).cpu().numpy() + 1.0) * 127.5).round().astype(np.uint8)
    return np.asarray(Image.fromarray(output).resize((image.shape[1], image.shape[0]), Image.Resampling.BICUBIC))


# ----------------------------------------------------------------- pipelines

STEPS = {
    'hist_eq': hist_eq,
    'hist_match': hist_match,
    'hist_spec': hist_spec,
    'intensity_window': intensity_window,
    'perc_hm': perc_hm,
    'line_removal': line_removal,
    'cross_removal': cross_removal,
    'cyclegan': cyclegan,
}


def parse_step(spec):
    """'name:key=value;key=value' -> (name, {key: value})."""
    name, _, params = spec.partition(':')
    if name not in STEPS:
        raise ValueError(f'unknown ultrasound preprocessing step {name!r}; choose from {sorted(STEPS)}')
    kwargs = dict(item.split('=', 1) for item in params.split(';') if item)
    return name, kwargs


class Pipeline:
    def __init__(self, specs):
        self.specs = list(specs)
        self.steps = [parse_step(spec) for spec in self.specs]

    def __call__(self, image):
        array = np.asarray(image.convert('L') if isinstance(image, Image.Image) else image, dtype=np.uint8)
        for name, kwargs in self.steps:
            array = STEPS[name](array, **kwargs)
        return Image.fromarray(array)

    def key(self):
        """Stable hash of the step specs plus any referenced files' contents
        (a retrained CycleGAN or a new reference histogram gets a fresh cache)."""
        digest = hashlib.sha1(json.dumps(self.specs).encode())
        for _, kwargs in self.steps:
            for value in kwargs.values():
                if Path(value).is_file():
                    digest.update(Path(value).read_bytes())
        return digest.hexdigest()[:12]


def build_pipeline(specs):
    return Pipeline(specs) if specs else None


class PreprocessedTransform:
    """Run a Pipeline on the PIL image before an existing image transform,
    caching each processed image as a PNG under cache_dir/<pipeline key>/."""

    def __init__(self, pipeline, transform, cache_dir=Path('datasets/ultrasound/_preprocessed')):
        self.pipeline = pipeline
        self.transform = transform
        self.cache_dir = Path(cache_dir) / pipeline.key()

    def preprocess(self, image):
        source = getattr(image, 'filename', '') or ''
        if not source:
            return self.pipeline(image)
        cached = self.cache_dir / (hashlib.sha1(str(Path(source).resolve()).encode()).hexdigest() + '.png')
        if cached.exists():
            return Image.open(cached)
        processed = self.pipeline(image)
        cached.parent.mkdir(parents=True, exist_ok=True)
        processed.save(cached)
        return processed

    def __call__(self, image, *args):
        return self.transform(self.preprocess(image), *args)
