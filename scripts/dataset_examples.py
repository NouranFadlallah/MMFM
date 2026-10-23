"""Example images per dataset for the dataset audit report.

For every audited dataset, picks one benign and one malignant image that the
training loader uses (or, where a dataset has no benign/malignant split --
ISPY2 -- two images with different label names), contrast-stretches it,
outlines the lesion box/mask in red when the dataset provides one, and saves
a square tile to docs/dataset_report/figures/examples/<dataset>_<slot>.png.
Picks are seeded, so re-running reproduces the same tiles.

Usage: .venv/bin/python3 -I scripts/dataset_examples.py [dataset ...]
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit.sample import load_mask, load_native
from audit.sources import ADAPTERS, DATASET_ROOTS

SEED = 42
TILE = 384
OUT_DIR = Path('docs/dataset_report/figures/examples')
DATASETS = ['busbra', 'busi', 'busc', 'breast', 'bus_cot', 'bus_uclm',
            'mias', 'cdd_cesm', 'cmmd', 'bcsdbt', 'cbis_ddsm', 'dmid',
            'breamdm', 'ispy2', 'odelia', 'ambl']


def _stretch(gray):
    lo, hi = np.percentile(gray, (1, 99.5))
    return np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1)


def _tile(gray, outline=None):
    """gray: 2D float in [0,1]; outline: 2D bool mask in the same coordinates.
    Returns a TILE x TILE RGB image, scaled to fit (up or down) and
    letterboxed on black, with the outline's bounding box drawn in red."""
    image = Image.fromarray((gray * 255).astype(np.uint8)).convert('RGB')
    scale = TILE / max(image.size)
    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.BILINEAR)
    if outline is not None and outline.any():
        rows, cols = np.nonzero(outline)
        box = [cols.min() * scale, rows.min() * scale, (cols.max() + 1) * scale, (rows.max() + 1) * scale]
        ImageDraw.Draw(image).rectangle([int(v) for v in box], outline=(255, 40, 40), width=3)
    canvas = Image.new('RGB', (TILE, TILE))
    canvas.paste(image, ((TILE - image.width) // 2, (TILE - image.height) // 2))
    return canvas


def _pick(samples, rng):
    """[(slot, sample)]: one benign + one malignant used sample, or two with
    different label names when the dataset has only one training class."""
    used = [s for s in samples if s.used] or samples
    by_label = {0: [s for s in used if s.label == 0], 1: [s for s in used if s.label == 1]}
    if by_label[0] and by_label[1]:
        return [('benign', rng.choice(by_label[0])), ('malignant', rng.choice(by_label[1]))]
    names = sorted({s.label_name for s in used})
    if len(names) >= 2:
        return [(f'ex{k}', rng.choice([s for s in used if s.label_name == n])) for k, n in enumerate(names[:2])]
    return [(f'ex{k}', s) for k, s in enumerate(rng.sample(used, 2))]


def sample_tiles(dataset, rng):
    samples = ADAPTERS[dataset](DATASET_ROOTS[dataset])
    out = []
    for slot, sample in _pick(samples, rng):
        loaded = load_native(sample)
        mask = load_mask(sample, loaded.gray.shape)
        out.append((slot, sample.label_name, sample.sample_id, _tile(_stretch(loaded.gray), mask)))
    return out


def _prepared_frame(dataset):
    """(image, label) frame of the 8-bit PNG cache training reads, for the
    datasets whose native DICOMs are no longer on this machine."""
    import pandas as pd
    if dataset == 'cmmd':
        from training.train import _cmmd_frame
        frames = _cmmd_frame(Path(DATASET_ROOTS['cmmd']), Path('datasets/mammography/cmmd_clinicaldata.xlsx'),
                             Path('datasets/mammography/cmmd_png'), SEED, 0.1)
        return pd.concat(frames)[['image', 'label']]
    if dataset == 'cbis_ddsm':
        meta = pd.read_csv('datasets/mammography/cbis_ddsm_png/metadata.csv')
        meta = meta[meta['pathology'].isin(['BENIGN', 'MALIGNANT'])]
        return meta.assign(label=(meta['pathology'] == 'MALIGNANT').astype(int))[['image', 'label']]
    raise KeyError(dataset)


def prepared_tiles(dataset, rng):
    frame = _prepared_frame(dataset)
    out = []
    for slot, label in (('benign', 0), ('malignant', 1)):
        image = rng.choice(sorted(frame.loc[frame['label'] == label, 'image']))
        gray = np.asarray(Image.open(image).convert('L'), dtype=np.float32)
        out.append((slot, slot, Path(image).stem, _tile(_stretch(gray))))
    return out


ODELIA_CACHE = Path('/mnt/data/mmfm_datasets/MRI/odelia_prepared/npz')


def odelia_tiles(rng):
    """ODELIA ships parquet volumes, not per-image files: show the
    subtraction (Sub_1) maximum-intensity projection of one benign and one
    malignant breast from the per-breast .npz cache that
    scripts/reproduce_odelia.py builds."""
    import pandas as pd

    meta = pd.read_csv(ODELIA_CACHE / 'meta.csv')
    out = []
    for slot, code in (('benign', 1), ('malignant', 2)):
        uid = rng.choice(sorted(meta.loc[meta['Lesion'] == code, 'UID']))
        volume = np.squeeze(np.load(ODELIA_CACHE / f'{uid}.npz')['Sub_1']).astype(np.float32)  # (X, Y, Z)
        mip = np.rot90(volume.max(axis=-1))
        out.append((slot, slot, uid, _tile(_stretch(mip))))
    return out


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for dataset in sys.argv[1:] or DATASETS:
        rng = random.Random(f'{SEED}-{dataset}')
        try:
            if dataset == 'odelia':
                tiles = odelia_tiles(rng)
            elif dataset in ('cmmd', 'cbis_ddsm'):
                tiles = prepared_tiles(dataset, rng)
            else:
                tiles = sample_tiles(dataset, rng)
        except Exception as error:  # a dataset missing on this machine shouldn't stop the rest
            print(f'{dataset}: skipped ({type(error).__name__}: {error})', flush=True)
            continue
        manifest[dataset] = []
        for slot, label_name, sample_id, tile in tiles:
            tile.save(OUT_DIR / f'{dataset}_{slot}.png')
            manifest[dataset].append({'slot': slot, 'label_name': label_name, 'sample_id': sample_id})
        print(f'{dataset}: {[(t[0], t[1], t[2]) for t in tiles]}', flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
