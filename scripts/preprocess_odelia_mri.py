"""Convert ODELIA lesion breasts into BreaDM-compatible img9Se arrays for the
MRI branch (design decision D3 in docs/new_datasets_training_plan.md).

ODELIA ships case-level labels without lesion masks, so instead of BreaDM's
tumour-box crops each breast becomes one 256x256 whole-breast slice stack: the
central axial slice and its two neighbours, for Pre, Sub_1 and T2 -> 9 uint8
channels ordered
[Pre z-1, z, z+1, Sub_1 z-1, z, z+1, T2 z-1, z, z+1]. (Picking the slice with
the most Sub_1 enhancement was tried first and mostly selected the enhancing
heart / chest wall at the image edge, not lesions.) Each sequence is scaled
by its own 0.5-99.5 percentile window over the breast volume, except Sub_1:
its background is padded with a large negative value (-800 to -2000 by centre),
so it is windowed to [0, 99.5th percentile of positive voxels].

Only benign (Lesion 1) and malignant (Lesion 2) breasts are written. Splits
follow fold 0 of split_unilateral.csv with the RSH centre held out entirely
(written to test_rsh), matching scripts/reproduce_odelia.py. Reads the .npz
cache from `scripts/reproduce_odelia.py cache`.

Usage:
    python3 scripts/preprocess_odelia_mri.py --dest datasets/MRI/ODELIA/cls/img9Se
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ODELIA_ROOT = Path('/mnt/data/mmfm_datasets/MRI/odelia')
SEQUENCES = ('Pre', 'Sub_1', 'T2')
CLASS_NAMES = {1: 'Benign', 2: 'Malignant'}
OOD_CENTRE = 'RSH'


def _to_uint8(volume, low=0.5, high=99.5):
    lo, hi = np.percentile(volume, [low, high])
    if hi <= lo:
        return np.zeros(volume.shape, dtype=np.uint8)
    return (np.clip((volume - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)


def breast_stack(arrays):
    """(256, 256, 9) uint8 from a cached breast's (1, X, Y, Z) Pre/Sub_1/T2,
    oriented like the released MST's input (flip X and Y, rows = Y)."""
    volumes = {s: np.flip(arrays[s][0].astype(np.float32), axis=(0, 1)) for s in SEQUENCES}
    z = volumes['Pre'].shape[2] // 2
    channels = []
    for sequence in SEQUENCES:
        volume = volumes[sequence]
        if sequence == 'Sub_1':
            positive = volume[volume > 0]
            high = np.percentile(positive, 99.5) if positive.size else 1.0
            scaled = (np.clip(volume / high, 0, 1) * 255).round().astype(np.uint8)
        else:
            scaled = _to_uint8(volume)
        channels += [scaled[:, :, k].T for k in (z - 1, z, z + 1)]
    return np.stack(channels, axis=-1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dest', type=Path, default=Path('datasets/MRI/ODELIA/cls/img9Se'))
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args(argv)

    cache = ODELIA_ROOT / 'npz'
    meta = pd.read_csv(cache / 'meta.csv')
    splits = pd.read_csv(ODELIA_ROOT / 'split_unilateral.csv')
    split_of = dict(splits[splits['Fold'] == args.fold][['UID', 'Split']].itertuples(index=False))
    counts = {}
    for row in meta[meta['Lesion'].isin(CLASS_NAMES)].itertuples(index=False):
        split = 'test_rsh' if row.Institution == OOD_CENTRE else split_of.get(row.UID)
        if split is None:
            continue
        out = args.dest / split / CLASS_NAMES[row.Lesion] / row.UID / 'p-000.npy'
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, breast_stack(np.load(cache / f'{row.UID}.npz')))
        counts[(split, CLASS_NAMES[row.Lesion])] = counts.get((split, CLASS_NAMES[row.Lesion]), 0) + 1
    for key in sorted(counts):
        print(f'{key[0]:9s} {key[1]:9s} {counts[key]}')


if __name__ == '__main__':
    main()
