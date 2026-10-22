"""Helpers around data/us_preprocessing.py.

  reference-hist  pooled 256-bin histogram of a dataset's *training* images, the
                  reference for hist_match / perc_hm (a training-split reference
                  keeps validation and test images out of the preprocessing)
  markers         synthetic scanner-annotation dataset from BrEaST (clean image +
                  lesion mask -> marked image), written as marked/ clean/ mask/
                  for scripts/train_cyclegan.py --paired
  preview         before/after grid of a step pipeline on a few images, like the
                  notebook's display_* cells

Usage:
  python3 scripts/preprocess_ultrasound.py reference-hist --dataset busbra --out datasets/ultrasound/busbra_train_hist.npy
  python3 scripts/preprocess_ultrasound.py markers --out datasets/ultrasound/markers
  python3 scripts/preprocess_ultrasound.py preview --dataset breast --steps perc_hm:reference=datasets/ultrasound/busbra_train_hist.npy line_removal:line_length=2;min_area=3
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.us_preprocessing import Pipeline, add_synthetic_markers, histogram_of

BREAST_ROOT = Path('datasets/ultrasound/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks')


def _training_images(dataset, seed=42):
    from scripts.train_cyclegan import domain_samples

    return [path for path, _ in domain_samples(dataset, seed)]


def reference_hist(args):
    paths = _training_images(args.dataset)
    histogram = histogram_of(np.asarray(Image.open(p).convert('L')) for p in paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, histogram)
    print(f'{args.dataset}: pooled histogram of {len(paths)} training images -> {args.out}')


def markers(args):
    """BrEaST images (case_XXX.png) with their _tumor masks -> marked/ clean/ mask/."""
    rng = np.random.default_rng(args.seed)
    for name in ('marked', 'clean', 'mask'):
        (args.out / name).mkdir(parents=True, exist_ok=True)
    written = 0
    for image_path in sorted(p for p in BREAST_ROOT.glob('*.png') if '_' not in p.stem):
        mask_path = image_path.with_name(f'{image_path.stem}_tumor.png')
        if not mask_path.exists():
            continue
        clean = np.asarray(Image.open(image_path).convert('L'))
        mask = np.asarray(Image.open(mask_path).convert('L'))
        Image.fromarray(add_synthetic_markers(clean, mask, rng)).save(args.out / 'marked' / image_path.name)
        Image.fromarray(clean).save(args.out / 'clean' / image_path.name)
        Image.fromarray(mask).save(args.out / 'mask' / image_path.name)
        written += 1
    print(f'{written} marked/clean/mask triples -> {args.out}')


def preview(args):
    paths = _training_images(args.dataset)
    random.Random(args.seed).shuffle(paths)
    pipeline = Pipeline(args.steps)
    rows = []
    for path in paths[:args.count]:
        before = Image.open(path).convert('L').resize((args.size, args.size))
        after = pipeline(Image.open(path).convert('L')).resize((args.size, args.size))
        rows.append(np.hstack([np.asarray(before), np.asarray(after)]))
    out = args.out or Path(f'{args.dataset}_preprocess_preview.png')
    Image.fromarray(np.vstack(rows)).save(out)
    print(f'{len(rows)} before|after rows for {args.steps} -> {out}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('reference-hist')
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('markers')
    p.add_argument('--out', type=Path, default=Path('datasets/ultrasound/markers'))
    p.add_argument('--seed', type=int, default=42)
    p = sub.add_parser('preview')
    p.add_argument('--dataset', required=True)
    p.add_argument('--steps', nargs='+', required=True)
    p.add_argument('--count', type=int, default=6)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, default=None)
    args = parser.parse_args(argv)
    {'reference-hist': reference_hist, 'markers': markers, 'preview': preview}[args.command](args)


if __name__ == '__main__':
    main()
