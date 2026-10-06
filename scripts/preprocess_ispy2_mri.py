"""Convert the BreastDCEDL-ISPY2 NIfTI release into BreaDM-compatible img9Se arrays.

I-SPY2 is a neoadjuvant-therapy trial: every case is pre-treatment, biopsy-confirmed
malignant, and the release ships no benign/malignant label file (see
docs/dataset_reproduction_plan.md section 6 for the existing BreaDM MRI branch this
augments). This script only emits Malignant-class samples, matching how the dataset
is actually labeled, and writes them into the same on-disk layout BreaDMDataset
already reads (`cls/img9Se/{train,val}/Malignant/<patient>/p-XXX.npy`, patches
cropped to the tumor bounding box, 9 DCE-phase channels, uint8).

Usage:
    python3 scripts/preprocess_ispy2_mri.py \
        --source "/mnt/data/mmfm_datasets/MRI/PKG - BreastDCEDL_ISPY2/BreastDCEDL_ISPY2" \
        --dest datasets/MRI/ISPY2/cls/img9Se
"""
import argparse
import random
import re
from pathlib import Path

import nibabel as nib
import numpy as np

NUM_PHASES = 9
AQC_RE = re.compile(r'dce_aqc_(\d+)\.nii\.gz$')
BBOX_MARGIN = 6
MIN_TUMOR_PIXELS = 4


def _sorted_dce_paths(dce_dir):
    paths = []
    for path in dce_dir.glob('*_dce_aqc_*.nii.gz'):
        match = AQC_RE.search(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    paths.sort(key=lambda item: item[0])
    return [path for _, path in paths]


def _phase_indices(count):
    return np.round(np.linspace(0, count - 1, NUM_PHASES)).astype(int)


def _intensity_range(volume):
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(nonzero, [1, 99])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _to_uint8(patch, low, high):
    scaled = (patch.astype(np.float32) - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0) * 255.0
    return scaled.astype(np.uint8)


def process_patient(patient_dir):
    """Yield one (h, w, 9) uint8 array per axial slice that contains tumor."""
    mask_paths = list((patient_dir / 'mask').glob('*_mask.nii.gz'))
    dce_paths = _sorted_dce_paths(patient_dir / 'dce')
    if not mask_paths or not dce_paths:
        return

    mask = nib.load(mask_paths[0]).get_fdata() > 0
    if not mask.any():
        return

    phase_idx = _phase_indices(len(dce_paths))
    phase_volumes = [nib.load(dce_paths[i]).get_fdata(dtype=np.float32) for i in phase_idx]
    low, high = _intensity_range(phase_volumes[0])

    axial_slices = np.where(mask.any(axis=(1, 2)))[0]
    height, width = mask.shape[1], mask.shape[2]
    for z in axial_slices:
        slice_mask = mask[z]
        if slice_mask.sum() < MIN_TUMOR_PIXELS:
            continue
        rows, cols = np.where(slice_mask)
        r0, r1 = max(rows.min() - BBOX_MARGIN, 0), min(rows.max() + BBOX_MARGIN + 1, height)
        c0, c1 = max(cols.min() - BBOX_MARGIN, 0), min(cols.max() + BBOX_MARGIN + 1, width)

        channels = [_to_uint8(volume[z, r0:r1, c0:c1], low, high) for volume in phase_volumes]
        yield int(z), np.stack(channels, axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                         help='Path to the BreastDCEDL_ISPY2 patient-folder root '
                              '(the dir containing ISPY2-*/ACRIN-6698-* subfolders).')
    parser.add_argument('--dest', type=Path, default=Path('datasets/MRI/ISPY2/cls/img9Se'))
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=None, help='Process only the first N patients (smoke test).')
    args = parser.parse_args()

    patient_dirs = sorted(p for p in args.source.iterdir() if p.is_dir())
    if args.limit:
        patient_dirs = patient_dirs[:args.limit]

    rng = random.Random(args.seed)
    shuffled = patient_dirs[:]
    rng.shuffle(shuffled)
    num_val = max(1, int(len(shuffled) * args.val_fraction))
    val_patients = set(p.name for p in shuffled[:num_val])

    num_patients_written = 0
    num_slices_written = 0
    num_skipped = 0
    for i, patient_dir in enumerate(patient_dirs, 1):
        split = 'val' if patient_dir.name in val_patients else 'train'
        out_dir = args.dest / split / 'Malignant' / patient_dir.name
        slices = list(process_patient(patient_dir))
        if not slices:
            num_skipped += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for z, array in slices:
            np.save(out_dir / f'p-{z:03d}.npy', array)
        num_patients_written += 1
        num_slices_written += len(slices)
        if i % 25 == 0 or i == len(patient_dirs):
            print(f'[{i}/{len(patient_dirs)}] patients written={num_patients_written} '
                  f'slices={num_slices_written} skipped={num_skipped}')

    print(f'Done. {num_patients_written} patients, {num_slices_written} slices, '
          f'{num_skipped} skipped (no mask/tumor). Output: {args.dest}')


if __name__ == '__main__':
    main()
