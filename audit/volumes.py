"""Case-level (3D) analysis for MRI sources whose raw form is a volume:
BreastDCEDL-ISPY2 NIfTI (DCE phases + tumor mask) and BreaDM seg3D arrays.
One row per patient: geometry, tumor size, kinetic curve, lesion CNR.
"""
import re
from pathlib import Path

import numpy as np
from scipy import ndimage

AQC_RE = re.compile(r'dce_aqc_(\d+)\.nii\.gz$')
NUM_SAMPLED_PHASES = 9  # scripts/preprocess_ispy2_mri.py NUM_PHASES


def _tumor_stats(mask, spacing):
    labels, components = ndimage.label(mask)
    slices = np.where(mask.any(axis=(1, 2)))[0]
    voxels = int(mask.sum())
    return {
        'tumor_voxels': voxels,
        'tumor_volume_ml': voxels * float(np.prod(spacing)) / 1000.0 if spacing is not None else float('nan'),
        'tumor_components': int(components),
        'tumor_slices_axis0': int(slices.size),
    }


def _kinetics(phase_volumes, mask):
    ring = ndimage.binary_dilation(mask, iterations=3) & ~mask
    means = [float(volume[mask].mean()) for volume in phase_volumes]
    out = {f'tumor_mean_phase{i}': m for i, m in enumerate(means)}
    pre = means[0]
    peak_index = int(np.argmax(means))
    out['peak_phase'] = peak_index
    out['enhancement_ratio'] = means[peak_index] / pre if pre > 0 else float('nan')
    out['washout'] = (means[peak_index] - means[-1]) / means[peak_index] if means[peak_index] > 0 else float('nan')
    peak = phase_volumes[peak_index]
    if ring.any():
        ring_std = float(peak[ring].std())
        out['lesion_cnr_peak'] = abs(means[peak_index] - float(peak[ring].mean())) / ring_std if ring_std > 0 else float('nan')
    nonzero = phase_volumes[0][phase_volumes[0] > 0]
    if nonzero.size:
        out['pre_p1'], out['pre_p99'] = (float(v) for v in np.percentile(nonzero, [1, 99]))
    return out


def analyze_ispy2_case(patient_dir):
    import nibabel as nib

    patient_dir = Path(patient_dir)
    row = {'group': patient_dir.name}
    dce = sorted(
        ((int(m.group(1)), p) for p in (patient_dir / 'dce').glob('*_dce_aqc_*.nii.gz')
         if (m := AQC_RE.search(p.name))),
        key=lambda item: item[0],
    )
    masks = list((patient_dir / 'mask').glob('*_mask.nii.gz'))
    row['n_phases'] = len(dce)
    row['has_mask'] = bool(masks)
    # the preprocessing samples 9 phases with linspace; fewer than 9 means repeated channels
    row['phases_repeated_in_img9'] = 0 < len(dce) < NUM_SAMPLED_PHASES
    if not dce:
        return row
    first = nib.load(dce[0][1])
    row['shape'] = 'x'.join(str(s) for s in first.shape)
    spacing = tuple(float(z) for z in first.header.get_zooms()[:3])
    row['spacing'] = 'x'.join(f'{s:.3g}' for s in spacing)
    row['slice_thickness'] = spacing[0]
    row['dtype'] = str(first.get_data_dtype())
    if not masks:
        return row
    mask = nib.load(masks[0]).get_fdata() > 0
    row['mask_shape_matches'] = mask.shape == first.shape
    row.update(_tumor_stats(mask, spacing))
    if mask.any() and row['mask_shape_matches']:
        volumes = [nib.load(p).get_fdata(dtype=np.float32) for _, p in dce]
        row.update(_kinetics(volumes, mask))
    return row


def analyze_breadm_seg3d_case(image_dir):
    """`image_dir` = seg3D/<split>/images/<patient>; labels live in the sibling
    labels/<patient> folder. Arrays are (H, W, Z), stored as float64 0-255."""
    image_dir = Path(image_dir)
    label_dir = image_dir.parents[1] / 'labels' / image_dir.name
    row = {'group': image_dir.name, 'split': image_dir.parents[1].name,
           'label_name': 'Benign' if '-Be-' in image_dir.name else 'Malignant' if '-Ma-' in image_dir.name else 'unknown'}
    sequences = {p.stem: p for p in image_dir.glob('*.npy')}
    row['sequences'] = ','.join(sorted(sequences))
    label_path = label_dir / 'SUB2.npy'
    if not label_path.exists():
        label_path = next(iter(sorted(label_dir.glob('*.npy'))), None)
    if 'SUB2' not in sequences or label_path is None:
        return row
    sub = np.load(sequences['SUB2'])
    row['shape'] = 'x'.join(str(s) for s in sub.shape)
    row['dtype'] = str(sub.dtype)
    row['stored_bytes'] = int(sub.nbytes)
    row['integer_valued'] = bool(np.all(np.mod(sub, 1) == 0))
    # (H, W, Z) -> (Z, H, W) so tumor stats count slices along the first axis
    mask = np.moveaxis(np.load(label_path) > 127, -1, 0)
    row.update(_tumor_stats(mask, None))
    if mask.any():
        sub_z = np.moveaxis(sub, -1, 0)
        ring = ndimage.binary_dilation(mask, iterations=3) & ~mask
        if ring.any() and sub_z[ring].std() > 0:
            row['lesion_cnr_sub2'] = float(abs(sub_z[mask].mean() - sub_z[ring].mean()) / sub_z[ring].std())
        if 'VIBRANT' in sequences and 'VIBRANT+C2' in sequences:
            pre = np.moveaxis(np.load(sequences['VIBRANT']), -1, 0)[mask].mean()
            post = np.moveaxis(np.load(sequences['VIBRANT+C2']), -1, 0)[mask].mean()
            row['enhancement_ratio'] = float(post / pre) if pre > 0 else float('nan')
    return row
