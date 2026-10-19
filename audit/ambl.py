"""Advanced-MRI-Breast-Lesions (AMBL) audit: the 99 patients with a DICOM SEG.

Raw layout (IDC download, data/sources/series/ambl_seg99.txt):
<patient>/<study>/<series>/*.dcm with three series per patient: the original
"AX Sen Vibrant MultiPhase" DCE (5 temporal positions x ~116 axial slices,
single-frame files), "Ax T2 FSE", and a multi-frame SEG ("ROI").

The SEG references the *registered* DCE series, which shares the original's
slice grid (positions and orientation agree to float32 rounding), so masks are
placed on the original DCE by slice position. Every segment is labelled
"Mass"; benign / malignant comes from the clinical XLSX, whose data dictionary
says SEG segments list all tumours first, then all benign lesions.
`side_agreement` tests that ordering against each lesion's L/R position.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from audit.sample import Sample

CLINICAL = Path('/mnt/data/mmfm_datasets/MRI/ambl/clinical/Advanced-MRI-Breast-Lesions-DA-Clinical-Sep2024.xlsx')
POST_NAT_REFERRAL = 6   # "post treatment - response to therapy assessment"


def clinical_lesions(path=CLINICAL):
    """{patient: [(label 1 tumour / 0 benign, side 'L'/'R'/None, pathology code), ...]} in report order,
    plus {patient: referral reason}."""
    frame = pd.read_excel(path, sheet_name=0, header=1).set_index('Patient ID')
    lesions, referral = {}, {}
    for patient, row in frame.iterrows():
        items = []
        for k in range(1, 7):
            label = row.get(f'tumor/benign{k}')
            if pd.isna(label) or label not in (0, 1):
                continue
            position = str(row.get(f'pos{k}', '') or '')
            side = position[0].upper() if position[:1].upper() in ('L', 'R') else None
            items.append((int(label), side, row.get(f'pathology{k}')))
        lesions[patient] = items
        referral[patient] = row.get('reason for referral ID#')
    return lesions, referral


def segment_labels(items, n_segments, ordering='tumours_first'):
    """Clinical (label, side, pathology) for each SEG segment under an ordering hypothesis."""
    if ordering == 'tumours_first':
        items = [i for i in items if i[0] == 1] + [i for i in items if i[0] == 0]
    if len(items) != n_segments:
        return [None] * n_segments
    return items


def _series(patient_dir):
    import pydicom

    out = {}
    for series_dir in Path(patient_dir).glob('*/*'):
        files = sorted(series_dir.glob('*.dcm'))
        if files:
            header = pydicom.dcmread(files[0], stop_before_pixels=True)
            out[header.Modality if header.Modality == 'SEG' else header.SeriesDescription] = files
    return out


def load_patient(patient_dir):
    """DCE file grid [phase][slice] sorted by z, slice z positions, spacing (row, col, z),
    x/y origin, and SEG masks (segments, slices, rows, cols)."""
    import pydicom

    series = _series(patient_dir)
    by_phase = {}
    for path in series['AX Sen Vibrant MultiPhase']:
        header = pydicom.dcmread(path, stop_before_pixels=True,
                                 specific_tags=['TemporalPositionIdentifier', 'ImagePositionPatient',
                                                'ImageOrientationPatient', 'PixelSpacing', 'SliceThickness'])
        by_phase.setdefault(int(header.TemporalPositionIdentifier), []).append(
            (float(header.ImagePositionPatient[2]), path, header))
    phases = [sorted(by_phase[t], key=lambda item: item[0]) for t in sorted(by_phase)]
    z = np.array([item[0] for item in phases[0]])
    first = phases[0][0][2]
    spacing = (float(first.PixelSpacing[0]), float(first.PixelSpacing[1]), float(first.SliceThickness))
    origin = (float(first.ImagePositionPatient[0]), float(first.ImagePositionPatient[1]))
    row_direction_x = float(first.ImageOrientationPatient[0])   # x step per column (+1 or -1 for axial)

    seg = pydicom.dcmread(series['SEG'][0])
    pixels = seg.pixel_array
    pixels = pixels[None] if pixels.ndim == 2 else pixels
    n_segments = len(seg.SegmentSequence)
    masks = np.zeros((n_segments, len(z), pixels.shape[1], pixels.shape[2]), dtype=bool)
    for frame, groups in zip(pixels, seg.PerFrameFunctionalGroupsSequence):
        number = int(groups.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        index = int(np.argmin(np.abs(z - float(groups.PlanePositionSequence[0].ImagePositionPatient[2]))))
        masks[number - 1, index] |= frame > 0
    files = [[path for _, path, _ in phase] for phase in phases]
    return {'files': files, 'z': z, 'spacing': spacing, 'origin': origin, 'row_direction_x': row_direction_x,
            'masks': masks,
            't2_files': series.get('Ax T2 FSE', [])}


def _slices(files, indices):
    import pydicom

    return np.stack([pydicom.dcmread(files[i]).pixel_array.astype(np.float32) for i in indices])


def analyze_case(patient_dir, clinical=None):
    """One row per SEG segment: geometry, side, 5-phase kinetics, CNR, both label hypotheses."""
    from scipy import ndimage

    patient_dir = Path(patient_dir)
    lesions, referral = clinical or clinical_lesions()
    data = load_patient(patient_dir)
    items = lesions.get(patient_dir.name, [])
    hypotheses = {o: segment_labels(items, len(data['masks']), o) for o in ('tumours_first', 'report_order')}
    rows = []
    for k, mask in enumerate(data['masks']):
        row = {'group': patient_dir.name, 'segment': k + 1, 'n_segments': len(data['masks']),
               'n_clinical_lesions': len(items), 'n_phases': len(data['files']),
               'referral': referral.get(patient_dir.name), 'voxels': int(mask.sum())}
        for ordering, assigned in hypotheses.items():
            label, side, pathology = assigned[k] if assigned[k] is not None else (None, None, None)
            row[f'label_{ordering}'] = label
            row[f'side_{ordering}'] = side
            if ordering == 'tumours_first':
                row['pathology_code'] = pathology
        if not mask.any():
            rows.append(row)
            continue
        zs = np.where(mask.any(axis=(1, 2)))[0]
        _, components = ndimage.label(mask)
        voxel_ml = data['spacing'][0] * data['spacing'][1] * data['spacing'][2] / 1000.0
        cols = np.where(mask.any(axis=(0, 1)))[0]
        centre_col = float(np.average(np.arange(mask.shape[2]), weights=mask.sum(axis=(0, 1))))
        # LPS: +x is the patient's left. Bilateral axial breast MRI is centred near the
        # midline, so the side is which half of the image (along +x) holds the lesion.
        offset_x = (centre_col - (mask.shape[2] - 1) / 2) * data['spacing'][1] * data['row_direction_x']
        centre_x = data['origin'][0] + centre_col * data['spacing'][1] * data['row_direction_x']
        row.update({
            'volume_ml': row['voxels'] * voxel_ml, 'components': int(components),
            'slices': int(zs.size), 'z_peak': int(zs[np.argmax(mask.sum(axis=(1, 2))[zs])]),
            'width_mm': float((cols.max() - cols.min() + 1) * data['spacing'][1]),
            'centre_x_mm': centre_x, 'offset_from_centre_mm': offset_x,
            'side_from_mask': 'L' if offset_x > 0 else 'R',
        })
        # kinetics on the lesion's own slices only (5 phases x span)
        span = list(range(zs.min(), zs.max() + 1))
        sub_mask = mask[span]
        means = []
        for phase_files in data['files']:
            volume = _slices(phase_files, span)
            means.append(float(volume[sub_mask].mean()))
            if len(means) == 2:
                ring = ndimage.binary_dilation(sub_mask, iterations=3) & ~sub_mask
                ring_std = float(volume[ring].std()) if ring.any() else 0.0
                row['lesion_cnr_post1'] = abs(means[1] - float(volume[ring].mean())) / ring_std if ring_std > 0 else float('nan')
        for t, value in enumerate(means):
            row[f'mean_phase{t}'] = value
        peak = int(np.argmax(means))
        row['peak_phase'] = peak
        row['enhancement_ratio'] = means[peak] / means[0] if means[0] > 0 else float('nan')
        row['initial_enhancement'] = (means[1] - means[0]) / means[0] if means[0] > 0 else float('nan')
        row['washout'] = (means[peak] - means[-1]) / means[peak] if means[peak] > 0 else float('nan')
        rows.append(row)
    return rows


def samples(root):
    """One audit Sample per lesion: the first post-contrast slice where the lesion
    is largest, with the lesion's bounding box, labelled by the tumours-first rule."""
    clinical = clinical_lesions()
    lesions, _ = clinical
    out = []
    for patient_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        data = load_patient(patient_dir)
        assigned = segment_labels(lesions.get(patient_dir.name, []), len(data['masks']))
        for k, mask in enumerate(data['masks']):
            if not mask.any():
                continue
            z_peak = int(np.argmax(mask.sum(axis=(1, 2))))
            rows, cols = np.where(mask[z_peak])
            label = assigned[k][0] if assigned[k] is not None else None
            out.append(Sample(
                sample_id=f'{patient_dir.name}/lesion{k + 1}', dataset='ambl', modality='mri',
                label_name={1: 'tumor', 0: 'benign'}.get(label, 'unmatched'), label=label,
                used=label is not None, group=patient_dir.name, split=None, fold=None,
                native_kind='dicom', native_path=str(data['files'][1][z_peak]),
                box=(int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1),
                extra={'segment': k + 1, 'z_peak': z_peak},
            ))
    return out


def side_agreement(cases):
    """Share of lesions whose mask side matches the clinical side, per ordering hypothesis."""
    result = {}
    for ordering in ('tumours_first', 'report_order'):
        known = cases[cases[f'side_{ordering}'].notna() & cases['side_from_mask'].notna()]
        result[ordering] = {'lesions': int(len(known)),
                            'side_agrees': float((known[f'side_{ordering}'] == known['side_from_mask']).mean())
                            if len(known) else None}
    return result


def resolve_labels(cases, clinical=None):
    """Benign/malignant per SEG segment without relying on segment order:
      uniform    every clinical lesion of the patient has the same label
      by_side    labels differ, but each breast holds one label; use the mask's side
      ambiguous  both labels in the same breast (cannot be assigned from metadata)
      side_conflict / count_mismatch  mask side or lesion count disagrees with the XLSX
    Returns cases with `label`, `label_method`, and whether tumours-first agrees."""
    lesions, _ = clinical or clinical_lesions()
    out = []
    for patient, group in cases.groupby('group'):
        items = lesions.get(patient, [])
        labels = {label for label, _, _ in items}
        for _, row in group.iterrows():
            row = row.copy()
            if len(labels) == 1:
                row['label'], row['label_method'] = next(iter(labels)), 'uniform'
            elif len(group) != len(items):
                row['label'], row['label_method'] = None, 'count_mismatch'
            else:
                side_labels = {label for label, side, _ in items if side == row.get('side_from_mask')}
                if len(side_labels) == 1:
                    row['label'], row['label_method'] = next(iter(side_labels)), 'by_side'
                elif not side_labels:
                    row['label'], row['label_method'] = None, 'side_conflict'
                else:
                    row['label'], row['label_method'] = None, 'ambiguous'
            row['tumours_first_agrees'] = (None if row['label'] is None or pd.isna(row.get('label_tumours_first'))
                                           else bool(row['label'] == row['label_tumours_first']))
            out.append(row)
    return pd.DataFrame(out)
