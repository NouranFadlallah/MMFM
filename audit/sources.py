"""Dataset adapters: enumerate every image on disk as a `Sample`, marking which
ones the training loader uses and which split/fold training assigns them.

Inclusion and splits come from calling the metadata/frame functions in
training/train.py with the CLI defaults (seed 42, --validation-fraction 0.1,
5 folds), so "used" and "split" mean exactly what training sees.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from audit.sample import Sample

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
VALIDATION_FRACTION = 0.1   # training/train.py --validation-fraction default
NUM_FOLDS = 5

DATASET_ROOTS = {
    'busbra': 'datasets/ultrasound/busbra/BUSBRA',
    'busi': 'datasets/ultrasound/Dataset_BUSI/Dataset_BUSI_with_GT',
    'busc': 'datasets/ultrasound/us-dataset',
    'breast': 'datasets/ultrasound/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023',
    'mias': 'datasets/mammography/mias/all-mias',
    'cdd_cesm': 'datasets/mammography/cdd_cesm',
    'cmmd': 'datasets/mammography/cmmd_dicom',
    'bcsdbt': 'datasets/mammography/bcsdbt_dicom',
    'breamdm': 'datasets/MRI/BreaDM',
    'ispy2': 'datasets/MRI/ISPY2/cls/img9Se',
}
MODALITY = {
    'busbra': 'ultrasound', 'busi': 'ultrasound', 'busc': 'ultrasound', 'breast': 'ultrasound',
    'mias': 'mammography', 'cdd_cesm': 'mammography', 'cmmd': 'mammography', 'bcsdbt': 'mammography',
    'breamdm': 'mri', 'ispy2': 'mri',
}
ISPY2_RAW_ROOT = Path('/mnt/data/mmfm_datasets/MRI/PKG - BreastDCEDL_ISPY2/BreastDCEDL_ISPY2')


def _train_module():
    sys.path.insert(0, str(REPO_ROOT))
    from training import train
    return train


def _resolve(path):
    return str(Path(path).resolve()) if path is not None else None


def _split_lookup(train_frame, validation_frame, column='image'):
    lookup = {_resolve(p): 'train' for p in train_frame[column]}
    lookup.update({_resolve(p): 'val' for p in validation_frame[column]})
    return lookup


def _fold_lookup(metadata, group_col, column='image'):
    folds = _train_module()._assign_stratified_group_folds(metadata, group_col, NUM_FOLDS, SEED)
    return {_resolve(p): int(f) for p, f in zip(metadata[column], folds)}


def _clean(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value if isinstance(value, (int, float, bool)) else str(value)


def busbra(root):
    T = _train_module()
    root = Path(root)
    csv = pd.read_csv(root / 'bus_data.csv')
    used = T._busbra_metadata(root)
    used_paths = {_resolve(p) for p in used['image']}
    split = _split_lookup(*T._busbra_frame(root, None, SEED, VALIDATION_FRACTION))
    samples = []
    for _, row in csv.iterrows():
        image = root / 'Images' / f"{row['ID']}.png"
        mask = root / 'Masks' / f"mask_{row['ID'].removeprefix('bus_')}.png"
        key = _resolve(image)
        x, y, w, h = ast.literal_eval(row['BBOX']) if isinstance(row['BBOX'], str) else (None,) * 4
        samples.append(Sample(
            sample_id=row['ID'], dataset='busbra', modality='ultrasound',
            label_name=row['Pathology'], label={'benign': 0, 'malignant': 1}.get(row['Pathology']),
            used=key in used_paths, group=f"case{row['Case']}", split=split.get(key),
            fold=int(row['K5B']) if not pd.isna(row['K5B']) else None,
            native_kind='image', native_path=str(image),
            mask_path=str(mask) if mask.exists() else None,
            box=(x, y, x + w, y + h) if x is not None else None,
            extra={k: _clean(row[k]) for k in ('Histology', 'BIRADS', 'Device', 'Side')},
        ))
    return samples


def busi(root):
    T = _train_module()
    root = Path(root)
    used = T._busi_metadata(root)
    used_paths = {_resolve(p) for p in used['image']}
    split = _split_lookup(*T._busi_frame(root, SEED, VALIDATION_FRACTION))
    folds = _fold_lookup(used.reset_index(drop=True), None)
    samples = []
    for class_name in ('benign', 'malignant', 'normal'):
        for image in sorted((root / class_name).glob('*.png')):
            if 'mask' in image.stem:
                continue
            masks = sorted((root / class_name).glob(f'{image.stem}_mask*.png'))
            key = _resolve(image)
            samples.append(Sample(
                sample_id=f'{class_name}/{image.stem}', dataset='busi', modality='ultrasound',
                label_name=class_name, label={'benign': 0, 'malignant': 1}.get(class_name),
                used=key in used_paths, group=None, split=split.get(key), fold=folds.get(key),
                native_kind='image', native_path=str(image),
                mask_path=str(root / class_name / f'{image.stem}_mask.png') if masks else None,
                extra={'n_mask_files': len(masks)},
            ))
    return samples


def busc(root):
    T = _train_module()
    root = Path(root)
    used = T._busc_metadata(root)
    split = _split_lookup(*T._busc_frame(root, SEED, VALIDATION_FRACTION))
    folds = _fold_lookup(used.reset_index(drop=True), None)
    samples = []
    for image in sorted((root / 'originals').rglob('*')):
        if not image.is_file():
            continue
        class_name = image.parent.name
        key = _resolve(image)
        samples.append(Sample(
            sample_id=f'{class_name}/{image.stem}', dataset='busc', modality='ultrasound',
            label_name=class_name, label={'benign': 0, 'malignant': 1}.get(class_name),
            used=key in split, group=None, split=split.get(key), fold=folds.get(key),
            native_kind='image', native_path=str(image),
        ))
    return samples


def breast(root):
    T = _train_module()
    root = Path(root)
    sheet = pd.read_excel(next(root.glob('*.xlsx')), sheet_name=0)
    image_dir = root / 'BrEaST-Lesions_USG-images_and_masks'
    used = T._breast_metadata(root)
    used_paths = {_resolve(p) for p in used['image']}
    split = _split_lookup(*T._breast_frame(root, SEED, VALIDATION_FRACTION))
    folds = _fold_lookup(used.reset_index(drop=True), 'CaseID')
    samples = []
    for _, row in sheet.iterrows():
        image = image_dir / str(row['Image_filename'])
        mask = image_dir / str(row['Mask_tumor_filename'])
        key = _resolve(image)
        samples.append(Sample(
            sample_id=str(row['Image_filename']), dataset='breast', modality='ultrasound',
            label_name=row['Classification'], label={'benign': 0, 'malignant': 1}.get(row['Classification']),
            used=key in used_paths, group=str(row['CaseID']), split=split.get(key), fold=folds.get(key),
            native_kind='image', native_path=str(image),
            mask_path=str(mask) if mask.exists() else None,
            extra={k: _clean(row[k]) for k in ('Age', 'Tissue_composition', 'BIRADS', 'Verification',
                                               'Diagnosis', 'Pixel_size', 'Mask_other_filename')},
        ))
    return samples


def mias(root):
    T = _train_module()
    root = Path(root)
    used = T._mias_metadata(root)
    used_paths = {_resolve(p) for p in used['image']}
    split = _split_lookup(*T._mias_frame(root, SEED, VALIDATION_FRACTION))
    folds = _fold_lookup(used.reset_index(drop=True), 'patient')
    lines = {}
    for line in (root / 'Info.txt').read_text(encoding='utf-8', errors='ignore').splitlines():
        fields = line.split()
        if fields and fields[0].startswith('mdb'):
            lines.setdefault(fields[0], []).append(fields)
    samples = []
    for image in sorted(root.glob('*.pgm')):
        entries = lines.get(image.stem, [])
        severities = [f[3] for f in entries if len(f) > 3 and f[3] in ('B', 'M')]
        label_name = {'B': 'benign', 'M': 'malignant'}.get(severities[0]) if severities else 'normal'
        key = _resolve(image)
        # lesion box = bounding box of the first annotated circle (Info.txt y is
        # measured from the bottom edge, as in training's _mias_metadata)
        circle = next((f for f in entries if len(f) >= 7 and f[3] in ('B', 'M')), None)
        box = None
        if circle:
            x, y, radius = map(int, circle[4:7])
            box = (x - radius, 1024 - y - radius, x + radius, 1024 - y + radius)
        samples.append(Sample(
            sample_id=image.stem, dataset='mias', modality='mammography',
            label_name=label_name, label={'benign': 0, 'malignant': 1}.get(label_name),
            used=key in used_paths, group=str((int(image.stem[3:]) + 1) // 2), split=split.get(key),
            fold=folds.get(key), native_kind='image', native_path=str(image), box=box,
            extra={
                'tissue': entries[0][1] if entries else None,
                'abnormality': ','.join(sorted({f[2] for f in entries if len(f) > 2})) or None,
                'n_annotation_lines': len(entries),
            },
        ))
    return samples


def cdd_cesm(root, annotation_path='datasets/mammography/cdd_cesm_annotations.xlsx'):
    T = _train_module()
    root = Path(root)
    sheet = pd.read_excel(annotation_path, sheet_name='all')
    split = _split_lookup(*T._cdd_cesm_frame(root, annotation_path, SEED, VALIDATION_FRACTION))
    folders = {'CESM': 'Subtracted images of CDD-CESM', 'DM': 'Low energy images of CDD-CESM'}
    rows_per_image = sheet.groupby('Image_name').size()
    samples = []
    for _, row in sheet.drop_duplicates('Image_name').iterrows():
        image = root / folders[row['Type']] / f"{row['Image_name']}.jpg"
        if not image.exists():
            continue
        key = _resolve(image)
        pathology = row['Pathology Classification/ Follow up']
        samples.append(Sample(
            sample_id=str(row['Image_name']), dataset='cdd_cesm', modality='mammography',
            label_name=f"{pathology} ({row['Type']})", label={'Benign': 0, 'Malignant': 1}.get(pathology),
            used=key in split, group=str(row['Patient_ID']), split=split.get(key), fold=None,
            native_kind='image', native_path=str(image),
            extra={'type': row['Type'], 'n_annotation_rows': int(rows_per_image[row['Image_name']]),
                   **{k: _clean(row[k]) for k in ('Side', 'View', 'Age', 'Breast density (ACR)',
                                                   'BIRADS', 'Findings', 'Machine')}},
        ))
    return samples


def cmmd(root, clinical_path='datasets/mammography/cmmd_clinicaldata.xlsx',
         cache_dir='datasets/mammography/cmmd_png'):
    T = _train_module()
    root = Path(root)
    train_frame, validation_frame = T._cmmd_frame(root, clinical_path, cache_dir, SEED, VALIDATION_FRACTION)
    split = {Path(p).stem: 'train' for p in train_frame['image']}
    split.update({Path(p).stem: 'val' for p in validation_frame['image']})
    labels = {Path(p).stem: (int(l), str(g)) for p, l, g in
              zip(pd.concat([train_frame, validation_frame])['image'],
                  pd.concat([train_frame, validation_frame])['label'],
                  pd.concat([train_frame, validation_frame])['patient'])}
    clinical = pd.read_excel(clinical_path, sheet_name='Sheet1')
    clinical_by_patient = {pid: g for pid, g in clinical.groupby('ID1')}
    samples = []
    for dicom in sorted(root.rglob('*.dcm')):
        patient = dicom.relative_to(root).parts[0]
        label, group = labels.get(dicom.stem, (None, patient))
        rows = clinical_by_patient.get(patient)
        extra = {}
        if rows is not None:
            extra = {'age': _clean(rows['Age'].iloc[0]), 'abnormality': _clean(rows['abnormality'].iloc[0]),
                     'subtype': _clean(rows['subtype'].iloc[0])}
        samples.append(Sample(
            sample_id=f'{patient}/{dicom.stem}', dataset='cmmd', modality='mammography',
            label_name={0: 'Benign', 1: 'Malignant'}.get(label, 'no clinical label'), label=label,
            used=dicom.stem in split, group=group, split=split.get(dicom.stem), fold=None,
            native_kind='dicom', native_path=str(dicom),
            prepared_kind='image' if dicom.stem in split else None,
            prepared_path=str(Path(cache_dir) / f'{dicom.stem}.png') if dicom.stem in split else None,
            extra=extra,
        ))
    return samples


def bcsdbt(root, labels_path='datasets/mammography/bcsdbt_labels.csv',
           file_paths_path='datasets/mammography/bcsdbt_file_paths.csv',
           boxes_path='datasets/mammography/bcsdbt_boxes.csv', cache_dir='datasets/mammography/bcsdbt_png'):
    T = _train_module()
    root = Path(root)
    train_frame, validation_frame = T._dbt_frame(root, labels_path, file_paths_path, boxes_path, cache_dir,
                                                 SEED, VALIDATION_FRACTION)
    split = _split_lookup(train_frame, validation_frame)
    file_paths = pd.read_csv(file_paths_path).set_index(['PatientID', 'StudyUID', 'View'])['descriptive_path']
    boxes = pd.read_csv(boxes_path).set_index(['PatientID', 'StudyUID', 'View'])
    samples = []
    for _, row in pd.read_csv(labels_path).iterrows():
        key = (row['PatientID'], row['StudyUID'], row['View'])
        label_name = next((c for c in ('Cancer', 'Benign', 'Actionable', 'Normal') if row[c] == 1), 'unknown')
        dicom = T._dbt_resolve_dicom(root, row['PatientID'], file_paths[key]) if key in file_paths.index else None
        if dicom is None:
            continue
        box = boxes.loc[key] if key in boxes.index else None
        if isinstance(box, pd.DataFrame):
            box = box.iloc[0]
        frame = int(box['Slice']) if box is not None else None
        prepared = Path(cache_dir) / f'{dicom.parent.name}_s{frame:03d}.png' if frame is not None else None
        prepared_key = _resolve(prepared) if prepared is not None else None
        samples.append(Sample(
            sample_id=f"{row['PatientID']}/{row['StudyUID']}/{row['View']}", dataset='bcsdbt',
            modality='mammography', label_name=label_name,
            label={'Benign': 0, 'Cancer': 1}.get(label_name), used=prepared_key in split,
            group=row['PatientID'], split=split.get(prepared_key), fold=None,
            native_kind='dicom', native_path=str(dicom), native_frame=frame,
            prepared_kind='image' if prepared is not None and prepared.exists() else None,
            prepared_path=str(prepared) if prepared is not None else None,
            box=(int(box['X']), int(box['Y']), int(box['X'] + box['Width']), int(box['Y'] + box['Height']))
            if box is not None else None,
            extra={'view': row['View'], 'volume_slices': int(box['VolumeSlices']) if box is not None else None},
        ))
    return samples


def _img9se_samples(root, dataset, used_splits):
    root = Path(root)
    samples = []
    for path in sorted(root.glob('*/*/*/*.npy')):
        split_name, class_name, patient = path.parts[-4], path.parts[-3], path.parts[-2]
        samples.append(Sample(
            sample_id=f'{split_name}/{class_name}/{patient}/{path.stem}', dataset=dataset, modality='mri',
            label_name=class_name, label={'Benign': 0, 'Malignant': 1}.get(class_name),
            used=split_name in used_splits, group=patient, split=split_name, fold=None,
            native_kind='npy', native_path=str(path),
        ))
    return samples


def breamdm(root):
    # training reads only cls/img9Se train + val; the official test split is unused
    return _img9se_samples(Path(root) / 'cls' / 'img9Se', 'breamdm', {'train', 'val'})


def ispy2(root):
    return _img9se_samples(root, 'ispy2', {'train', 'val'})


ADAPTERS = {
    'busbra': busbra, 'busi': busi, 'busc': busc, 'breast': breast, 'mias': mias,
    'cdd_cesm': cdd_cesm, 'cmmd': cmmd, 'bcsdbt': bcsdbt, 'breamdm': breamdm, 'ispy2': ispy2,
}


def case_dirs(dataset):
    """Patient-level volume folders for the 3D case analysis (MRI only)."""
    if dataset == 'ispy2':
        return sorted(p for p in ISPY2_RAW_ROOT.iterdir() if p.is_dir()) if ISPY2_RAW_ROOT.exists() else []
    if dataset == 'breamdm':
        return sorted(Path(DATASET_ROOTS['breamdm'], 'seg3D').glob('*/images/*'))
    return []
