"""ODELIA Breast MRI Challenge 2025 (unilateral configuration) audit.

The release is a set of parquet shards: one row per breast, with metadata
(UID, PatientID, Institution, Age, Lesion 0/1/2, Split, Fold) and 256x256x32
volumes for Pre, Post_1..7, Sub_1 and T2. This module turns each breast into
one audit row (sequence availability, intensity/enhancement statistics,
perceptual hashes and a thumbnail of the subtraction maximum-intensity
projection), so the generic duplicate/outlier/summary code can run on it,
and adds the two checks this dataset needs most: label balance per centre and
whether the centre (scanner/protocol) is identifiable from the images.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from audit import metrics

META_COLUMNS = ['UID', 'Fold', 'Split', 'PatientID', 'Institution', 'Age', 'Lesion']
SEQUENCES = ['Pre', 'Post_1', 'Post_2', 'Post_3', 'Post_4', 'Post_5', 'Post_6', 'Post_7', 'Sub_1', 'T2']
LOADED = ('Pre', 'Post_1', 'Sub_1', 'T2')
LESION_NAMES = {0: 'no lesion', 1: 'benign', 2: 'malignant'}
OOD_CENTRE = 'RSH'   # held out as the out-of-distribution test centre in the challenge
THUMB = 64


def _volumes(column):
    """pyarrow list<list<list<list<int16>>>> column -> list of (depth, height,
    width) arrays (None for nulls). Stored as (1, H, W, D) with D = 32 slices."""
    out = []
    for value in column.to_pylist():
        if value is None:
            out.append(None)
            continue
        volume = np.squeeze(np.asarray(value, dtype=np.int16))
        out.append(np.moveaxis(volume, -1, 0) if volume.shape[-1] < volume.shape[0] else volume)
    return out


def _stats(volume, mask):
    values = volume[mask].astype(np.float32)
    return float(values.mean()), float(values.std()), float(np.percentile(values, 99))


def analyze_shard(path):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    present_columns = set(parquet.schema_arrow.names)
    rows, thumbs = [], []
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=META_COLUMNS + [f'Image_{s}' for s in LOADED])
        meta = table.select(META_COLUMNS).to_pandas()
        volumes = {s: _volumes(table.column(f'Image_{s}')) for s in LOADED}
        for k, info in meta.iterrows():
            row = {
                'sample_id': info['UID'], 'dataset': 'odelia', 'modality': 'mri',
                'label_name': LESION_NAMES.get(int(info['Lesion']), 'unknown'),
                'label': {1: 0, 2: 1}.get(int(info['Lesion'])),
                'used': int(info['Lesion']) in (1, 2),
                'group': info['PatientID'], 'split': info['Split'], 'fold': int(info['Fold']),
                'native_path': str(path), 'has_mask': False, 'has_box': False, 'file_ext': '.parquet',
                'meta_institution': info['Institution'], 'meta_age_years': round(int(info['Age']) / 365.25, 1),
                'meta_ood_centre': info['Institution'] == OOD_CENTRE,
            }
            pre, post, sub, t2 = (volumes[s][k] for s in LOADED)
            for s in LOADED:
                row[f'has_{s}'] = volumes[s][k] is not None
            if pre is None:
                rows.append(row)
                thumbs.append(np.zeros((THUMB, THUMB), np.uint8))
                continue
            breast = pre > np.percentile(pre, 60)
            row['depth'], row['height'], row['width'] = pre.shape
            row['breast_frac'] = float(breast.mean())
            row['pre_mean'], row['pre_std'], row['pre_p99'] = _stats(pre, breast)
            if post is not None:
                row['post1_mean'], _, _ = _stats(post, breast)
                row['enhancement_ratio'] = row['post1_mean'] / row['pre_mean'] if row['pre_mean'] > 0 else np.nan
            image = sub if sub is not None else pre
            mip = image.max(axis=0).astype(np.float32)
            low, high = np.percentile(mip, [1, 99.5])
            img8 = np.clip((mip - low) / max(high - low, 1) * 255, 0, 255).astype(np.uint8)
            roi = np.ones(img8.shape, bool)
            quality, _ = metrics.quality_metrics(img8.astype(np.float32), 8, roi, 'mri')
            row.update({k: quality[k] for k in ('brightness', 'contrast', 'entropy', 'sharpness_lapvar',
                                                'noise_sigma', 'snr')})
            row['ahash'], row['dhash'], row['phash'] = metrics.ahash(img8), metrics.dhash(img8), metrics.phash(img8)
            if t2 is not None:
                row['t2_mean'] = float(t2.mean())
            rows.append(row)
            thumbs.append(metrics.thumbnail(img8, THUMB))
    # which post-contrast phases exist, per breast (cheap: validity bitmap only)
    validity = {}
    for s in SEQUENCES:
        name = f'Image_{s}'
        if name in present_columns:
            validity[s] = pq.read_table(path, columns=[name]).column(0).is_valid().to_pylist()
    for k, row in enumerate(rows):
        row['n_post'] = sum(bool(validity.get(f'Post_{i}', [False] * len(rows))[k]) for i in range(1, 8))
        row['has_T2'] = bool(validity.get('T2', [False] * len(rows))[k])
    return rows, thumbs


def centre_probes(frame):
    """Can the centre be predicted from simple image statistics, and how much
    benign/malignant signal does the centre alone carry?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    features = ['brightness', 'contrast', 'entropy', 'pre_mean', 'pre_std', 'enhancement_ratio', 'breast_frac', 'n_post']
    data = frame.dropna(subset=['pre_mean']).copy()
    X = data[features].fillna(data[features].median())
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    centre_pred = cross_val_predict(model, X, data['meta_institution'], cv=folds)
    out = {'centre_accuracy_from_image_stats': float((centre_pred == data['meta_institution']).mean()),
           'centre_chance_accuracy': float(data['meta_institution'].value_counts(normalize=True).max())}
    lesions = data[data['label'].notna()]
    y = lesions['label'].astype(int)
    onehot = make_pipeline(OneHotEncoder(handle_unknown='ignore'), LogisticRegression(max_iter=3000))
    out['benign_vs_malignant_auc_from_centre_only'] = float(cross_val_score(
        onehot, lesions[['meta_institution']], y, cv=folds, scoring='roc_auc').mean())
    out['benign_vs_malignant_auc_from_image_stats'] = float(cross_val_score(
        model, X.loc[lesions.index], y, cv=folds, scoring='roc_auc').mean())
    return out


def centre_table(frame):
    table = frame.groupby(['meta_institution', 'label_name']).size().unstack(fill_value=0)
    lesions = table.get('benign', 0) + table.get('malignant', 0)
    table['benign_share_of_lesions'] = (table.get('benign', 0) / lesions.replace(0, np.nan)).round(3)
    table['n_post_median'] = frame.groupby('meta_institution')['n_post'].median()
    table['t2_available'] = frame.groupby('meta_institution')['has_T2'].mean().round(3)
    return table


def mip_sheet(frame, thumbs, path):
    """Mean subtraction-MIP per centre x label: a quick visual of protocol differences."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    centres = sorted(frame['meta_institution'].unique())
    labels = ['no lesion', 'benign', 'malignant']
    fig, axes = plt.subplots(len(labels), len(centres), figsize=(1.8 * len(centres), 1.9 * len(labels)), squeeze=False)
    for i, label in enumerate(labels):
        for j, centre in enumerate(centres):
            idx = np.nonzero(((frame['meta_institution'] == centre) & (frame['label_name'] == label)).to_numpy())[0]
            ax = axes[i, j]
            if len(idx):
                ax.imshow(thumbs[idx].astype(np.float32).mean(axis=0), cmap='gray')
            ax.set_title(f'{centre} {label} ({len(idx)})', fontsize=6)
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build(root, workers):
    from concurrent.futures import ProcessPoolExecutor

    shards = sorted(Path(root, 'unilateral').glob('*.parquet'))
    rows, thumbs = [], []
    with ProcessPoolExecutor(workers) as pool:
        for k, (shard_rows, shard_thumbs) in enumerate(pool.map(analyze_shard, shards), 1):
            rows += shard_rows
            thumbs += shard_thumbs
            print(f'  shards: {k}/{len(shards)}', flush=True)
    return pd.DataFrame(rows), np.stack(thumbs)
