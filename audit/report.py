"""Outlier flagging (D), the per-dataset summary table (A/G), and figures:
metric distributions, mean/std images, lesion-centroid heatmaps, and contact
sheets for manual review of outliers, duplicates and artifact flags."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from audit.metrics import thumbnail, to_uint8
from audit.reported import REPORTED
from audit.sample import Sample, load_native

OUTLIER_METRICS = ('brightness', 'contrast', 'sharpness_lapvar', 'entropy', 'dynamic_range',
                   'noise_sigma', 'snr', 'hf_energy_frac', 'aspect_ratio', 'height', 'width')
Z_THRESHOLD = 3.0
FLAG_COLUMNS = ('doppler_flag', 'paper_doppler_flag', 'paper_marker_flag', 'caliper_flag', 'marker_flag', 'text_flag',
                'dual_panel_flag', 'burnt_label_flag', 'implant_suspect_flag', 'mask_shape_mismatch')
REVIEW_COLUMNS = ['sample_id', 'reasons', 'decision', 'note']


def flag_outliers(frame):
    """|z| > 3 within the dataset on any metric (paper), plus a multivariate
    isolation-forest flag that catches unusual combinations."""
    from sklearn.ensemble import IsolationForest

    columns = [c for c in OUTLIER_METRICS if c in frame and frame[c].notna().sum() > 2]
    values = frame[columns].astype(float)
    z = (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    reasons = z.abs().gt(Z_THRESHOLD).apply(
        lambda r: ','.join(f'{c}(z={z.at[r.name, c]:+.1f})' for c in columns if r[c]), axis=1)
    out = frame[['sample_id', 'label_name', 'used', 'split']].copy()
    out['max_abs_z'] = z.abs().max(axis=1)
    out['z_reasons'] = reasons
    filled = ((values - values.mean()) / values.std(ddof=0).replace(0, 1)).fillna(0)
    if len(filled) >= 20:
        forest = IsolationForest(contamination=0.01, random_state=42).fit(filled)
        out['iforest_score'] = -forest.score_samples(filled)
        out['iforest_flag'] = forest.predict(filled) == -1
    else:
        out['iforest_score'], out['iforest_flag'] = np.nan, False
    flagged = out[(out['z_reasons'] != '') | out['iforest_flag']]
    return flagged.sort_values('max_abs_z', ascending=False)


def update_review_file(path, flagged):
    """Append newly flagged samples to the manual review sheet without touching
    decisions already recorded there."""
    path = Path(path)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=REVIEW_COLUMNS)
    reasons = flagged['z_reasons'].where(flagged['z_reasons'] != '', 'isolation_forest')
    new = pd.DataFrame({'sample_id': flagged['sample_id'], 'reasons': reasons, 'decision': '', 'note': ''})
    new = new[~new['sample_id'].isin(existing['sample_id'])]
    pd.concat([existing, new], ignore_index=True).to_csv(path, index=False)


def _describe(series):
    series = series.dropna()
    if series.empty:
        return None
    return {'min': float(series.min()), 'median': float(series.median()), 'max': float(series.max())}


def summarize(name, frame, headers, pairs, leaks, group_conflicts, candidates, cases=None):
    """Section A table for one dataset: reported vs. on-disk vs. used-by-loader."""
    used = frame[frame['used'].astype(bool)]

    def block(rows):
        groups = rows['group'].dropna()
        per_group = rows.groupby('group').size() if not groups.empty else pd.Series(dtype=int)
        return {
            'images': int(len(rows)),
            'by_label': rows['label_name'].value_counts().to_dict(),
            'groups': int(groups.nunique()) if not groups.empty else None,
            'images_per_group': _describe(per_group),
            'with_mask': int(rows['has_mask'].sum()),
            'with_box': int(rows['has_box'].sum()),
            'by_split': rows['split'].value_counts().to_dict(),
        }

    summary = {
        'dataset': name,
        'modality': frame['modality'].iloc[0] if len(frame) else None,
        'reported': REPORTED.get(name),
        'on_disk': block(frame),
        'used_by_loader': block(used),
        'load_errors': int(frame['load_error'].notna().sum()) if 'load_error' in frame else 0,
        'unique_sizes': int(frame[['height', 'width']].drop_duplicates().shape[0]) if 'height' in frame else None,
        'bits_stored': frame['bits_stored'].value_counts().to_dict() if 'bits_stored' in frame else None,
        'color_images': int(frame['is_color'].sum()) if 'is_color' in frame else None,
        'file_types': frame['file_ext'].value_counts().to_dict(),
        'flags': {c: int(frame[c].fillna(False).astype(bool).sum()) for c in FLAG_COLUMNS if c in frame},
        'multi_lesion_candidates': int(((frame.get('lesion_components', pd.Series(dtype=float)) > 1)
                                        | (frame.get('meta_n_mask_files', pd.Series(dtype=float)) > 1)).sum()),
        'duplicates': {
            'hash_candidates': int(candidates),
            'confirmed_pairs': int(len(pairs)),
            'groups': int(pairs['group_id'].nunique()) if len(pairs) else 0,
            'images_in_groups': int(len(set(pairs['i']) | set(pairs['j']))) if len(pairs) else 0,
            'label_conflict_pairs': int(pairs['label_conflict'].sum()) if len(pairs) else 0,
            'split_leak_pairs': int(len(leaks)),
            'groups_in_multiple_splits': len(group_conflicts),
        },
    }
    for column in ('meta_Device', 'meta_Machine'):
        if column in frame:
            summary[column.removeprefix('meta_').lower() + '_counts'] = frame[column].value_counts().to_dict()
    if headers is not None and len(headers):
        for column in ('Manufacturer', 'ManufacturerModelName', 'PhotometricInterpretation', 'BitsStored'):
            if column in headers:
                summary[f'dicom_{column}'] = headers[column].value_counts().to_dict()
    if cases is not None and len(cases):
        summary['cases'] = {
            'count': int(len(cases)),
            **{c: _describe(cases[c]) for c in ('n_phases', 'tumor_volume_ml', 'tumor_slices_axis0',
                                                 'enhancement_ratio', 'lesion_cnr_peak', 'lesion_cnr_sub2',
                                                 'slice_thickness') if c in cases},
        }
        for c in ('phases_repeated_in_img9', 'integer_valued'):
            if c in cases:
                summary['cases'][c] = int(cases[c].fillna(False).astype(bool).sum())
        if 'stored_bytes' in cases:
            summary['cases']['stored_gb'] = float(cases['stored_bytes'].sum() / 1e9)
    return summary


# ---------------------------------------------------------------- figures

def metric_distributions(frame, path):
    columns = [c for c in ('brightness', 'contrast', 'sharpness_lapvar', 'entropy', 'dynamic_range_frac',
                           'snr', 'hf_energy_frac', 'jpeg_blockiness', 'speckle_snr', 'lesion_cnr',
                           'lesion_area_frac', 'aspect_ratio') if c in frame and frame[c].notna().any()]
    if not columns:
        return
    cols = 4
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, column in zip(axes.ravel(), columns):
        for label_name, group in frame.groupby('label_name'):
            values = group[column].dropna()
            if len(values):
                ax.hist(values, bins=40, alpha=0.5, label=f'{label_name} ({len(values)})', density=True)
        ax.set_title(column, fontsize=9)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=7)
    for ax in axes.ravel()[len(columns):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def mean_std_images(frame, thumbs, path):
    labels = [l for l in frame['label_name'].dropna().unique()]
    if not labels:
        return
    fig, axes = plt.subplots(2, len(labels), figsize=(3 * len(labels), 6), squeeze=False)
    for k, label_name in enumerate(sorted(labels)):
        idx = np.nonzero((frame['label_name'] == label_name).to_numpy())[0]
        stack = thumbs[idx].astype(np.float32)
        axes[0, k].imshow(stack.mean(axis=0), cmap='gray')
        axes[0, k].set_title(f'mean: {label_name} (n={len(idx)})', fontsize=8)
        axes[1, k].imshow(stack.std(axis=0), cmap='magma')
        axes[1, k].set_title('per-pixel std', fontsize=8)
        for ax in axes[:, k]:
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def lesion_centroids(frame, path):
    if 'lesion_cx' not in frame or frame['lesion_cx'].isna().all():
        return
    labels = sorted(frame.loc[frame['lesion_cx'].notna(), 'label_name'].unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(3.2 * len(labels), 3.2), squeeze=False)
    for ax, label_name in zip(axes[0], labels):
        group = frame[(frame['label_name'] == label_name) & frame['lesion_cx'].notna()]
        ax.hist2d(group['lesion_cx'], group['lesion_cy'], bins=20, range=[[0, 1], [0, 1]], cmap='viridis')
        ax.invert_yaxis()
        ax.set_title(f'{label_name} lesion centroids (n={len(group)})', fontsize=8)
        ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _tile(row, size):
    sample = Sample(sample_id=row['sample_id'], dataset=row['dataset'], modality=row['modality'],
                    label_name=row['label_name'], label=None, used=False, group=None, split=None, fold=None,
                    native_kind=_kind(row['native_path']), native_path=row['native_path'],
                    native_frame=None if pd.isna(row.get('native_frame')) else int(row['native_frame']))
    try:
        loaded = load_native(sample)
        if loaded.rgb is not None:
            tile = np.asarray(Image.fromarray(loaded.rgb).resize((size, size)))
        else:
            tile = np.stack([thumbnail(to_uint8(loaded.gray, loaded.bits), size)] * 3, axis=-1)
    except Exception:
        tile = np.full((size, size, 3), 128, np.uint8)
    return tile


def _kind(path):
    suffix = Path(path).suffix.lower()
    return 'dicom' if suffix == '.dcm' else 'npy' if suffix == '.npy' else 'image'


def contact_sheet(rows, captions, path, size=224, cols=6):
    """Re-load the native images (raw stage) so small burnt-in text stays legible."""
    rows = list(rows)
    if not rows:
        return
    grid_rows = int(np.ceil(len(rows) / cols))
    sheet = Image.new('RGB', (cols * size, grid_rows * (size + 28)), 'white')
    draw = ImageDraw.Draw(sheet)
    for k, (row, caption) in enumerate(zip(rows, captions)):
        x, y = (k % cols) * size, (k // cols) * (size + 28)
        sheet.paste(Image.fromarray(_tile(row, size)), (x, y))
        for line_no, line in enumerate(str(caption)[:76].split('\n')[:2]):
            draw.text((x + 2, y + size + 2 + 12 * line_no), line[:38], fill='black')
    sheet.save(path)


def review_sheets(frame, outliers, pairs, figures_dir, rng):
    figures_dir = Path(figures_dir)
    by_id = frame.set_index('sample_id', drop=False)
    top = outliers.head(36)
    contact_sheet([by_id.loc[s] for s in top['sample_id']],
                  [f"{s}\n{r[:38] or 'iforest'}" for s, r in zip(top['sample_id'], top['z_reasons'])],
                  figures_dir / 'outliers.png')
    if len(pairs):
        first_groups = pairs.drop_duplicates('group_id').head(18)
        rows, captions = [], []
        for _, pair in first_groups.iterrows():
            rows += [frame.iloc[pair['i']], frame.iloc[pair['j']]]
            captions += [f"g{pair['group_id']} {pair['sample_id_a']}\n{pair['label_name_a']} {pair['split_a']}",
                         f"ssim={pair['ssim']:.3f} {pair['sample_id_b']}\n{pair['label_name_b']} {pair['split_b']}"]
        contact_sheet(rows, captions, figures_dir / 'duplicates.png')
    for flag in FLAG_COLUMNS:
        if flag in frame and frame[flag].fillna(False).astype(bool).any():
            hits = frame[frame[flag].fillna(False).astype(bool)]
            hits = hits.sample(min(24, len(hits)), random_state=rng)
            contact_sheet([r for _, r in hits.iterrows()],
                          [f"{r['sample_id']}\n{r['label_name']}" for _, r in hits.iterrows()],
                          figures_dir / f'flag_{flag}.png')
    for label_name, group in frame.groupby('label_name'):
        picks = group.sample(min(12, len(group)), random_state=rng)
        safe = str(label_name).replace('/', '_').replace(' ', '_')
        contact_sheet([r for _, r in picks.iterrows()], [r['sample_id'] for _, r in picks.iterrows()],
                      figures_dir / f'random_{safe}.png')


def cross_dataset_violins(frames, path):
    combined = pd.concat(frames, ignore_index=True)
    columns = [c for c in ('brightness', 'contrast', 'entropy', 'snr', 'sharpness_lapvar', 'hf_energy_frac',
                           'jpeg_blockiness', 'lesion_cnr') if c in combined]
    names = sorted(combined['dataset'].unique(), key=lambda n: (combined.loc[combined['dataset'] == n, 'modality'].iloc[0], n))
    fig, axes = plt.subplots(len(columns), 1, figsize=(max(8, 1.1 * len(names)), 2.4 * len(columns)), squeeze=False)
    for ax, column in zip(axes[:, 0], columns):
        data = [combined.loc[combined['dataset'] == n, column].dropna().to_numpy() for n in names]
        keep = [k for k, d in enumerate(data) if len(d)]
        if keep:
            ax.violinplot([data[k] for k in keep], positions=keep, showmedians=True)
        ax.set_xticks(range(len(names)), names, fontsize=8)
        ax.set_ylabel(column, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
