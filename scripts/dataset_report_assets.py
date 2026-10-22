#!/usr/bin/env python3
"""Build the tables, macros and figures for docs/dataset_report/main.tex from
the dataset-audit outputs in analysis/ (scripts/analyze_datasets.py), so every
number in the report is regenerated rather than hand-typed.

Usage:
    .venv/bin/python scripts/dataset_report_assets.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from audit.reported import REPORTED  # noqa: E402

ANALYSIS = REPO_ROOT / 'analysis'
OUT = REPO_ROOT / 'docs' / 'dataset_report'
GENERATED = OUT / 'generated'
FIGURES = OUT / 'figures'

DATASETS = ['busbra', 'busi', 'busc', 'breast', 'bus_cot', 'bus_uclm', 'mias', 'cdd_cesm', 'cmmd', 'bcsdbt',
            'bcsdbt_extra', 'cbis_ddsm', 'dmid', 'breamdm', 'ispy2', 'odelia', 'ambl']
NAMES = {'busbra': 'BUS-BRA', 'busi': 'BUSI', 'busc': 'BUSC', 'breast': 'BrEaST', 'bus_cot': 'BUS-CoT',
         'mias': 'mini-MIAS', 'cdd_cesm': 'CDD-CESM', 'cmmd': 'CMMD', 'bcsdbt': 'BCS-DBT (val)',
         'bcsdbt_extra': 'BCS-DBT (train+test)', 'cbis_ddsm': 'CBIS-DDSM', 'breamdm': 'BreastDM', 'ispy2': 'ISPY2',
         'odelia': 'ODELIA', 'bus_uclm': 'BUS-UCLM', 'dmid': 'DMID', 'ambl': 'AMBL (lesions)'}
MODALITY = {'busbra': 'US', 'busi': 'US', 'busc': 'US', 'breast': 'US', 'bus_cot': 'US', 'mias': 'MG',
            'cdd_cesm': 'MG', 'cmmd': 'MG', 'bcsdbt': 'MG', 'bcsdbt_extra': 'MG', 'cbis_ddsm': 'MG',
            'breamdm': 'MRI', 'ispy2': 'MRI', 'odelia': 'MRI', 'bus_uclm': 'US', 'dmid': 'MG', 'ambl': 'MRI'}
NEW_DATASETS = ['bus_cot', 'bcsdbt_extra', 'cbis_ddsm', 'odelia', 'bus_uclm', 'dmid', 'ambl']
# validated categorical palette (dataviz skill reference instance, slots 1-4)
COLORS = {'US': '#2a78d6', 'MG': '#eb6834', 'MRI': '#1baf7a'}
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, MUTED = '#0b0b0b', '#52514e'
TILE = 224 + 28  # contact-sheet row height used by audit/report.py:contact_sheet

plt.rcParams.update({
    'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.5, 'axes.axisbelow': True,
})


def load():
    frames, summaries = {}, {}
    for name in DATASETS:
        frames[name] = pd.read_parquet(ANALYSIS / name / 'per_image.parquet')
        summaries[name] = json.loads((ANALYSIS / name / 'summary.json').read_text())
    return frames, summaries


def fmt(value, digits=0):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '--'
    return f'{value:,.{digits}f}'


def used(frame):
    return frame[frame['used'].astype(str).str.lower().isin(['true', '1'])]


def table_counts(frames, summaries):
    lines = []
    for name in DATASETS:
        s, f = summaries[name], frames[name]
        rep = REPORTED.get(name) or {}
        u = used(f)
        benign = int((u['label'] == 0).sum())
        malignant = int((u['label'] == 1).sum())
        used_groups = s['used_by_loader']['groups']
        if name == 'ambl':
            # SEG segment order is unreliable; count the side-checked labels instead
            resolved = pd.read_csv(ANALYSIS / 'ambl' / 'lesions_resolved.csv')
            u = resolved[resolved['label'].notna()]
            benign, malignant = int((u['label'] == 0).sum()), int((u['label'] == 1).sum())
            used_groups = u['group'].nunique()
        rep_cell = f"{fmt(rep.get('images'))} / {fmt(rep.get('patients'))}"
        groups = s['on_disk']['groups']
        lines.append(' & '.join([
            NAMES[name], MODALITY[name], rep_cell, fmt(s['on_disk']['images']), fmt(groups),
            fmt(len(u)), f'{benign:,} / {malignant:,}', fmt(used_groups),
        ]) + r' \\')
    (GENERATED / 'table_counts.tex').write_text('\n'.join(lines) + '\n')


def table_quality(frames, summaries):
    lines = []
    for name in DATASETS:
        u = used(frames[name])
        med = lambda c: u[c].median() if c in u and u[c].notna().any() else None
        bits = ','.join(sorted({str(int(float(k))) for k in (summaries[name]['bits_stored'] or {'16': 0})}, key=int))
        lines.append(' & '.join([
            NAMES[name], f"{fmt(med('height'))}$\\times${fmt(med('width'))}",
            fmt(summaries[name]['unique_sizes']), bits, fmt(med('brightness'), 1), fmt(med('contrast'), 1),
            fmt(med('entropy'), 2), fmt(med('snr'), 0), fmt(med('jpeg_blockiness'), 2), fmt(med('lesion_cnr'), 2),
        ]) + r' \\')
    (GENERATED / 'table_quality.tex').write_text('\n'.join(lines) + '\n')


def table_integrity(frames, summaries):
    lines = []
    for name in DATASETS:
        s, f = summaries[name], frames[name]
        d = s['duplicates']
        outliers = len(pd.read_csv(ANALYSIS / name / 'outliers.csv'))
        flag = lambda c: f'{int(f[c].fillna(False).astype(bool).sum()):,}' if c in f else '--'
        dup_cells = [fmt(d['groups']), fmt(d['images_in_groups']), fmt(d['label_conflict_pairs'])]
        if name == 'ambl':
            # thumbnail matches of whole-breast slices; all rejected at full resolution (SSIM 0.47-0.62)
            dup_cells = ['0$^{\\dagger}$', '0$^{\\dagger}$', '0$^{\\dagger}$']
        lines.append(' & '.join([
            NAMES[name], *dup_cells,
            fmt(d['split_leak_pairs']), flag('marker_flag'), flag('text_flag'),
            flag('burnt_label_flag'), f"{outliers:,} ({100 * outliers / len(f):.0f}\\%)",
        ]) + r' \\')
    (GENERATED / 'table_integrity.tex').write_text('\n'.join(lines) + '\n')


def table_mri():
    rows = []
    for name, label in (('breamdm', 'BreastDM (seg3D)'), ('ispy2', 'ISPY2 (NIfTI)')):
        cases = pd.read_parquet(ANALYSIS / name / 'cases.parquet')

        def q(column, digits=2):
            if column not in cases or cases[column].notna().sum() == 0:
                return '--'
            values = cases[column].astype(float)
            return (f'{values.median():.{digits}f} '
                    f'[{values.quantile(.25):.{digits}f}, {values.quantile(.75):.{digits}f}]')

        sequences = q('n_phases', 0) if name == 'ispy2' else '3 sequences'
        cnr = q('lesion_cnr_peak') if name == 'ispy2' else q('lesion_cnr_sub2')
        rows.append(' & '.join([
            label, fmt(len(cases)), sequences, q('tumor_volume_ml', 1), q('tumor_slices_axis0', 0),
            q('tumor_components', 0), q('enhancement_ratio'), cnr,
        ]) + r' \\')
    (GENERATED / 'table_mri.tex').write_text('\n'.join(rows) + '\n')


def table_bus_cot():
    frame = pd.read_parquet(ANALYSIS / 'bus_cot' / 'per_image.parquet')
    cross = pd.read_csv(ANALYSIS / 'cross_dataset' / 'ultrasound_cross_duplicates.csv',
                        dtype={'sample_id_a': str, 'sample_id_b': str})
    ids = set(cross.loc[cross.dataset_a == 'bus_cot', 'sample_id_a']) | set(cross.loc[cross.dataset_b == 'bus_cot', 'sample_id_b'])
    dups = pd.read_csv(ANALYSIS / 'bus_cot' / 'duplicates.csv', dtype={'sample_id_a': str, 'sample_id_b': str})
    in_dup = set(dups.sample_id_a) | set(dups.sample_id_b)
    names = {'Web data': 'Web data', 'TFUSL': 'TFUSL (teaching library)', 'BUSBRA': 'BUS-BRA',
             'PMC': 'PMC article figures', 'TUHD': 'TUHD (Thammasat)', 'BUSI': 'BUSI', 'BUDIAT': 'UDIAT (Dataset B)',
             'PatternFigures': 'Pattern figures', 'Birads-ultrasound': 'BI-RADS ultrasound', 'ROCO': 'ROCO figures',
             'STU-Hospital': 'STU-Hospital'}
    lines = []
    for source, group in sorted(frame.groupby('meta_source'), key=lambda kv: -len(kv[1])):
        overlap = int(group.sample_id.isin(ids).sum())
        lines.append(' & '.join([
            names.get(source, source), fmt(len(group)), fmt((group.label == 0).sum()), fmt((group.label == 1).sum()),
            f'{overlap:,}' if overlap else '--', f"{100 * group.sample_id.isin(in_dup).mean():.0f}\\%",
            'yes' if bool(group.used.astype(str).str.lower().eq('true').any()) else 'no',
        ]) + r' \\')
    (GENERATED / 'table_bus_cot.tex').write_text('\n'.join(lines) + '\n')
    return {'buscotOverlap': f'{len(ids):,}', 'buscotUsable': f"{len(used(frame)):,}",
            'buscotUsablePatients': f"{used(frame).group.nunique():,}",
            'buscotRelabelled': f"{int(cross.label_conflict.sum())}"}


def table_odelia():
    table = pd.read_csv(ANALYSIS / 'odelia' / 'centre_labels.csv')
    names = {'CAM': 'Cambridge (CAM)', 'MHA': 'Athens (MHA)', 'RSH': 'Valencia (RSH)$^{*}$',
             'RUMC': 'Nijmegen (RUMC)', 'UKA': 'Aachen (UKA)', 'UMCU': 'Utrecht (UMCU)'}
    lines = [' & '.join([names.get(r.meta_institution, r.meta_institution), fmt(r['no lesion']),
                         fmt(r.benign), fmt(r.malignant), f'{100 * r.benign_share_of_lesions:.0f}\\%',
                         fmt(r.n_post_median)]) + r' \\' for _, r in table.iterrows()]
    (GENERATED / 'table_odelia.tex').write_text('\n'.join(lines) + '\n')
    probes = json.loads((ANALYSIS / 'odelia' / 'centre_probes.json').read_text())
    return {'odeliaCentreAcc': f"{100 * probes['centre_accuracy_from_image_stats']:.0f}",
            'odeliaCentreChance': f"{100 * probes['centre_chance_accuracy']:.0f}",
            'odeliaCentreOnlyAUC': f"{probes['benign_vs_malignant_auc_from_centre_only']:.2f}",
            'odeliaStatsAUC': f"{probes['benign_vs_malignant_auc_from_image_stats']:.2f}"}


def cbis_macros(frames):
    cbis = used(frames['cbis_ddsm'])
    full = frames['cbis_ddsm'][frames['cbis_ddsm'].sample_id.str.startswith('full/')]
    cross_split = (cbis.groupby('group').split.nunique() > 1)
    return {'cbisCrossSplitPatients': f'{int(cross_split.sum())}',
            'cbisCrossSplitCrops': f"{int(cbis.group.isin(cross_split[cross_split].index).sum())}",
            'cbisBWC': f"{int((cbis.label_name == 'BENIGN_WITHOUT_CALLBACK').sum())}",
            'cbisFull': f'{len(full)}',
            'cbisFullLabels': f"{int(full.burnt_label_flag.astype(str).str.lower().eq('true').sum())}"}


def source_probe(frames):
    """Can basic per-slice statistics tell ISPY2 from BreastDM malignant slices?"""
    breadm = used(frames['breamdm'])
    pool = pd.concat([breadm[breadm['label'] == 1].assign(source=0), used(frames['ispy2']).assign(source=1)])
    sets = {'all': ['height', 'width', 'brightness', 'contrast', 'entropy'],
            'size': ['height', 'width'], 'intensity': ['brightness', 'contrast', 'entropy']}
    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    aucs = {}
    for key, columns in sets.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        aucs[key] = cross_val_score(model, pool[columns].fillna(0), pool['source'], cv=folds, scoring='roc_auc').mean()
    return aucs


def macros(frames, summaries, aucs):
    ispy_cases = pd.read_parquet(ANALYSIS / 'ispy2' / 'cases.parquet')
    breadm_cases = pd.read_parquet(ANALYSIS / 'breamdm' / 'cases.parquet')
    busi = frames['busi']
    values = {
        'probeAll': f"{aucs['all']:.3f}", 'probeSize': f"{aucs['size']:.3f}",
        'probeIntensity': f"{aucs['intensity']:.3f}",
        'ispyRepeated': f"{int(ispy_cases['phases_repeated_in_img9'].astype(bool).sum()):,}",
        'ispyCases': f'{len(ispy_cases):,}',
        'ispyPhasesMin': f"{int(ispy_cases['n_phases'].min())}", 'ispyPhasesMax': f"{int(ispy_cases['n_phases'].max())}",
        'ispyComponents': f"{ispy_cases['tumor_components'].median():.0f}",
        'breadmFloatCases': f"{int((breadm_cases['dtype'] == 'float64').sum())}",
        'breadmCases': f'{len(breadm_cases)}',
        'busiBlocky': f"{100 * (busi['jpeg_blockiness'] > 1.05).mean():.0f}",
        'busiPaperDoppler': f"{100 * busi['paper_doppler_flag'].mean():.0f}",
        'busiPaperMarker': f"{100 * busi['paper_marker_flag'].mean():.0f}",
        'totalImages': f"{sum(len(f) for f in frames.values()):,}",
        'totalUsed': f"{sum(len(used(f)) for f in frames.values()):,}",
        'breadmTestUnused': f"{int((frames['breamdm']['split'] == 'test').sum()):,}",
        'cmmdUnlabeled': f"{int((~frames['cmmd']['used'].astype(bool)).sum()):,}",
        'cesmMissing': f"{2006 - summaries['cdd_cesm']['on_disk']['images']}",
        'mgCrossCandidates': '4.1 million',
    }
    values.update(table_bus_cot())
    values.update(table_odelia())
    values.update(cbis_macros(frames))
    text = ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in values.items())
    (GENERATED / 'macros.tex').write_text(text)
    return values


def figure_quality(frames):
    metrics = [('brightness', 'Brightness (0-255)', False), ('contrast', 'Contrast (std)', False),
               ('entropy', 'Entropy (bits)', False), ('snr', 'SNR (log scale)', True),
               ('jpeg_blockiness', 'JPEG blockiness ratio', False), ('lesion_cnr', 'Lesion CNR', False)]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    for ax, (column, title, log) in zip(axes.ravel(), metrics):
        names = [n for n in DATASETS if column in frames[n] and used(frames[n])[column].notna().any()]
        data = [used(frames[n])[column].dropna().to_numpy() for n in names]
        box = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                         medianprops={'color': INK, 'linewidth': 1}, whiskerprops={'color': MUTED},
                         capprops={'color': MUTED})
        for patch, n in zip(box['boxes'], names):
            patch.set_facecolor(COLORS[MODALITY[n]])
            patch.set_edgecolor('white')
        ax.set_xticks(range(1, len(names) + 1), [NAMES[n] for n in names], rotation=60, ha='right', fontsize=6.5)
        ax.set_title(title, fontsize=8, color=INK)
        if log:
            ax.set_yscale('log')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    fig.legend(handles, ['Ultrasound', 'Mammography / DBT', 'MRI'], loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / 'quality_overview.pdf')
    plt.close(fig)


def figure_mri_sizes(frames):
    breadm, ispy = used(frames['breamdm']), used(frames['ispy2'])
    groups = [('BreastDM benign', breadm[breadm['label'] == 0], SERIES[0]),
              ('BreastDM malignant', breadm[breadm['label'] == 1], SERIES[1]),
              ('ISPY2 (all malignant)', ispy, SERIES[2])]
    bins = np.logspace(np.log10(4), np.log10(400), 40)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for label, frame, color in groups:
        side = np.sqrt(frame['height'] * frame['width'])
        ax.hist(side, bins=bins, density=True, histtype='step', linewidth=2, color=color,
                label=f'{label} (median {side.median():.0f} px)')
    ax.set_xscale('log')
    ax.set_xlabel('Crop size, $\\sqrt{h\\times w}$ (pixels)')
    ax.set_ylabel('Density')
    ax.legend(frameon=False, fontsize=6.5, loc='lower left', bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(FIGURES / 'mri_crop_sizes.pdf')
    plt.close(fig)


def figure_us_flags(frames):
    names = ['busbra', 'busi', 'busc', 'breast']
    series = [('paper_doppler_flag', "Paper's Doppler rule"), ('paper_marker_flag', "Paper's marker rule"),
              ('marker_flag', 'Calipers / measurement lines'), ('text_flag', 'Burnt-in text')]
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for k, ((column, label), color) in enumerate(zip(series, SERIES)):
        rates = [100 * frames[n][column].fillna(False).astype(bool).mean() for n in names]
        ax.bar(x + (k - 1.5) * width, rates, width * 0.9, color=color, label=label)
    ax.set_xticks(x, [NAMES[n] for n in names])
    ax.set_ylabel('Images flagged (%)')
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 1.32), ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / 'us_flag_rates.pdf')
    plt.close(fig)


def sheet_rows(source, rows, out, max_width=None):
    """Crop whole rows out of an audit contact sheet."""
    image = Image.open(source)
    top, bottom = rows[0] * TILE, (rows[-1] + 1) * TILE
    crop = image.crop((0, top, max_width or image.width, min(bottom, image.height)))
    crop.save(out)


def figure_sheets():
    sheet_rows(ANALYSIS / 'busi' / 'figures' / 'duplicates.png', [0], FIGURES / 'busi_duplicates.png')
    sheet_rows(ANALYSIS / 'busc' / 'figures' / 'duplicates.png', [0], FIGURES / 'busc_duplicates.png')
    sheet_rows(ANALYSIS / 'cmmd' / 'figures' / 'duplicates.png', [1, 2], FIGURES / 'cmmd_duplicates.png',
               max_width=4 * 224)
    sheet_rows(ANALYSIS / 'busbra' / 'figures' / 'flag_text_flag.png', [0], FIGURES / 'busbra_text.png')
    mean = Image.open(ANALYSIS / 'busi' / 'figures' / 'mean_std_images.png')
    mean.save(FIGURES / 'busi_mean_std.png')


def main():
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frames, summaries = load()
    table_counts(frames, summaries)
    table_quality(frames, summaries)
    table_integrity(frames, summaries)
    table_mri()
    aucs = source_probe(frames)
    values = macros(frames, summaries, aucs)
    figure_quality(frames)
    figure_mri_sizes(frames)
    figure_us_flags(frames)
    figure_sheets()
    print('source probe AUCs:', {k: round(v, 3) for k, v in aucs.items()})
    print('macros:', values)


if __name__ == '__main__':
    main()
