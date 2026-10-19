"""Rebuild the PNG caches that training reads for CBIS-DDSM and the full
BCS-DBT set, from their raw DICOMs. This is the `prepare` command in
data/sources/registry.yaml, so the caches can be regenerated after the raw
data has been pruned and re-fetched.

Usage:
    python3 scripts/prepare_mammography_cache.py cbis_ddsm
    python3 scripts/prepare_mammography_cache.py bcsdbt_full
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import train as T


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', choices=('cbis_ddsm', 'bcsdbt_full'))
    args = parser.parse_args(argv)
    if args.dataset == 'cbis_ddsm':
        # include_bwc=True caches every crop; training drops BENIGN_WITHOUT_CALLBACK by default
        metadata = T._cbis_ddsm_metadata(Path('/mnt/data/mmfm_datasets/mammography/cbis_ddsm'), include_bwc=True)
        print(f"cbis_ddsm: {len(metadata)} crops cached ({metadata['pathology'].value_counts().to_dict()})")
    else:
        metadata = T._dbt_full_metadata(T.DATASET_KFOLD_SPECS['bcsdbt_full']['default_root'])
        print(f"bcsdbt_full: {len(metadata)} slices from {metadata['patient'].nunique()} patients cached")


if __name__ == '__main__':
    main()
