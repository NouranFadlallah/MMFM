#!/usr/bin/env python3
"""Manage local datasets from data/sources/registry.yaml.

    list                     one line per dataset: modality, license, where it lives
    status [NAME ...]        present? size? checksummed? backed up?
    checksum NAME|all        write data/sources/checksums/<name>.sha256 for raw_dir
    verify NAME|all          re-hash raw_dir and compare with the stored checksums
    fetch NAME [--dest DIR]  re-download raw data (zenodo / direct / tcia / idc);
                             prints the manual steps for gated sources
    backup NAME|all          upload the training inputs to a private HF dataset repo
    restore NAME             download the HF backup back into place

Checksum files use `sha256sum` format with paths relative to raw_dir, so
`cd <raw_dir> && sha256sum -c <file>` also works without this script.
"""
import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / 'data' / 'sources' / 'registry.yaml'
CHECKSUMS = REPO_ROOT / 'data' / 'sources' / 'checksums'
CHUNK = 8 * 1024 * 1024


def load_registry():
    return yaml.safe_load(REGISTRY.read_text())


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def files_under(root):
    root = resolve(root)
    return sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink())


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def size_of(paths):
    return sum(p.stat().st_size for p in paths)


def human(n):
    for unit in ('B', 'K', 'M', 'G', 'T'):
        if n < 1024:
            return f'{n:.1f}{unit}'
        n /= 1024
    return f'{n:.1f}P'


def _hash_all(root, paths, workers):
    with ThreadPoolExecutor(workers) as pool:
        for k, (path, digest) in enumerate(zip(paths, pool.map(sha256, paths)), 1):
            if k % 500 == 0 or k == len(paths):
                print(f'  {k}/{len(paths)} files', flush=True)
            yield path.relative_to(root).as_posix(), digest


def cmd_checksum(names, registry, workers):
    CHECKSUMS.mkdir(parents=True, exist_ok=True)
    for name in names:
        entry = registry[name]
        root = resolve(entry['raw_dir'])
        if not root.exists():
            print(f'{name}: raw_dir missing ({root}); skipped')
            continue
        paths = files_under(root)
        print(f'{name}: hashing {len(paths)} files, {human(size_of(paths))}', flush=True)
        out = CHECKSUMS / f'{name}.sha256'
        tmp = out.with_suffix('.sha256.partial')
        with open(tmp, 'w') as handle:
            for rel, digest in _hash_all(root, paths, workers):
                handle.write(f'{digest}  {rel}\n')
            for extra in entry.get('extra_files', []):
                extra = resolve(extra)
                if extra.exists():
                    handle.write(f'{sha256(extra)}  {extra}\n')
        tmp.replace(out)
        print(f'{name}: wrote {out}')


def cmd_verify(names, registry, workers):
    failures = 0
    for name in names:
        out = CHECKSUMS / f'{name}.sha256'
        if not out.exists():
            print(f'{name}: no checksum file; run `checksum {name}` first')
            failures += 1
            continue
        root = resolve(registry[name]['raw_dir'])
        expected = {}
        for line in out.read_text().splitlines():
            digest, rel = line.split('  ', 1)
            expected[rel] = digest
        paths = [Path(rel) if Path(rel).is_absolute() else root / rel for rel in expected]
        missing = [p for p in paths if not p.exists()]
        present = [p for p in paths if p.exists()]
        with ThreadPoolExecutor(workers) as pool:
            digests = list(pool.map(sha256, present))
        keys = [p.as_posix() if p.is_absolute() and not p.is_relative_to(root) else p.relative_to(root).as_posix()
                for p in present]
        changed = [k for k, d in zip(keys, digests) if expected[k] != d]
        status = 'OK' if not missing and not changed else 'MISMATCH'
        if missing and len(missing) == len(paths):
            # a re-download (e.g. TCIA vs NBIA retriever) can use a different
            # folder layout: fall back to matching file contents regardless of path
            with ThreadPoolExecutor(workers) as pool:
                found = set(pool.map(sha256, files_under(root)))
            absent = [k for k, d in expected.items() if d not in found]
            status = 'OK (by content)' if not absent else 'MISMATCH'
            print(f'{name}: {status} ({len(expected) - len(absent)}/{len(expected)} files matched by content)')
            failures += bool(absent)
            continue
        print(f'{name}: {status} ({len(present)} files checked, {len(missing)} missing, {len(changed)} changed)')
        failures += status != 'OK'
    return failures


def _hf_repo_exists(api, repo):
    try:
        api.repo_info(repo, repo_type='dataset')
        return True
    except Exception:
        return False


def cmd_status(names, registry):
    try:
        from huggingface_hub import HfApi
        api = HfApi()
    except Exception:
        api = None
    print(f"{'dataset':10} {'raw':>9} {'prepared':>9}  checksums  backup")
    for name in names:
        entry = registry[name]
        raw = resolve(entry['raw_dir'])
        raw_size = human(size_of(files_under(raw))) if raw.exists() else 'missing'
        prepared = entry.get('prepared')
        prep_size = human(size_of(files_under(prepared))) if prepared and resolve(prepared).exists() else '-'
        checksum = 'yes' if (CHECKSUMS / f'{name}.sha256').exists() else 'no'
        repo = entry.get('backup')
        backup = 'not allowed' if not repo else ('yes' if api and _hf_repo_exists(api, repo) else 'no')
        print(f'{name:10} {raw_size:>9} {prep_size:>9}  {checksum:9}  {backup}')


def cmd_list(registry):
    for name, entry in registry.items():
        print(f"{name:10} {entry['modality']:22} {entry['license']:18} {entry['source']['type']:7} {entry['raw_dir']}")


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'  downloading {url}')
    with urllib.request.urlopen(url) as response, open(dest, 'wb') as handle:
        while block := response.read(CHUNK):
            handle.write(block)


RANGE_CHUNK = 64 * 1024 * 1024


def _content_length(url):
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = response.headers.get('Content-Range', '').rsplit('/', 1)[-1]
    return int(total) if total.isdigit() else None


def _download_ranged(url, dest, md5=None, workers=4):
    """Parallel, resumable download in 64 MB byte ranges. Each range re-requests
    the URL, so short-lived redirect links (e.g. Figshare -> S3, valid ~10 s)
    keep working. Finished ranges are recorded in <dest>.done, so an interrupted
    download resumes where it stopped. Verifies md5 at the end if given."""
    import time

    size = _content_length(url)
    if size is None:
        return _download(url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size != size:
        with open(dest, 'wb') as handle:
            handle.truncate(size)
    done_file = dest.with_name(dest.name + '.done')
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    parts = [(start, min(start + RANGE_CHUNK, size) - 1) for start in range(0, size, RANGE_CHUNK)
             if str(start) not in done]
    print(f'  {dest.name}: {human(size)}, {len(parts)} of {-(-size // RANGE_CHUNK)} ranges to fetch')

    def fetch(part):
        start, end = part
        for attempt in range(8):
            try:
                request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=300) as response:
                    data = response.read()
                if len(data) != end - start + 1:
                    raise IOError(f'short read {len(data)}')
                with open(dest, 'r+b') as handle:
                    handle.seek(start)
                    handle.write(data)
                with open(done_file, 'a') as handle:
                    handle.write(f'{start}\n')
                return
            except Exception as error:
                print(f'  range {start}: attempt {attempt + 1} failed ({error})', flush=True)
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f'range starting at {start} failed after 8 attempts')

    with ThreadPoolExecutor(workers) as pool:
        for k, _ in enumerate(pool.map(fetch, parts), 1):
            if k % 5 == 0 or k == len(parts):
                print(f'  {k}/{len(parts)} ranges', flush=True)
    if md5:
        digest = hashlib.md5()
        with open(dest, 'rb') as handle:
            while block := handle.read(CHUNK):
                digest.update(block)
        if digest.hexdigest() != md5:
            raise RuntimeError(f'{dest.name}: md5 mismatch ({digest.hexdigest()} != {md5})')
        print(f'  {dest.name}: md5 OK')
    done_file.unlink(missing_ok=True)


def cmd_fetch(name, registry, dest=None, used_only=False):
    entry = registry[name]
    source = entry['source']
    dest = resolve(dest or entry['raw_dir'])
    kind = source['type']
    if kind == 'manual':
        print(f"{name}: manual download required.\n  {source['landing']}\n  {source.get('steps', '')}\n  -> {dest}")
        return
    if kind == 'direct':
        archive = dest / source.get('filename', Path(source['url']).name)
        _download_ranged(source['url'], archive, md5=source.get('md5'))
        if archive.suffix == '.zip':
            import zipfile
            zipfile.ZipFile(archive).extractall(dest / source.get('unpack_dir', 'extracted'))
        if archive.name.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive) as tar:
                tar.extractall(resolve(source.get('unpack_to', dest)), filter='data')
        return
    if kind == 'zenodo':
        import json
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{source['record']}") as response:
            record = json.load(response)
        for item in record['files']:
            target = dest / item['key']
            _download(item['links']['self'], target)
            if target.suffix == '.zip':
                import zipfile
                zipfile.ZipFile(target).extractall(dest)
        return
    series_file = resolve(source['series_list_used' if used_only and 'series_list_used' in source else 'series_list'])
    series = [s for s in series_file.read_text().split() if s]
    if kind == 'idc':
        from idc_index import IDCClient  # pip install idc-index
        IDCClient().download_from_selection(seriesInstanceUID=series, downloadDir=str(dest),
                                            dirTemplate='%PatientID/%StudyInstanceUID/%SeriesInstanceUID')
        return
    if kind == 'tcia':
        import io
        import zipfile
        base = 'https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID='
        for k, uid in enumerate(series, 1):
            print(f'  series {k}/{len(series)}')
            with urllib.request.urlopen(base + uid) as response:
                zipfile.ZipFile(io.BytesIO(response.read())).extractall(dest / uid)
        return
    raise ValueError(f'unknown source type {kind}')


def _backup_paths(entry):
    if entry.get('backup_paths'):
        return [resolve(p) for p in entry['backup_paths']]
    paths = [resolve(entry['prepared'] or entry['raw_dir'])]
    if entry.get('prepared') is None:
        paths += [resolve(p) for p in entry.get('extra_files', [])]
    return paths


def cmd_backup(names, registry, dry_run):
    from huggingface_hub import HfApi
    api = HfApi()
    for name in names:
        entry = registry[name]
        repo = entry.get('backup')
        if not repo:
            print(f'{name}: backup not allowed by license ({entry["license"]}); skipped')
            continue
        paths = [p for p in _backup_paths(entry) if p.exists()]
        size = sum(size_of(files_under(p)) if p.is_dir() else p.stat().st_size for p in paths)
        print(f'{name}: {human(size)} -> hf://datasets/{repo} (private)')
        if dry_run:
            continue
        api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
        for path in paths:
            if path.is_dir():
                api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=str(path),
                                  path_in_repo=path.name, commit_message=f'Back up {path.name}')
            else:
                api.upload_file(path_or_fileobj=str(path), path_in_repo=f'extra/{path.name}',
                                repo_id=repo, repo_type='dataset')
        checksum = CHECKSUMS / f'{name}.sha256'
        if checksum.exists():
            api.upload_file(path_or_fileobj=str(checksum), path_in_repo=f'raw_checksums/{checksum.name}',
                            repo_id=repo, repo_type='dataset')


def cmd_restore(name, registry, dest=None):
    from huggingface_hub import snapshot_download
    entry = registry[name]
    target = resolve(dest) if dest else REPO_ROOT / 'datasets' / 'restored' / name
    snapshot_download(entry['backup'], repo_type='dataset', local_dir=str(target))
    print(f'{name}: restored to {target}')
    for path in _backup_paths(entry):
        print(f'  {path.name}: move {target / (path.name if path.is_dir() else "extra/" + path.name)} -> {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list')
    for command in ('status', 'checksum', 'verify', 'backup'):
        p = sub.add_parser(command)
        p.add_argument('names', nargs='*', default=['all'])
        p.add_argument('--workers', type=int, default=4)
        if command == 'backup':
            p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('fetch')
    p.add_argument('name')
    p.add_argument('--dest')
    p.add_argument('--used-only', action='store_true', help='Only the series training uses (BCS-DBT).')
    p = sub.add_parser('restore')
    p.add_argument('name')
    p.add_argument('--dest')
    args = parser.parse_args()

    registry = load_registry()
    names = getattr(args, 'names', None)
    if names is not None:
        names = list(registry) if 'all' in names else names
        unknown = set(names) - set(registry)
        if unknown:
            parser.error(f'unknown dataset(s): {sorted(unknown)}')

    if args.command == 'list':
        cmd_list(registry)
    elif args.command == 'status':
        cmd_status(names, registry)
    elif args.command == 'checksum':
        cmd_checksum(names, registry, args.workers)
    elif args.command == 'verify':
        sys.exit(1 if cmd_verify(names, registry, args.workers) else 0)
    elif args.command == 'fetch':
        cmd_fetch(args.name, registry, args.dest, args.used_only)
    elif args.command == 'backup':
        cmd_backup(names, registry, args.dry_run)
    elif args.command == 'restore':
        cmd_restore(args.name, registry, args.dest)


if __name__ == '__main__':
    os.chdir(REPO_ROOT)
    main()
