import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('datasets_cli', Path(__file__).resolve().parents[1] / 'scripts' / 'datasets.py')
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    (raw / 'a').mkdir(parents=True)
    (raw / 'a' / 'one.bin').write_bytes(b'one')
    (raw / 'two.bin').write_bytes(b'two')
    monkeypatch.setattr(cli, 'CHECKSUMS', tmp_path / 'checksums')
    return raw, {'toy': {'raw_dir': str(raw)}}


def test_checksum_then_verify_ok(toy):
    raw, registry = toy
    cli.cmd_checksum(['toy'], registry, workers=2)
    lines = (cli.CHECKSUMS / 'toy.sha256').read_text().splitlines()
    assert sorted(line.split('  ')[1] for line in lines) == ['a/one.bin', 'two.bin']
    assert cli.cmd_verify(['toy'], registry, workers=2) == 0


def test_verify_detects_changed_file(toy):
    raw, registry = toy
    cli.cmd_checksum(['toy'], registry, workers=2)
    (raw / 'two.bin').write_bytes(b'tampered')
    assert cli.cmd_verify(['toy'], registry, workers=2) == 1


def test_verify_matches_by_content_after_layout_change(toy):
    raw, registry = toy
    cli.cmd_checksum(['toy'], registry, workers=2)
    # simulate a re-download that stores the same files under different names
    (raw / 'a' / 'one.bin').rename(raw / 'renamed_one.bin')
    (raw / 'two.bin').rename(raw / 'a' / 'renamed_two.bin')
    assert cli.cmd_verify(['toy'], registry, workers=2) == 0


def test_backup_paths_prefer_explicit_list():
    entry = {'raw_dir': 'datasets/x', 'prepared': 'datasets/x/prep', 'backup_paths': ['datasets/y', '/abs/file.csv']}
    assert [p.name for p in cli._backup_paths(entry)] == ['y', 'file.csv']
    entry = {'raw_dir': 'datasets/x', 'prepared': None, 'extra_files': ['/abs/labels.xlsx']}
    assert [p.name for p in cli._backup_paths(entry)] == ['x', 'labels.xlsx']
