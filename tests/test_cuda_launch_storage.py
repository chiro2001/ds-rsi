import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "run_5080_checked", Path(__file__).resolve().parents[1] / "scripts/run_5080_checked.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_storage_paths_accept_external_root_and_relative_snapshot_files(tmp_path):
    root = tmp_path / "external"
    code = root / "code/commit"
    code.mkdir(parents=True)
    module.check_storage_paths(["--run-dir", str(root / "runs/A"), "--manifest", "data/manifest.json"], code, root)


def test_storage_paths_reject_internal_or_neighbor_directory(tmp_path):
    root = tmp_path / "external"
    root.mkdir()
    with pytest.raises(RuntimeError, match="Windows"):
        module.check_storage_paths(["--run-dir", str(tmp_path / "external-other/run")], root, root)


def test_storage_paths_resolve_symlinks_before_accepting(tmp_path):
    root = tmp_path / "external"
    root.mkdir()
    (root / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(RuntimeError, match="Windows"):
        module.check_storage_paths(["--output", str(root / "link/report.json")], root, root)
