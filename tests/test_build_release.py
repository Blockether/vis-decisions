"""The public asset builder must reject untracked bytes and reproduce archives."""

import hashlib
import json
import zipfile
from fnmatch import fnmatch
from pathlib import Path

import pytest

from scripts.build_release import inventory, pack


def prepared_bundle(root):
    root.mkdir()
    payloads = {
        "model.onnx": b"complete FP32 model graph",
        "rl_agent_config.json": b"{}",
        "tokenizer/tokenizer.json": b"{}",
    }
    for name, data in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (root / "LICENSE.txt").write_text("Apache-2.0 test notice")
    provenance = {
        "kind": "inference",
        "format": "onnx",
        "precision": "fp32",
        "model": "test-model",
        "revision": "test-revision",
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in payloads.items()
        },
    }
    (root / "PROVENANCE.json").write_text(json.dumps(provenance))
    return root


def test_archive_is_reproducible_and_contains_exactly_declared_files(tmp_path):
    source = prepared_bundle(tmp_path / "source")
    (source / ".vis-verified").write_text("installer marker, not release content")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    assert pack(source, first) == pack(source, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert sorted(archive.namelist()) == [name for name, _ in inventory(source)]
        assert "model.onnx" in archive.namelist()
        assert ".vis-verified" not in archive.namelist()


def test_corruption_or_untracked_file_aborts_before_publication(tmp_path):
    source = prepared_bundle(tmp_path / "source")
    output = tmp_path / "output.zip"
    (source / "model.onnx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        pack(source, output)
    assert not output.exists()
    (source / "model.onnx").write_bytes(b"complete FP32 model graph")
    (source / "private-data.txt").write_text("must not publish")
    with pytest.raises(ValueError, match="untracked"):
        pack(source, output)
    assert not output.exists()


def test_refuses_symlinks_and_non_fp32_graphs(tmp_path):
    source = prepared_bundle(tmp_path / "source")
    (source / "link").symlink_to(source / "model.onnx")
    with pytest.raises(ValueError, match="symlink"):
        pack(source, tmp_path / "bad.zip")
    (source / "link").unlink()
    path = source / "PROVENANCE.json"
    provenance = json.loads(path.read_text())
    provenance["precision"] = "int8"
    path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="FP32"):
        pack(source, tmp_path / "bad.zip")


def test_release_builder_rejects_dependency_archives(tmp_path):
    source = tmp_path / "dependencies"
    source.mkdir()
    (source / "PROVENANCE.json").write_text(
        json.dumps({"kind": "training-dependencies"})
    )
    with pytest.raises(ValueError, match="Unknown bundle kind"):
        pack(source, tmp_path / "dependencies.zip")


def test_extension_release_publishes_only_distribution_archives():
    workflow = Path(__file__).parents[1] / ".github/workflows/ci.yml"
    value = workflow.read_text().rsplit("files:", 1)[1].strip()
    patterns = [line.strip() for line in value.splitlines() if line.strip() != "|"]
    expected = {
        "dist/vis_decisions-0.1.0-py3-none-any.whl",
        "dist/vis_decisions-0.1.0.tar.gz",
    }
    candidates = expected | {"dist/default.gitignore", "dist/private-data.txt"}
    selected = {
        name
        for name in candidates
        if any(fnmatch(name, pattern) for pattern in patterns)
    }
    assert selected == expected
