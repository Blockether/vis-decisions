"""Catalog reads are explicit, bounded and leave model pins with Vis."""

import io
import json
import os
import subprocess
import sys

import pytest

from blockether.vis_decisions import training


def test_catalog_reads_release_without_downloading_weights(monkeypatch):
    calls = []
    catalog = [{"id": "laya-typed-decisions", "revision": "a" * 40}]

    def open_url(url, *, timeout):
        calls.append((url, timeout))
        return io.BytesIO(json.dumps(catalog).encode())

    monkeypatch.setattr(training, "urlopen", open_url)
    assert training._manifest() == catalog
    assert calls == [
        (
            "https://github.com/Blockether/vis/releases/download/assets-pack/decisions.json",
            60,
        )
    ]


@pytest.mark.parametrize("payload", [b"{}", b"[1]", b"[]" + b" " * 1_048_576])
def test_catalog_rejects_invalid_shape_or_oversize_response(monkeypatch, payload):
    monkeypatch.setattr(
        training, "urlopen", lambda *_args, **_kwargs: io.BytesIO(payload)
    )
    with pytest.raises(ValueError, match="catalog"):
        training._manifest()


def test_public_import_and_local_inventory_are_lightweight_and_offline(tmp_path):
    script = """
import socket
import sys
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unexpected network"))
from blockether.vis_decisions import Decisions, Trainer, TrainingBundle
from blockether.vis_decisions.tools import DecisionTools
assert not {"torch", "transformers", "gliner2", "laya"} & set(sys.modules)
try:
    TrainingBundle.open("missing-checkpoint")
except FileNotFoundError:
    pass
assert not {"torch", "transformers", "gliner2", "laya"} & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
