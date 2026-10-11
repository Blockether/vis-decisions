"""Explicit offline D3 checkpoints and fine-tuning of the code readout model.

Importing this module is lightweight. A D3 training checkpoint keeps the upstream
layout without the vision tower: the Qwen3.5 text weights with their
``language_model.`` prefix, the 255-way readout, the settings and the tokenizer.
"""

from __future__ import annotations

import json
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

from blockether.vis._contracts import definition

from ._models import D3, D3_FORMAT
from ._trainer import TrainingResult
from .decision2_training import _pinned
from .training import (
    TrainingBundle,
    _inventory,
    _safe_name,
    _sha256,
    _valid_partial,
)

_REQUIRED = {
    "config.json",
    "decision_config.json",
    "MODEL_MANIFEST.json",
    "model.safetensors",
    "readout.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
}
_MAX_EXPANDED_BYTES = definition("gateway", "decision_expanded_bytes")["maximum"]
_PREFIX = "language_model."


def _checkpoint_model(root: Path) -> str:
    """Return the D3 model whose manifest, readout settings and backbone match."""
    manifest = json.loads((root / "MODEL_MANIFEST.json").read_text("utf-8"))
    metadata = json.loads((root / "decision_config.json").read_text("utf-8"))
    config = json.loads((root / "config.json").read_text("utf-8"))
    model_id = manifest.get("model_name") if isinstance(manifest, dict) else None
    if (
        model_id not in D3
        or not isinstance(metadata, dict)
        or not isinstance(config, dict)
        or manifest.get("repo_id") != f"vllm-sr/{model_id}"
        or config.get("model_type") != "qwen3_5"
        or not isinstance(config.get("text_config"), dict)
        or metadata.get("format_id") != D3_FORMAT
        or metadata.get("prompt") != "d3"
        or metadata.get("attention_mode") != "noncausal_full_attention"
        or metadata.get("pooling", "last") != "last"
        or metadata.get("readout_dtype", "float32") != "float32"
        or metadata.get("temperature", 1.0) != 1.0
        or len(metadata.get("codes", ())) != 255
    ):
        raise ValueError("Checkpoint is not a supported D3 code-readout model")
    return model_id


def _tensor_names(path: Path) -> list[str]:
    """Read the tensor names of a safetensors file from its header only."""
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    return [name for name in header if name != "__metadata__"]


def _copy_text_weights(source: Path, destination: Path) -> None:
    """Write ``model.safetensors`` with only the text weights of ``source``."""
    files = sorted(source.glob("model*.safetensors"))
    if not files or any(path.is_symlink() for path in files):
        raise FileNotFoundError("D3 checkpoint weights are missing")
    if [path.name for path in files] == ["model.safetensors"] and all(
        name.startswith(_PREFIX) for name in _tensor_names(files[0])
    ):
        shutil.copyfile(files[0], destination)
        return
    from safetensors.torch import load_file, save_file

    weights = {}
    for path in files:
        for name, value in load_file(path).items():
            if name.startswith(_PREFIX):
                weights[name] = value
    if not weights:
        raise ValueError("D3 checkpoint has no text weights")
    save_file(weights, str(destination))


@dataclass(frozen=True)
class D3TrainingBundle(TrainingBundle):
    """A verified D3 text checkpoint with its readout, never the exported ONNX graph."""

    @property
    def model_id(self) -> str:
        return json.loads((self.path / "PROVENANCE.json").read_text(encoding="utf-8"))[
            "model"
        ]

    @classmethod
    def open(cls, path: str | Path) -> D3TrainingBundle:
        """Check every file and its digest without loading Torch or accessing the network."""
        root = Path(path).expanduser().resolve()
        provenance_path = root / "PROVENANCE.json"
        license_path = root / "LICENSE.txt"
        if (
            not root.is_dir()
            or not provenance_path.is_file()
            or provenance_path.is_symlink()
            or provenance_path.stat().st_size > 1_048_576
            or not license_path.is_file()
            or license_path.is_symlink()
        ):
            raise FileNotFoundError(
                "Complete D3 checkpoint provenance and license required"
            )
        metadata = json.loads(provenance_path.read_text(encoding="utf-8"))
        model_id = metadata.get("model")
        if (
            metadata.get("schema_version") != 1
            or metadata.get("kind") != "training"
            or metadata.get("format") != "safetensors"
            or metadata.get("family") != "d3"
            or metadata.get("license") != "Apache-2.0"
            or model_id not in D3
            or metadata.get("architecture") != D3[model_id]
            or metadata.get("prompt_version") != D3_FORMAT
            or not _pinned(metadata.get("revision"))
            or (
                "parent_revision" in metadata
                and not _pinned(metadata["parent_revision"])
            )
            or ("partial" in metadata and not _valid_partial(metadata["partial"]))
        ):
            raise ValueError("Unsupported D3 checkpoint identity or architecture")
        declared = metadata.get("files")
        if not isinstance(declared, dict) or set(declared) != _REQUIRED:
            raise ValueError(
                "D3 checkpoint needs exactly its weights, readout, settings and tokenizer"
            )
        actual: set[str] = set()
        for item in root.rglob("*"):
            if item.is_symlink():
                raise ValueError("D3 checkpoint contains a symlink")
            if item.is_file():
                actual.add(item.relative_to(root).as_posix())
        if actual - {"PROVENANCE.json", "LICENSE.txt", ".vis-verified"} != _REQUIRED:
            raise ValueError("D3 checkpoint has untracked or missing files")
        size = 0
        for name, detail in declared.items():
            target = root.joinpath(*_safe_name(name).parts)
            if not isinstance(detail, dict):
                raise ValueError("D3 checkpoint inventory is invalid")
            size += target.stat().st_size
            if (
                size > _MAX_EXPANDED_BYTES
                or detail.get("bytes") != target.stat().st_size
                or detail.get("sha256") != _sha256(target)
            ):
                raise ValueError(f"D3 checkpoint checksum failed: {name}")
        if _checkpoint_model(root) != model_id:
            raise ValueError("D3 checkpoint settings belong to another model")
        return cls(root)

    @classmethod
    def from_local(
        cls,
        source: str | Path,
        destination: str | Path,
        *,
        model_id: str,
        revision: str,
        license_file: str | Path,
        parent_revision: str | None = None,
        partial: dict | None = None,
    ) -> D3TrainingBundle:
        """Atomically inventory an explicit local D3 checkpoint without downloading.

        ``source`` has the upstream layout: ``config.json``, ``model*.safetensors``,
        ``readout.safetensors``, ``decision_config.json``, ``MODEL_MANIFEST.json`` and
        the tokenizer files. Only the ``language_model.`` text weights are kept.
        ``partial`` marks unfinished training with its step and run digest, so that
        ``D3Trainer.finetune`` can resume the same run from this checkpoint.
        """
        if model_id not in D3:
            raise ValueError("Unsupported D3 checkpoint model")
        if not _pinned(revision):
            raise ValueError("A pinned checkpoint revision is required")
        source = Path(source).expanduser()
        destination = Path(destination).expanduser().resolve()
        license_file = Path(license_file).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(destination)
        if source.is_symlink() or not source.is_dir() or not license_file.is_file():
            raise FileNotFoundError(
                "Local D3 checkpoint and Apache-2.0 license required"
            )
        if any(
            not (source / name).is_file() or (source / name).is_symlink()
            for name in _REQUIRED - {"model.safetensors"}
        ):
            raise FileNotFoundError(
                "Complete D3 weights, readout, settings and tokenizer required"
            )
        if _checkpoint_model(source) != model_id:
            raise ValueError(f"Checkpoint is not a {model_id} model")
        if parent_revision is not None and not _pinned(parent_revision):
            raise ValueError("Invalid parent checkpoint revision")
        if partial is not None and not _valid_partial(partial):
            raise ValueError("Invalid partial training progress")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".d3-checkpoint-", dir=destination.parent
        ) as temporary:
            staging = Path(temporary) / "checkpoint"
            staging.mkdir()
            for name in sorted(_REQUIRED - {"model.safetensors"}):
                shutil.copyfile(source / name, staging / name)
            _copy_text_weights(source, staging / "model.safetensors")
            shutil.copyfile(license_file, staging / "LICENSE.txt")
            metadata = {
                "schema_version": 1,
                "kind": "training",
                "format": "safetensors",
                "family": "d3",
                "architecture": D3[model_id],
                "prompt_version": D3_FORMAT,
                "model": model_id,
                "revision": revision,
                "license": "Apache-2.0",
                "files": _inventory(staging),
            }
            if parent_revision is not None:
                metadata["parent_revision"] = parent_revision
            if partial is not None:
                metadata["partial"] = partial
            (staging / "PROVENANCE.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            cls.open(staging)
            staging.rename(destination)
        return cls.open(destination)


from ._d3_trainer import D3Trainer  # noqa: E402

__all__ = ["D3Trainer", "D3TrainingBundle", "TrainingResult"]
