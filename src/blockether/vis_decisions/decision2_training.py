"""Explicit offline Decision 2.0 checkpoints and full CPU fine-tuning.

Importing this module is lightweight. Decision 2.0 uses the same pinned
vis-decisions environment as the Laya and GLiNER families.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from blockether.vis._contracts import definition

from ._models import DECISION2, DECISION2_PROMPT_VERSION
from ._trainer import TrainingResult
from .training import (
    TrainingBundle,
    _inventory,
    _safe_name,
    _sha256,
    _valid_partial,
)

_REQUIRED = {
    "backbone/config.json",
    "backbone/model.safetensors",
    "decision_config.json",
    "decision_head.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
}
_MAX_EXPANDED_BYTES = definition("gateway", "decision_expanded_bytes")["maximum"]


def _pinned(value: object) -> bool:
    return isinstance(value, str) and (
        re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value) is not None
    )


def _check_settings(root: Path, model_id: str) -> None:
    """Accept only the shared-head architecture, its prompt and a Qwen3.5 backbone."""
    metadata = json.loads((root / "decision_config.json").read_text(encoding="utf-8"))
    backbone = json.loads(
        (root / "backbone" / "config.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(metadata, dict)
        or not isinstance(backbone, dict)
        or metadata.get("architecture") != DECISION2[model_id]
        or metadata.get("prompt_version") != DECISION2_PROMPT_VERSION
        or metadata.get("head_variant", "shared") != "shared"
        or metadata.get("checkpoint_format", "full") != "full"
        or backbone.get("model_type") != "qwen3_5_text"
    ):
        raise ValueError(
            "Decision 2.0 checkpoint config has an incompatible architecture"
        )


@dataclass(frozen=True)
class Decision2TrainingBundle(TrainingBundle):
    """A verified full FP32 checkpoint, never the exported ONNX graph."""

    @property
    def model_id(self) -> str:
        return json.loads((self.path / "PROVENANCE.json").read_text(encoding="utf-8"))[
            "model"
        ]

    @classmethod
    def open(cls, path: str | Path) -> Decision2TrainingBundle:
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
                "Complete Decision 2.0 checkpoint provenance and license required"
            )
        metadata = json.loads(provenance_path.read_text(encoding="utf-8"))
        model_id = metadata.get("model")
        if (
            metadata.get("schema_version") != 1
            or metadata.get("kind") != "training"
            or metadata.get("format") != "safetensors"
            or metadata.get("family") != "decision2"
            or metadata.get("license") != "Apache-2.0"
            or model_id not in DECISION2
            or metadata.get("architecture") != DECISION2[model_id]
            or not _pinned(metadata.get("revision"))
            or (
                "parent_revision" in metadata
                and not _pinned(metadata["parent_revision"])
            )
            or ("partial" in metadata and not _valid_partial(metadata["partial"]))
        ):
            raise ValueError(
                "Unsupported Decision 2.0 checkpoint identity or architecture"
            )
        declared = metadata.get("files")
        if not isinstance(declared, dict) or set(declared) != _REQUIRED:
            raise ValueError(
                "Decision 2.0 checkpoint needs exactly its weights, settings and tokenizer"
            )
        actual: set[str] = set()
        for item in root.rglob("*"):
            if item.is_symlink():
                raise ValueError("Decision 2.0 checkpoint contains a symlink")
            if item.is_file():
                actual.add(item.relative_to(root).as_posix())
        if actual - {"PROVENANCE.json", "LICENSE.txt", ".vis-verified"} != _REQUIRED:
            raise ValueError("Decision 2.0 checkpoint has untracked or missing files")
        size = 0
        for name, detail in declared.items():
            target = root.joinpath(*_safe_name(name).parts)
            if not isinstance(detail, dict):
                raise ValueError("Decision 2.0 checkpoint inventory is invalid")
            size += target.stat().st_size
            if (
                size > _MAX_EXPANDED_BYTES
                or detail.get("bytes") != target.stat().st_size
                or detail.get("sha256") != _sha256(target)
            ):
                raise ValueError(f"Decision 2.0 checkpoint checksum failed: {name}")
        _check_settings(root, model_id)
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
    ) -> Decision2TrainingBundle:
        """Atomically inventory an explicit local full checkpoint without downloading.

        ``source`` has the upstream layout: ``backbone/``, ``decision_head.safetensors``,
        ``decision_config.json`` and the tokenizer files. Other upstream files stay out.
        ``partial`` marks unfinished training with its step and run digest, so that
        ``Decision2Trainer.finetune`` can resume the same run from this checkpoint.
        """
        if model_id not in DECISION2:
            raise ValueError("Unsupported Decision 2.0 checkpoint model")
        if not _pinned(revision):
            raise ValueError("A pinned checkpoint revision is required")
        source = Path(source).expanduser()
        destination = Path(destination).expanduser().resolve()
        license_file = Path(license_file).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(destination)
        if source.is_symlink() or not source.is_dir() or not license_file.is_file():
            raise FileNotFoundError(
                "Local full checkpoint and Apache-2.0 license required"
            )
        if any(
            not (source / name).is_file() or (source / name).is_symlink()
            for name in _REQUIRED
        ):
            raise FileNotFoundError(
                "Complete Decision 2.0 weights, settings and tokenizer required"
            )
        _check_settings(source, model_id)
        if parent_revision is not None and not _pinned(parent_revision):
            raise ValueError("Invalid parent checkpoint revision")
        if partial is not None and not _valid_partial(partial):
            raise ValueError("Invalid partial training progress")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".decision2-checkpoint-", dir=destination.parent
        ) as temporary:
            staging = Path(temporary) / "checkpoint"
            staging.mkdir()
            for name in sorted(_REQUIRED):
                copied = staging / name
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, copied)
            shutil.copyfile(license_file, staging / "LICENSE.txt")
            metadata = {
                "schema_version": 1,
                "kind": "training",
                "format": "safetensors",
                "family": "decision2",
                "architecture": DECISION2[model_id],
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


from ._decision2_trainer import Decision2Trainer  # noqa: E402

__all__ = ["Decision2Trainer", "Decision2TrainingBundle", "TrainingResult"]
