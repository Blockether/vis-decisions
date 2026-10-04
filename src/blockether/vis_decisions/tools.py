"""Typed tools for local training; no host imports or implicit alias activation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ._publication import package
from .api import Trainer, TrainingBundle
from .training import _manifest


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    revision: str
    inference_bytes: int
    training_bytes: int


@dataclass(frozen=True)
class CheckpointInfo:
    model_id: str
    revision: str
    path: str
    step: int
    max_steps: int


@dataclass(frozen=True)
class TrainingOutcome:
    checkpoint_dir: str
    inference_bundle: str
    validation_report: str
    decision_accuracy: float
    action_accuracy: float | None


@dataclass(frozen=True)
class ArchiveInfo:
    path: str
    sha256: str
    bytes: int


def _checkpoint_info(path: Path) -> CheckpointInfo:
    metadata = json.loads((path / "PROVENANCE.json").read_text(encoding="utf-8"))
    partial = metadata.get("partial", {})
    return CheckpointInfo(
        metadata["model"],
        metadata["revision"],
        str(path),
        partial.get("step", 0),
        partial.get("max_steps", 0),
    )


def _outcome(result) -> TrainingOutcome:
    report = json.loads(result.validation_report.read_text(encoding="utf-8"))
    return TrainingOutcome(
        str(result.checkpoint_dir),
        str(result.inference_bundle),
        str(result.validation_report),
        report["decision_accuracy"],
        report.get("action_accuracy"),
    )


class DecisionTools:
    """Download verified checkpoints, train every family and export validated ONNX."""

    def models(self) -> list[ModelInfo]:
        """Read the public Vis release catalog without downloading model weights."""
        return [
            ModelInfo(
                row["id"],
                row["revision"],
                row["artifacts"]["inference"]["bytes"],
                row["artifacts"]["training"]["bytes"],
            )
            for row in _manifest()
        ]

    def fetch(self, model_ref: str, destination: str) -> CheckpointInfo:
        """Download and verify the pinned training checkpoint, not dependencies.

        Use <id>@<revision> from models. Existing verified checkpoints are reused.
        Large archives are joined from ordered parts and checked before extraction.
        """
        return _checkpoint_info(TrainingBundle.fetch(model_ref, destination).path)

    def inspect_checkpoint(self, path: str) -> CheckpointInfo:
        """Verify every local checkpoint file and read its resumable step; no network."""
        return _checkpoint_info(TrainingBundle.open(path).path)

    def train(
        self,
        checkpoint: str,
        *,
        train_data: str,
        eval_data: str,
        training_config: str,
        validation_policy: str,
        output_dir: str,
    ) -> TrainingOutcome:
        """Train or resume a checkpoint, then validate and export its heads.

        Rows stay local. Matching partial runs resume; new rows start from saved weights.
        Validation rows must be disjoint from training. No alias is activated.
        """
        with Trainer(TrainingBundle.open(checkpoint)) as trainer:
            return _outcome(
                trainer.train(
                    train_data=train_data,
                    eval_data=eval_data,
                    training_config=training_config,
                    validation_policy=validation_policy,
                    output_dir=output_dir,
                )
            )

    def prepare(
        self,
        checkpoint: str,
        *,
        eval_data: str,
        validation_policy: str,
        output_dir: str,
    ) -> TrainingOutcome:
        """Export and validate unchanged weights; never train or activate an alias."""
        with Trainer(TrainingBundle.open(checkpoint)) as trainer:
            return _outcome(
                trainer.prepare(
                    eval_data=eval_data,
                    validation_policy=validation_policy,
                    output_dir=output_dir,
                )
            )

    def package(self, inference_bundle: str, destination: str) -> ArchiveInfo:
        """Verify and package FP32 inference files for import; exclude checkpoint and rows."""
        archive = Path(destination).expanduser().resolve()
        digest, size = package(Path(inference_bundle).expanduser().resolve(), archive)
        return ArchiveInfo(str(archive), digest, size)
