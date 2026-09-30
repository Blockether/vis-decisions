"""One training interface for complete, verified Laya and GLiNER checkpoints."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ._models import ARCHITECTURES
from ._trainer import ModernBertTrainer, TrainingResult
from .gliner_training import GlinerTrainer, GlinerTrainingBundle
from .training import TrainingBundle as LayaTrainingBundle

Bundle = LayaTrainingBundle | GlinerTrainingBundle


class TrainingBundle:
    """Open or download a full checkpoint without importing the training runtime."""

    @staticmethod
    def open(path: str | Path) -> Bundle:
        """Verify the file inventory, then select the model family from provenance."""
        root = Path(path).expanduser().resolve()
        metadata = json.loads((root / "PROVENANCE.json").read_text(encoding="utf-8"))
        model_id = metadata.get("model")
        if model_id == "laya-typed-decisions":
            return LayaTrainingBundle.open(root)
        if model_id in ARCHITECTURES:
            return GlinerTrainingBundle.open(root)
        raise ValueError("Unsupported decision training model identity")

    @staticmethod
    def fetch(model_ref: str, destination: str | Path) -> Bundle:
        """Download a pinned <id>@<revision> checkpoint; never download dependencies."""
        model_id = model_ref.split("@", 1)[0]
        if model_id == "laya-typed-decisions":
            return LayaTrainingBundle.fetch(model_ref=model_ref, cache_dir=destination)
        if model_id in ARCHITECTURES:
            return GlinerTrainingBundle.fetch(
                model_ref=model_ref, cache_dir=destination
            )
        raise ValueError("Unsupported decision training model identity")


class Trainer:
    """Train, continue or resume either family with disjoint held-out validation."""

    def __init__(self, checkpoint: Bundle) -> None:
        if not isinstance(checkpoint, LayaTrainingBundle):
            raise TypeError("TrainingBundle.open or TrainingBundle.fetch is required")
        self.checkpoint = checkpoint
        self._trainer = (
            GlinerTrainer(checkpoint)
            if isinstance(checkpoint, GlinerTrainingBundle)
            else ModernBertTrainer(checkpoint)
        )

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release weights and native tensor storage."""
        self._trainer.close()

    def train(
        self,
        *,
        train_data: str | Path,
        eval_data: str | Path,
        training_config: str | Path,
        validation_policy: str | Path,
        output_dir: str | Path,
        progress: Callable[[dict], None] | None = None,
    ) -> TrainingResult:
        """Fine-tune both heads and export only after independent validation.

        A matching partial checkpoint resumes its exact step and batch order.
        New data or settings start a new run from the checkpoint's saved weights.
        Inputs remain local. No gateway alias changes.
        """
        return self._trainer.finetune(
            train_data=train_data,
            eval_data=eval_data,
            training_config=training_config,
            validation_policy=validation_policy,
            output_dir=output_dir,
            progress=progress,
        )

    def prepare(
        self,
        *,
        output_dir: str | Path,
        eval_data: str | Path,
        validation_policy: str | Path,
        progress: Callable[[dict], None] | None = None,
    ) -> TrainingResult:
        """Export an unchanged checkpoint and validate both heads without training."""
        return self._trainer.prepare_fp32(
            output_dir=output_dir,
            eval_data=eval_data,
            validation_policy=validation_policy,
            progress=progress,
        )
