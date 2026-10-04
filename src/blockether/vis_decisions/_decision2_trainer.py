"""Offline Decision 2.0 labels, held-out evaluation and full FP32 fine-tuning.

Only constructing the trainer imports tensor dependencies. JSONL rows and their
contents never enter provenance, reports or upload archives. A partial checkpoint
records only a digest of its rows and settings, so that the same run can resume.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._trainer import (
    TrainingResult,
    _config,
    _export_fp32,
    _fingerprint,
    _publish_checkpoint,
    _quality_report,
    _Schedule,
    _training_config,
)
from .training import _sha256

if TYPE_CHECKING:
    from .decision2_training import Decision2TrainingBundle


@dataclass(frozen=True)
class _Example:
    prompt: dict[str, Any]
    target: int


def _options(question: dict) -> list[dict[str, Any]]:
    """Map a gateway question to Decision 2.0 options, like the Vis runtime."""
    kind, criteria = question.get("type"), question.get("criteria")
    if kind == "choice":
        if isinstance(criteria, dict):
            options = list(criteria.items())
        elif isinstance(criteria, list):
            options = [(label, None) for label in criteria]
        else:
            raise ValueError("Choice criteria must contain labeled options")
    elif kind == "score":
        if not isinstance(criteria, list):
            raise ValueError("Score criteria must contain levels")
        options = [(str(level), text) for level, text in enumerate(criteria)]
    elif kind == "noul":
        criteria = {} if criteria is None else criteria
        if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
            raise ValueError("Noul criteria accept only false and true")
        options = (
            list(criteria.items())
            if len(criteria) == 2
            else [
                ("false", criteria.get("false", "No")),
                ("true", criteria.get("true", "Yes")),
            ]
        )
    else:
        raise ValueError("Decision question needs a valid type and instructions")
    limit = 10 if kind == "score" else 64
    if not 2 <= len(options) <= limit or any(
        not isinstance(key, str) or not key.strip() for key, _ in options
    ):
        raise ValueError(f"Decision 2.0 needs 2-{limit} options with text keys")
    return [{"key": key, "description": text} for key, text in options]


def _example(row: dict) -> _Example:
    if not isinstance(row, dict) or not {"state", "question", "target"} <= row.keys():
        raise ValueError("A decision label is required")
    question = row["question"]
    if not isinstance(question, dict):
        raise ValueError("Decision question must be an object")
    if question.get("instructions") is None:
        raise ValueError("Decision question needs a valid type and instructions")
    if not isinstance(row["state"], (str, dict, list)):
        raise ValueError("Decision state must be text, an object or a list")
    options = _options(question)
    target, action = row["target"], row.get("action", 0)
    if type(action) is not int or action not in (0, 1):
        raise ValueError("An action label must be 0 or 1")
    if type(target) is not int or not 0 <= target < len(options):
        raise ValueError("Decision target exceeds the options")
    if question["type"] == "noul":
        # The shared label format uses 0 for false and 1 for true, like GLiNER.
        if target not in (0, 1):
            raise ValueError("Noul target must be 0 (false) or 1 (true)")
        keys = [option["key"] for option in options]
        target = keys.index(("false", "true")[target])
    prompt = {
        "state": row["state"],
        "task_type": question["type"],
        "instructions": question["instructions"],
        "options": options,
    }
    return _Example(prompt, target)


def _examples(source: str | Path) -> list[_Example]:
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Labeled examples are missing: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(_example(json.loads(line)))
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"Invalid labeled example at line {number}: {error}"
                ) from None
    if not rows:
        raise ValueError("Labeled examples cannot be empty")
    return rows


def _identity(row: _Example) -> str:
    return json.dumps(row.prompt, ensure_ascii=False, sort_keys=True)


def _weights_revision(checkpoint: Path) -> str:
    """Identify trained weights by the digests of the backbone and the head."""
    digest = hashlib.sha256()
    for name in ("backbone/model.safetensors", "decision_head.safetensors"):
        digest.update(f"{_sha256(checkpoint / name)}\n".encode())
    return digest.hexdigest()


class Decision2Trainer:
    """Explicit local full fine-tuning. A validated ONNX bundle is never a checkpoint."""

    def __init__(self, checkpoint: Decision2TrainingBundle) -> None:
        from .decision2_training import Decision2TrainingBundle

        if not isinstance(checkpoint, Decision2TrainingBundle):
            raise TypeError("Decision2TrainingBundle.open/fetch/from_local is required")
        self.checkpoint = Decision2TrainingBundle.open(checkpoint.path)

        from . import _decision2 as exporter

        self._exporter = exporter
        self._torch = exporter.torch
        self._torch.set_num_threads(min(self._torch.get_num_threads(), 4))
        self._tokenizer = exporter.Tokenizer.from_file(
            str(self.checkpoint.path / "tokenizer.json")
        )
        self._closed = False
        self.model = None

    def __enter__(self) -> Decision2Trainer:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release the model and its native tensor storage."""
        self.model = None
        gc.collect()
        self._closed = True

    def _encode(self, rows: list[_Example]) -> list[dict[str, Any]]:
        """Check the token limit of every row before any slow step starts."""
        encoded = []
        for number, row in enumerate(rows, 1):
            try:
                encoded.append(self._exporter.encode(row.prompt, self._tokenizer))
            except ValueError as error:
                raise ValueError(f"Invalid labeled example {number}: {error}") from None
        return encoded

    def _prepare(
        self,
        checkpoint: Path,
        destination: Path,
        rows: list[_Example],
        encoded: list[dict[str, Any]],
        policy: dict,
    ) -> TrainingResult:
        if self._closed:
            raise RuntimeError("Trainer is closed")
        from .decision2_training import Decision2TrainingBundle

        Decision2TrainingBundle.open(checkpoint)
        metadata = json.loads(
            (checkpoint / "PROVENANCE.json").read_text(encoding="utf-8")
        )
        inference = destination / "inference"
        parity = self._exporter.prepare_fp32(
            checkpoint,
            inference,
            license_file=checkpoint / "LICENSE.txt",
            revision=metadata["revision"],
            model_id=metadata["model"],
        )
        model, _ = self._exporter.load_checkpoint(checkpoint)
        options = self._exporter.ort.SessionOptions()
        options.intra_op_num_threads = 4
        runtime = self._exporter.ort.InferenceSession(
            str(inference / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        np = self._exporter.np
        correct = 0
        largest_error = parity["max_abs_logit_error"]
        try:
            for row, item in zip(rows, encoded, strict=True):
                arguments = self._exporter.collate([item])
                with self._torch.inference_mode():
                    expected = model(*arguments).numpy()
                actual = runtime.run(
                    None,
                    {
                        name: value.numpy()
                        for name, value in zip(
                            self._exporter.INPUT_NAMES, arguments, strict=True
                        )
                    },
                )[0]
                if (
                    actual.shape != expected.shape
                    or not np.isfinite(actual).all()
                    or not np.allclose(expected, actual, rtol=1e-4, atol=1e-3)
                ):
                    raise ValueError(
                        "FP32 export disagrees with the held-out checkpoint"
                    )
                largest_error = max(
                    largest_error, float(np.max(np.abs(expected - actual)))
                )
                correct += int(actual[0].argmax() == row.target)
        finally:
            del runtime, model
            gc.collect()
        report = _quality_report(
            destination,
            policy,
            revision=metadata["revision"],
            examples=len(rows),
            decisions=correct,
            actions=None,
            largest_error=largest_error,
        )
        return TrainingResult(checkpoint, inference, report)

    def prepare_fp32(
        self,
        *,
        eval_data: str | Path,
        validation_policy: str | Path,
        output_dir: str | Path,
        progress: Callable[[dict], None] | None = None,
    ) -> TrainingResult:
        """Export an existing full checkpoint and gate its decisions on held-out rows."""
        if self._closed:
            raise RuntimeError("Trainer is closed")
        rows = _examples(eval_data)
        policy = _config(validation_policy, kind="quality policy", actions=False)
        encoded = self._encode(rows)
        return _export_fp32(
            output_dir,
            checkpoint=self.checkpoint.path,
            prefix=".decision2-prepare-",
            prepare=lambda prepared: self._prepare(
                self.checkpoint.path, prepared, rows, encoded, policy
            ),
            progress=progress,
        )

    def _fit(
        self,
        model: Any,
        rows: list[_Example],
        encoded: list[dict[str, Any]],
        schedule: _Schedule,
        config: dict,
        save: Callable[[Any, int], Path],
        progress: Callable[[dict], None] | None,
    ) -> Path:
        """Train the backbone and the head; the optimizer state ends with this call."""
        torch = self._torch
        torch.manual_seed(config.get("seed", 42))
        model.train()
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": config["encoder_lr"]},
                {"params": model.head.parameters(), "lr": config["task_lr"]},
            ]
        )
        every = config.get("checkpoint_steps")
        interval = max(1, schedule.total // 100)
        order = iter(schedule)
        checkpoint = None
        if progress:
            progress(
                {
                    "stage": "training",
                    "step": schedule.done,
                    "max_steps": schedule.total,
                }
            )
        for step in range(schedule.done + 1, schedule.total + 1):
            batch = [next(order) for _ in range(schedule.batch)]
            arguments = self._exporter.collate([encoded[index] for index in batch])
            counts = torch.tensor(
                [len(encoded[index]["candidate_positions"]) for index in batch]
            )
            logits = model(*arguments)
            valid = torch.arange(logits.shape[1])[None, :] < counts[:, None]
            loss = torch.nn.functional.cross_entropy(
                logits.masked_fill(~valid, float("-inf")),
                torch.tensor([rows[index].target for index in batch]),
            )
            if not math.isfinite(loss.item()):
                raise ValueError(
                    "Decision 2.0 training did not complete with finite loss"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if progress and (
                step == schedule.done + 1
                or step % interval == 0
                or step == schedule.total
            ):
                progress(
                    {
                        "stage": "training",
                        "step": step,
                        "max_steps": schedule.total,
                        "epoch": round(step / schedule.per_pass, 4),
                        "loss": loss.item(),
                    }
                )
            if step == schedule.total or (every and step % every == 0):
                checkpoint = save(model, step)
        return checkpoint

    def finetune(
        self,
        *,
        train_data: str | Path,
        eval_data: str | Path,
        training_config: str | Path,
        validation_policy: str | Path,
        output_dir: str | Path,
        progress: Callable[[dict], None] | None = None,
    ) -> TrainingResult:
        """Train the backbone and head, keep resumable checkpoints and publish FP32.

        ``progress`` receives the step, ``max_steps``, epoch and loss about once per
        percent of the run. With ``checkpoint_steps``, ``output_dir/checkpoint`` keeps
        the latest partial checkpoint when training fails or stops. Training that
        checkpoint with the same rows and settings resumes at its saved step.
        """
        if self._closed:
            raise RuntimeError("Trainer is closed")
        rows, evaluation = _examples(train_data), _examples(eval_data)
        if {_identity(row) for row in rows} & {_identity(row) for row in evaluation}:
            raise ValueError("Training and evaluation examples must be disjoint")
        config = _training_config(training_config)
        policy = _config(validation_policy, kind="quality policy", actions=False)
        target = Path(output_dir).expanduser().resolve()
        if target.exists():
            raise FileExistsError(target)
        encoded, held_out = self._encode(rows), self._encode(evaluation)
        target.parent.mkdir(parents=True, exist_ok=True)
        from .decision2_training import Decision2TrainingBundle

        Decision2TrainingBundle.open(self.checkpoint.path)
        original = json.loads(
            (self.checkpoint.path / "PROVENANCE.json").read_text(encoding="utf-8")
        )
        fingerprint = _fingerprint(
            {
                "epochs": config.get("epochs", 1),
                "max_steps": config.get("max_steps"),
                "batch_size": config.get("batch_size", 1),
                "encoder_lr": config["encoder_lr"],
                "task_lr": config["task_lr"],
                "seed": config.get("seed", 42),
            },
            ([row.prompt, row.target] for row in rows),
        )
        schedule = _Schedule(len(rows), config)
        if schedule.total > 100_000:
            raise ValueError("Decision 2.0 training must plan at most 100000 steps")
        partial = original.get("partial")
        if (
            partial
            and partial["fingerprint"] == fingerprint
            and partial["max_steps"] == schedule.total
        ):
            schedule.done = partial["step"]
        with tempfile.TemporaryDirectory(
            prefix=".decision2-train-", dir=target.parent
        ) as temporary:
            staging = Path(temporary)

            def save(model: Any, step: int) -> Path:
                saved = self._exporter.save_checkpoint(
                    model, self.checkpoint.path, staging / f"checkpoint-{step}"
                )
                pending = staging / "pending"
                Decision2TrainingBundle.from_local(
                    saved,
                    pending,
                    model_id=self.checkpoint.model_id,
                    revision=_weights_revision(saved),
                    license_file=self.checkpoint.path / "LICENSE.txt",
                    parent_revision=original["revision"],
                    partial=None
                    if step == schedule.total
                    else {
                        "step": step,
                        "max_steps": schedule.total,
                        "fingerprint": fingerprint,
                    },
                )
                shutil.rmtree(saved)
                return _publish_checkpoint(
                    pending,
                    target,
                    {"steps": step, "max_steps": schedule.total},
                    progress,
                )

            model, _ = self._exporter.load_checkpoint(self.checkpoint.path)
            self.model = model
            try:
                checkpoint = self._fit(
                    model, rows, encoded, schedule, config, save, progress
                )
            finally:
                self.model = None
                del model
                gc.collect()
            if progress:
                progress({"stage": "exporting"})
            prepared = staging / "prepared"
            prepared.mkdir()
            self._prepare(checkpoint, prepared, evaluation, held_out, policy)
            (prepared / "inference").rename(target / "inference")
            (prepared / "validation_report.json").rename(
                target / "validation_report.json"
            )
        if progress:
            progress({"stage": "validated"})
        return TrainingResult(
            target / "checkpoint",
            target / "inference",
            target / "validation_report.json",
        )


__all__ = ["Decision2Trainer"]
