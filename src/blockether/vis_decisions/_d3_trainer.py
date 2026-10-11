"""Offline D3 labels, held-out evaluation and FP32 fine-tuning.

Training raises the readout logit of the correct option's answer code. Only
constructing the trainer imports tensor dependencies. JSONL rows and their contents
never enter provenance, reports or upload archives. A partial checkpoint records
only a digest of its rows and settings, so that the same run can resume.
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
    _device,
    _device_errors,
    _export_fp32,
    _fingerprint,
    _publish_checkpoint,
    _quality_report,
    _Schedule,
    _threads,
    _training_config,
    _with_lora,
)
from .training import _sha256

if TYPE_CHECKING:
    from .d3_training import D3TrainingBundle


@dataclass(frozen=True)
class _Example:
    prompt: dict[str, Any]
    target: int


def _example(row: dict, options: Callable[[dict], tuple[list, list]]) -> _Example:
    if not isinstance(row, dict) or not {"state", "question", "target"} <= row.keys():
        raise ValueError("A decision label is required")
    question = row["question"]
    if not isinstance(question, dict):
        raise ValueError("Decision question must be an object")
    if question.get("instructions") is None:
        raise ValueError("Decision question needs a valid type and instructions")
    if not isinstance(row["state"], (str, dict, list)):
        raise ValueError("Decision state must be text, an object or a list")
    keys, _ = options(question)
    target, action = row["target"], row.get("action", 0)
    if type(action) is not int or action not in (0, 1):
        raise ValueError("An action label must be 0 or 1")
    if type(target) is not int or not 0 <= target < len(keys):
        raise ValueError("Decision target exceeds the options")
    # The noul codes list false, then true, so the shared 0/1 labels apply unchanged.
    return _Example({"state": row["state"], "question": question}, target)


def _examples(
    source: str | Path, options: Callable[[dict], tuple[list, list]]
) -> list[_Example]:
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Labeled examples are missing: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(_example(json.loads(line), options))
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
    """Identify trained weights by the digests of the text weights and the readout."""
    digest = hashlib.sha256()
    for name in ("model.safetensors", "readout.safetensors"):
        digest.update(f"{_sha256(checkpoint / name)}\n".encode())
    return digest.hexdigest()


class D3Trainer:
    """Explicit local fine-tuning. A validated ONNX bundle is never a checkpoint."""

    def __init__(self, checkpoint: D3TrainingBundle) -> None:
        from .d3_training import D3TrainingBundle

        if not isinstance(checkpoint, D3TrainingBundle):
            raise TypeError("D3TrainingBundle.open/fetch/from_local is required")
        self.checkpoint = D3TrainingBundle.open(checkpoint.path)

        from . import _d3 as exporter

        self._exporter = exporter
        self._torch = exporter.torch
        self._torch.set_num_threads(_threads(self._torch))
        self._tokenizer = exporter.Tokenizer.from_file(
            str(self.checkpoint.path / "tokenizer.json")
        )
        settings = json.loads(
            (self.checkpoint.path / "decision_config.json").read_text(encoding="utf-8")
        )
        self._codes = list(settings["codes"])
        self._closed = False
        self.model = None

    def __enter__(self) -> D3Trainer:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release the model and its native tensor storage."""
        self.model = None
        gc.collect()
        self._closed = True

    def _examples(self, source: str | Path) -> list[_Example]:
        return _examples(source, self._exporter.options)

    def _encode(
        self, rows: list[_Example], max_length: int | None = None
    ) -> list[list[int]]:
        """Check the token limit of every row before any slow step starts.

        With ``max_length``, the end of a long state is cut until the row fits.
        """
        limit = max_length or self._exporter.MAX_INPUT_TOKENS
        encoded = []
        for number, row in enumerate(rows, 1):
            try:
                prompt = row.prompt
                if max_length is not None:
                    prompt = self._exporter.fit(
                        prompt, self._tokenizer, self._codes, max_length
                    )
                ids = self._exporter.encode(prompt, self._tokenizer, self._codes)
                if len(ids) > limit:
                    raise ValueError(
                        f"{len(ids)} tokens exceed the {limit} token limit"
                    )
                encoded.append(ids)
            except ValueError as error:
                raise ValueError(f"Invalid labeled example {number}: {error}") from None
        return encoded

    def _count(self, row: _Example) -> int:
        return len(self._exporter.options(row.prompt["question"])[0])

    def _prepare(
        self,
        checkpoint: Path,
        destination: Path,
        rows: list[_Example],
        encoded: list[list[int]],
        policy: dict,
    ) -> TrainingResult:
        if self._closed:
            raise RuntimeError("Trainer is closed")
        from .d3_training import D3TrainingBundle

        D3TrainingBundle.open(checkpoint)
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
        options.intra_op_num_threads = self._torch.get_num_threads()
        runtime = self._exporter.ort.InferenceSession(
            str(inference / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        np = self._exporter.np
        correct = 0
        largest_error = parity["max_abs_logit_error"]
        try:
            for row, ids in zip(rows, encoded, strict=True):
                count = self._count(row)
                input_ids = self._torch.tensor([ids])
                with self._torch.inference_mode():
                    expected = model(input_ids).numpy()[0, :count]
                actual = runtime.run(None, {"input_ids": input_ids.numpy()})[0][
                    0, :count
                ]
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
                correct += int(actual.argmax() == row.target)
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
        """Export an existing checkpoint and gate its decisions on held-out rows."""
        if self._closed:
            raise RuntimeError("Trainer is closed")
        rows = self._examples(eval_data)
        policy = _config(validation_policy, kind="quality policy", actions=False)
        encoded = self._encode(rows)
        return _export_fp32(
            output_dir,
            checkpoint=self.checkpoint.path,
            prefix=".d3-prepare-",
            prepare=lambda prepared: self._prepare(
                self.checkpoint.path, prepared, rows, encoded, policy
            ),
            progress=progress,
        )

    def _fit(
        self,
        model: Any,
        rows: list[_Example],
        encoded: list[list[int]],
        schedule: _Schedule,
        config: dict,
        save: Callable[[Any, int], Path],
        progress: Callable[[dict], None] | None,
    ) -> Path:
        """Train the backbone and the readout; the optimizer state ends with this call.

        D3 prompts have no padding, so each row runs alone and a step adds the
        gradients of its batch. With ``lora``, only adapters train in the backbone;
        the readout trains fully.
        """
        torch = self._torch
        device = _device(torch, config.get("device", "auto"))
        torch.manual_seed(config.get("seed", 42))
        if "lora" in config:
            model.backbone = _with_lora(model.backbone, config["lora"])
        model.readout.requires_grad_(True)
        model.to(device)
        model.train()
        backbone = [item for item in model.backbone.parameters() if item.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone, "lr": config["encoder_lr"]},
                {"params": [model.readout], "lr": config["task_lr"]},
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
                    "device": device.type,
                }
            )
        for step in range(schedule.done + 1, schedule.total + 1):
            batch = [next(order) for _ in range(schedule.batch)]
            with _device_errors(device):
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for index in batch:
                    count = self._count(rows[index])
                    logits = model(torch.tensor([encoded[index]], device=device))
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :count],
                        torch.tensor([rows[index].target], device=device),
                    ) / len(batch)
                    loss.backward()
                    total += loss.item()
                if not math.isfinite(total):
                    raise ValueError("D3 training did not complete with finite loss")
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
                        "loss": total,
                        "device": device.type,
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
        """Train the backbone and readout, keep resumable checkpoints and publish FP32.

        Each row's ``target`` selects the option whose answer code must score highest.
        The configuration ``device`` (``auto``, ``cpu``, ``mps`` or ``cuda``) selects
        the FP32 training device; checkpoints, export and validation use the CPU.
        ``lora`` trains low-rank adapters that each checkpoint merges into full
        weights. ``max_input_tokens`` cuts long training states to fit.
        ``progress`` receives the step, ``max_steps``, epoch, loss and device about once per
        percent of the run. With ``checkpoint_steps``, ``output_dir/checkpoint`` keeps
        the latest partial checkpoint when training fails or stops. Training that
        checkpoint with the same rows and settings resumes at its saved step.
        """
        if self._closed:
            raise RuntimeError("Trainer is closed")
        rows, evaluation = self._examples(train_data), self._examples(eval_data)
        if {_identity(row) for row in rows} & {_identity(row) for row in evaluation}:
            raise ValueError("Training and evaluation examples must be disjoint")
        config = _training_config(training_config, tokens=True)
        limit, context = config.get("max_input_tokens"), self._exporter.MAX_INPUT_TOKENS
        if limit is not None and limit > context:
            raise ValueError(
                f"Training max_input_tokens exceeds the {context} token limit "
                f"of {self.checkpoint.model_id}"
            )
        device = _device(self._torch, config.get("device", "auto"))
        policy = _config(validation_policy, kind="quality policy", actions=False)
        target = Path(output_dir).expanduser().resolve()
        if target.exists():
            raise FileExistsError(target)
        encoded, held_out = self._encode(rows, limit), self._encode(evaluation)
        target.parent.mkdir(parents=True, exist_ok=True)
        from .d3_training import D3TrainingBundle

        D3TrainingBundle.open(self.checkpoint.path)
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
                **{
                    name: config[name]
                    for name in ("lora", "max_input_tokens")
                    if name in config
                },
            },
            ([row.prompt, row.target] for row in rows),
        )
        schedule = _Schedule(len(rows), config)
        if schedule.total > 100_000:
            raise ValueError("D3 training must plan at most 100000 steps")
        partial = original.get("partial")
        if (
            partial
            and partial["fingerprint"] == fingerprint
            and partial["max_steps"] == schedule.total
        ):
            schedule.done = partial["step"]
        with tempfile.TemporaryDirectory(
            prefix=".d3-train-", dir=target.parent
        ) as temporary:
            staging = Path(temporary)

            def save(model: Any, step: int) -> Path:
                saved = self._exporter.save_checkpoint(
                    model, self.checkpoint.path, staging / f"checkpoint-{step}"
                )
                pending = staging / "pending"
                D3TrainingBundle.from_local(
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
                    {
                        "steps": step,
                        "max_steps": schedule.total,
                        "device": device.type,
                    },
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


__all__ = ["D3Trainer"]
