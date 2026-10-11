"""D3 training checkpoints, LoRA and full fine-tuning, resume and FP32 export."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from test_d3 import CODES, _tiny_model

from blockether.vis_decisions import Trainer, TrainingBundle, _d3
from blockether.vis_decisions.d3_training import D3Trainer, D3TrainingBundle

REVISION = "a" * 40
QUESTION = {
    "type": "choice",
    "instructions": "Pick the support queue for this message.",
    "criteria": {"refund": "Money back requests", "repair": None, "shipping": None},
}
STATES = [
    "Customer: my parcel arrived broken, I want my money back.",
    "The release build failed twice with a timeout.",
    "Customer: where is my parcel?",
    "Customer: the screen is broken, please fix it.",
]


def source_checkpoint(root: Path, model_id: str = "d3-edge") -> Path:
    """Write a tiny checkpoint in the upstream layout, with a vision weight."""
    model, tokenizer = _tiny_model()
    root.mkdir(parents=True)
    weights = {
        f"language_model.{name}": value.to(torch.bfloat16)
        for name, value in model.backbone.state_dict().items()
    }
    weights["visual.patch_embed.weight"] = torch.zeros(2, 2)
    save_file(weights, str(root / "model-00001-of-00001.safetensors"))
    save_file({"weight": model.readout.detach()}, str(root / "readout.safetensors"))
    text = model.backbone.config.to_dict()
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": text, "vision_config": {}})
    )
    (root / "decision_config.json").write_text(
        json.dumps(
            {
                "format_id": "d3-code-readout-v1",
                "prompt": "d3",
                "codes": CODES,
                "token_ids": [tokenizer.token_to_id(code) for code in CODES],
                "attention_mode": "noncausal_full_attention",
            }
        )
    )
    (root / "MODEL_MANIFEST.json").write_text(
        json.dumps({"model_name": model_id, "repo_id": f"vllm-sr/{model_id}"})
    )
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text("{}")
    (root / "LICENSE").write_text("Apache-2.0")
    (root / "README.md").write_text("upstream model card")
    return root


def base_checkpoint(root: Path) -> D3TrainingBundle:
    source = source_checkpoint(root / "source")
    return D3TrainingBundle.from_local(
        source,
        root / "checkpoint",
        model_id="d3-edge",
        revision=REVISION,
        license_file=source / "LICENSE",
    )


def provenance(root: Path) -> dict:
    return json.loads((root / "PROVENANCE.json").read_text())


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def training_inputs(root: Path, targets=(0, 1, 2, 0), **settings) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    config = {"max_steps": 2, "batch_size": 2, "encoder_lr": 1e-4, "task_lr": 1e-3}
    config.update({"device": "cpu", **settings})
    (root / "config.json").write_text(json.dumps(config))
    (root / "policy.json").write_text(json.dumps({"min_decision_accuracy": 0}))
    rows = [
        {"state": state, "question": QUESTION, "target": target}
        for state, target in zip(STATES, targets, strict=True)
    ]
    held_out = [
        {
            "state": "Customer: refund the duplicate charge.",
            "question": QUESTION,
            "target": 0,
        },
        {
            "state": {"user": "alice", "verified": True},
            "question": {"type": "noul", "instructions": "Is the user verified?"},
            "target": 1,
        },
    ]
    return {
        "train_data": write_rows(root / "train.jsonl", rows),
        "eval_data": write_rows(root / "eval.jsonl", held_out),
        "training_config": root / "config.json",
        "validation_policy": root / "policy.json",
    }


def test_checkpoint_keeps_only_text_weights_and_verifies_files(tmp_path):
    bundle = base_checkpoint(tmp_path)
    metadata = provenance(bundle.path)
    assert metadata["family"] == "d3" and metadata["kind"] == "training"
    assert metadata["model"] == "d3-edge" and metadata["revision"] == REVISION
    names = set(load_file(bundle.path / "model.safetensors"))
    assert names and all(name.startswith("language_model.") for name in names)
    assert not (bundle.path / "README.md").exists()
    assert isinstance(TrainingBundle.open(bundle.path), D3TrainingBundle)

    (bundle.path / "readout.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        D3TrainingBundle.open(bundle.path)


def test_checkpoint_rejects_another_model_or_unpinned_revision(tmp_path):
    source = source_checkpoint(tmp_path / "source")
    settings = {"license_file": source / "LICENSE"}
    with pytest.raises(ValueError, match="not a d3-lite model"):
        D3TrainingBundle.from_local(
            source, tmp_path / "lite", model_id="d3-lite", revision=REVISION, **settings
        )
    with pytest.raises(ValueError, match="pinned"):
        D3TrainingBundle.from_local(
            source, tmp_path / "main", model_id="d3-edge", revision="main", **settings
        )
    assert not (tmp_path / "lite").exists() and not (tmp_path / "main").exists()


def test_labels_must_name_an_option(tmp_path):
    bundle = base_checkpoint(tmp_path)
    inputs = training_inputs(tmp_path / "inputs")
    write_rows(
        inputs["train_data"], [{"state": "x", "question": QUESTION, "target": 3}]
    )
    with D3Trainer(bundle) as trainer, pytest.raises(ValueError, match="line 1"):
        trainer.finetune(**inputs, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_lora_training_merges_weights_and_exports_fp32(tmp_path):
    bundle = base_checkpoint(tmp_path)
    inputs = training_inputs(tmp_path / "inputs", lora={"rank": 2})
    events = []
    with Trainer(bundle) as trainer:
        result = trainer.train(
            **inputs, output_dir=tmp_path / "trained", progress=events.append
        )
    assert events[-1] == {"stage": "validated"}
    weights = load_file(result.checkpoint_dir / "model.safetensors")
    assert not any("lora_" in name for name in weights)
    before = load_file(bundle.path / "model.safetensors")
    assert any(not torch.equal(before[name].float(), weights[name]) for name in before)
    readout = load_file(result.checkpoint_dir / "readout.safetensors")["weight"]
    original = load_file(bundle.path / "readout.safetensors")["weight"]
    assert not torch.equal(readout, original)
    inference = provenance(result.inference_bundle)
    checkpoint = provenance(result.checkpoint_dir)
    assert inference["family"] == "d3" and inference["kind"] == "inference"
    assert inference["revision"] == checkpoint["revision"]
    assert checkpoint["parent_revision"] == REVISION
    report = json.loads(result.validation_report.read_text())
    assert report["examples"] == 2 and report["action_accuracy"] is None


def test_full_training_raises_the_target_code_and_resumes(tmp_path):
    bundle = base_checkpoint(tmp_path)
    inputs = training_inputs(
        # One target for every row, so that the training loss must fall.
        tmp_path / "inputs",
        targets=(0, 0, 0, 0),
        checkpoint_steps=1,
        max_steps=3,
        task_lr=0.01,
    )
    events = []

    def stop_after_first_checkpoint(event):
        events.append(event)
        if event["stage"] == "checkpoint_saved" and event["step"] == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt), D3Trainer(bundle) as trainer:
        trainer.finetune(
            **inputs,
            output_dir=tmp_path / "stopped",
            progress=stop_after_first_checkpoint,
        )
    partial = TrainingBundle.open(tmp_path / "stopped" / "checkpoint")
    assert provenance(partial.path)["partial"]["step"] == 1

    events.clear()
    with Trainer(partial) as trainer:
        result = trainer.train(
            **inputs, output_dir=tmp_path / "resumed", progress=events.append
        )
    training = [event for event in events if event["stage"] == "training"]
    assert [event["step"] for event in training] == [1, 2, 3]
    assert "partial" not in provenance(result.checkpoint_dir)

    def loss(source: Path) -> float:
        model, tokenizer = _d3.load_checkpoint(source)
        rows = [
            json.loads(line) for line in inputs["train_data"].read_text().splitlines()
        ]
        total = 0.0
        with torch.inference_mode():
            for row in rows:
                logits = model(torch.tensor([_d3.encode(row, tokenizer, model.codes)]))
                total += float(
                    torch.nn.functional.cross_entropy(
                        logits[:, :3], torch.tensor([row["target"]])
                    )
                )
        return total / len(rows)

    assert loss(result.checkpoint_dir) < loss(bundle.path)


def test_max_input_tokens_cannot_exceed_the_d3_limit(tmp_path):
    bundle = base_checkpoint(tmp_path)
    inputs = training_inputs(
        tmp_path / "inputs", max_input_tokens=_d3.MAX_INPUT_TOKENS + 1
    )
    with D3Trainer(bundle) as trainer, pytest.raises(ValueError, match="token limit"):
        trainer.finetune(**inputs, output_dir=tmp_path / "out")
