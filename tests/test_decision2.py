"""Offline Decision 2.0 checkpoints, FP32 export, training and resume.

Tiny random Qwen3.5 and Qwen3 checkpoints with the upstream layout replace the real
weights. No test downloads checkpoints or accesses Hugging Face.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel, Qwen3Config, Qwen3Model

from blockether.vis_decisions import Trainer, TrainingBundle
from blockether.vis_decisions._decision2 import (
    PROBES,
    PROMPT_VERSION,
    CandidateHead,
    make_batch,
    prepare_fp32,
)
from blockether.vis_decisions._decision2_trainer import _examples
from blockether.vis_decisions._models import DECISION2
from blockether.vis_decisions._publication import package
from blockether.vis_decisions._trainer import _quality_report
from blockether.vis_decisions.decision2_training import (
    Decision2Trainer,
    Decision2TrainingBundle,
)

MODEL_ID = "decision2.0-eos-0.8b"
KAI_ID = "decision2.0-kai-0.6b"
SCORE_OFFSETS = {"5": [0.039188, 0.203049, 0.079362, -0.15162, -0.169979]}
REVISION = "b" * 40
STATES = [
    "Customer: the parcel arrived broken, I want my money back.",
    "Customer: where is my parcel? It is three days late.",
    "Customer: please change the delivery address.",
    "Customer: the charger stopped working after one day.",
]


def source_checkpoint(root: Path, *, seed: int = 0, model_id: str = MODEL_ID) -> Path:
    """Write a tiny upstream-layout checkpoint, with upstream files Vis must omit."""
    torch.manual_seed(seed)
    corpus = [json.dumps(probe) for probe in PROBES] + STATES
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=320,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    if model_id == KAI_ID:
        config = Qwen3Config(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )
        backbone = Qwen3Model(config)
    else:
        config = Qwen3_5TextConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": True,
                "mrope_section": [2, 1, 1],
            },
        )
        backbone = Qwen3_5TextModel(config)
    root.mkdir(parents=True)
    backbone.save_pretrained(root / "backbone")
    save_file(
        CandidateHead(32, head_dim=16).state_dict(), root / "decision_head.safetensors"
    )
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 4096}))
    (root / "decision_config.json").write_text(
        json.dumps(
            {
                "architecture": DECISION2[model_id],
                "prompt_version": PROMPT_VERSION,
                "head_dim": 16,
                "head_variant": "shared",
                "checkpoint_format": "full",
            }
        )
    )
    (root / "README.md").write_text("upstream model card")
    (root / "modeling_decision2.py").write_text("raise RuntimeError('remote code')\n")
    return root


def base_checkpoint(root: Path, model_id: str = MODEL_ID) -> Decision2TrainingBundle:
    license_file = root / "LICENSE"
    license_file.parent.mkdir(parents=True, exist_ok=True)
    license_file.write_text("Apache-2.0")
    return Decision2TrainingBundle.from_local(
        source_checkpoint(root / "source", model_id=model_id),
        root / "checkpoint",
        model_id=model_id,
        revision=REVISION,
        license_file=license_file,
    )


def score_offsets(root: Path) -> None:
    """Bind upstream-format Score offsets to the current checkpoint files."""
    names = [
        path.relative_to(root).as_posix()
        for path in sorted((root / "backbone").iterdir())
    ]
    names += [
        "decision_config.json",
        "decision_head.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }
    identity = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    bias = json.dumps(
        {
            "format": "dev2-score-bias-v1",
            "model_sha256": identity,
            "offsets": SCORE_OFFSETS,
            "fit": {},
        }
    )
    (root / "score_bias.json").write_text(bias)
    manifest = {
        "identity": {"fingerprint_files": files, "model_sha256": identity},
        "score_bias": {
            "file": "score_bias.json",
            "offsets": SCORE_OFFSETS,
            "sha256": hashlib.sha256(bias.encode()).hexdigest(),
        },
    }
    (root / "MODEL_MANIFEST.json").write_text(json.dumps(manifest))


def row(state: str, target: int, **question) -> dict:
    question = {
        "type": "choice",
        "instructions": "Pick the support queue.",
        "criteria": {"refund": "Refunds", "shipping": "Delivery", "other": None},
        **question,
    }
    return {"state": state, "question": question, "target": target, "action": 0}


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    return path


def provenance(path: Path) -> dict:
    return json.loads((path / "PROVENANCE.json").read_text(encoding="utf-8"))


def test_labels_map_to_the_vis_runtime_options(tmp_path):
    rows = [
        row("a", 2),
        row("b", 0, criteria=["refund", "other"]),
        row("c", 2, type="score", criteria=["missing", "partial", "complete"]),
        row("d", 1, type="noul", criteria=None),
        row("e", 1, type="noul", criteria={"true": "Holds", "false": "Fails"}),
        {"state": {"queue": "x"}, "question": row("f", 0)["question"], "target": 1},
    ]
    examples = _examples(write_rows(tmp_path / "rows.jsonl", rows))
    options = [example.prompt["options"] for example in examples]
    assert options[0] == [
        {"key": "refund", "description": "Refunds"},
        {"key": "shipping", "description": "Delivery"},
        {"key": "other", "description": None},
    ]
    assert options[1] == [
        {"key": "refund", "description": None},
        {"key": "other", "description": None},
    ]
    assert [option["key"] for option in options[2]] == ["0", "1", "2"]
    assert options[3] == [
        {"key": "false", "description": "No"},
        {"key": "true", "description": "Yes"},
    ]
    # Target 1 means true. The explicit criteria order puts true first.
    assert [option["key"] for option in options[4]] == ["true", "false"]
    assert [example.target for example in examples] == [2, 0, 2, 1, 0, 1]
    assert examples[5].prompt["state"] == {"queue": "x"}
    assert examples[5].prompt["task_type"] == "choice"

    for invalid, message in [
        (row("a", 0, criteria=["only"]), "2-64 options"),
        (row("a", 0, type="score", criteria=list("abcdefghijk")), "2-10 options"),
        (row("a", 0, type="noul", criteria={"maybe": "?"}), "false and true"),
        (row("a", 3), "exceeds the options"),
        (row("a", True), "exceeds the options"),
        ({**row("a", 0), "action": 2}, "action label"),
        (row("a", 0, type="rank"), "valid type"),
        ({"state": "a", "question": {"type": "choice"}}, "label is required"),
    ]:
        with pytest.raises(ValueError, match=message):
            _examples(write_rows(tmp_path / "invalid.jsonl", [row("ok", 0), invalid]))


def test_local_checkpoint_is_inventoried_offline_and_fails_closed(tmp_path):
    bundle = base_checkpoint(tmp_path)
    metadata = provenance(bundle.path)
    assert metadata["family"] == "decision2"
    assert metadata["architecture"] == DECISION2[MODEL_ID]
    assert metadata["model"] == bundle.model_id == MODEL_ID
    assert set(metadata["files"]) == {
        "backbone/config.json",
        "backbone/model.safetensors",
        "decision_config.json",
        "decision_head.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert not (bundle.path / "modeling_decision2.py").exists()
    assert isinstance(TrainingBundle.open(bundle.path), Decision2TrainingBundle)

    with pytest.raises(FileExistsError):
        base_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="pinned"):
        Decision2TrainingBundle.from_local(
            tmp_path / "source",
            tmp_path / "other",
            model_id=MODEL_ID,
            revision="main",
            license_file=tmp_path / "LICENSE",
        )
    with pytest.raises(ValueError, match="Unsupported"):
        Decision2TrainingBundle.from_local(
            tmp_path / "source",
            tmp_path / "other",
            model_id="gliner2.5-base",
            revision=REVISION,
            license_file=tmp_path / "LICENSE",
        )
    (bundle.path / "notes.txt").write_text("untracked")
    with pytest.raises(ValueError, match="untracked"):
        Decision2TrainingBundle.open(bundle.path)
    (bundle.path / "notes.txt").unlink()
    (bundle.path / "decision_head.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        TrainingBundle.open(bundle.path)

    source = tmp_path / "source"
    settings = json.loads((source / "decision_config.json").read_text())
    (source / "decision_config.json").write_text(
        json.dumps({**settings, "head_variant": "per-type"})
    )
    with pytest.raises(ValueError, match="incompatible architecture"):
        Decision2TrainingBundle.from_local(
            source,
            tmp_path / "other",
            model_id=MODEL_ID,
            revision=REVISION,
            license_file=tmp_path / "LICENSE",
        )
    with pytest.raises(TypeError, match="Decision2TrainingBundle"):
        Decision2Trainer(TrainingBundle)


@pytest.mark.parametrize("model_id", [MODEL_ID, KAI_ID])
def test_tiny_checkpoint_exports_validated_fp32_and_packages(tmp_path, model_id):
    bundle = base_checkpoint(tmp_path, model_id)
    report = prepare_fp32(
        bundle.path,
        tmp_path / "inference",
        license_file=bundle.path / "LICENSE.txt",
        revision=REVISION,
    )
    assert report["examples"] == len(PROBES)
    assert report["max_abs_logit_error"] <= 1e-3
    metadata = provenance(tmp_path / "inference")
    assert metadata["model"] == model_id
    assert metadata["precision"] == "fp32"
    assert metadata["family"] == "decision2"
    assert metadata["prompt_version"] == PROMPT_VERSION
    assert "score_bias" not in metadata
    assert set(metadata["files"]) == {
        "decision_config.json",
        "model.onnx",
        "model.onnx.data",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    }
    digest, size = package(tmp_path / "inference", tmp_path / "inference.zip")
    assert len(digest) == 64 and size > 0

    settings = tmp_path / "inference" / "decision_config.json"
    settings.write_text(settings.read_text().replace(DECISION2[model_id], "other"))
    with pytest.raises(ValueError, match="wrong architecture|checksum"):
        package(tmp_path / "inference", tmp_path / "tampered.zip")


def test_kai_export_keeps_score_offsets_only_for_unchanged_weights(tmp_path):
    source = source_checkpoint(tmp_path / "source", model_id=KAI_ID)
    score_offsets(source)
    license_file = tmp_path / "LICENSE"
    license_file.write_text("Apache-2.0")
    settings = {"license_file": license_file, "revision": REVISION}
    with pytest.raises(ValueError, match=f"not a {MODEL_ID} model"):
        prepare_fp32(source, tmp_path / "other", model_id=MODEL_ID, **settings)
    report = prepare_fp32(source, tmp_path / "inference", **settings)
    assert report["max_abs_logit_error"] <= 1e-3
    metadata = provenance(tmp_path / "inference")
    assert metadata["model"] == KAI_ID
    assert metadata["architecture"] == DECISION2[KAI_ID]
    assert metadata["score_bias"] == SCORE_OFFSETS

    # A training bundle keeps only the weights, so its exports have no offsets.
    bundle = Decision2TrainingBundle.from_local(
        source, tmp_path / "checkpoint", model_id=KAI_ID, **settings
    )
    prepare_fp32(bundle.path, tmp_path / "trained", **settings)
    assert "score_bias" not in provenance(tmp_path / "trained")

    torch.manual_seed(1)
    save_file(
        CandidateHead(32, head_dim=16).state_dict(),
        source / "decision_head.safetensors",
    )
    with pytest.raises(ValueError, match="Score offsets"):
        prepare_fp32(source, tmp_path / "changed", **settings)
    assert not (tmp_path / "changed").exists()
    with pytest.raises(ValueError, match="incompatible architecture"):
        Decision2TrainingBundle.from_local(
            source, tmp_path / "eos", model_id=MODEL_ID, **settings
        )


def training_inputs(root: Path, **settings) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    config = {"max_steps": 2, "batch_size": 2, "encoder_lr": 1e-4, "task_lr": 1e-3}
    config.update(settings)
    (root / "config.json").write_text(json.dumps(config))
    (root / "policy.json").write_text(json.dumps({"min_decision_accuracy": 0}))
    return {
        "train_data": write_rows(
            root / "train.jsonl", [row(state, n % 3) for n, state in enumerate(STATES)]
        ),
        "eval_data": write_rows(
            root / "eval.jsonl",
            [
                row("Customer: refund the duplicate charge.", 0),
                row("Did the owner ask for it?", 1, type="noul", criteria=None),
            ],
        ),
        "training_config": root / "config.json",
        "validation_policy": root / "policy.json",
    }


@pytest.mark.parametrize("model_id", [MODEL_ID, KAI_ID])
def test_training_exports_one_head_and_resumes_a_partial_checkpoint(tmp_path, model_id):
    bundle = base_checkpoint(tmp_path, model_id)
    inputs = training_inputs(tmp_path / "inputs", checkpoint_steps=1)
    events = []

    def stop_after_first_checkpoint(event):
        events.append(event)
        if event["stage"] == "checkpoint_saved" and event["step"] == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt), Trainer(bundle) as trainer:
        trainer.train(
            **inputs,
            output_dir=tmp_path / "stopped",
            progress=stop_after_first_checkpoint,
        )
    partial = TrainingBundle.open(tmp_path / "stopped" / "checkpoint")
    assert provenance(partial.path)["partial"]["step"] == 1
    assert provenance(partial.path)["parent_revision"] == REVISION
    assert not (tmp_path / "stopped" / "inference").exists()

    events.clear()
    with Trainer(partial) as trainer:
        result = trainer.train(
            **inputs, output_dir=tmp_path / "resumed", progress=events.append
        )
    training = [event for event in events if event["stage"] == "training"]
    assert [event["step"] for event in training] == [1, 2]
    assert all(
        isinstance(event["loss"], float) for event in training if "loss" in event
    )
    assert events[-1] == {"stage": "validated"}
    checkpoint = provenance(result.checkpoint_dir)
    assert "partial" not in checkpoint
    assert checkpoint["parent_revision"] == provenance(partial.path)["revision"]
    assert len(checkpoint["revision"]) == 64
    report = json.loads(result.validation_report.read_text())
    assert report["examples"] == 2
    assert report["action_accuracy"] is None
    assert report["checkpoint_revision"] == checkpoint["revision"]
    assert provenance(result.inference_bundle)["revision"] == checkpoint["revision"]


def test_decision2_policy_has_no_action_gate(tmp_path):
    bundle = base_checkpoint(tmp_path)
    inputs = training_inputs(tmp_path / "inputs")
    inputs["validation_policy"].write_text(
        json.dumps({"min_decision_accuracy": 0, "min_action_accuracy": 0})
    )
    with (
        Decision2Trainer(bundle) as trainer,
        pytest.raises(ValueError, match="no action head"),
    ):
        trainer.prepare_fp32(
            eval_data=inputs["eval_data"],
            validation_policy=inputs["validation_policy"],
            output_dir=tmp_path / "prepared",
        )
    assert not (tmp_path / "prepared").exists()
    policy = {"min_decision_accuracy": 0.5}
    with pytest.raises(ValueError, match="quality policy"):
        _quality_report(
            tmp_path,
            policy,
            revision=REVISION,
            examples=2,
            decisions=0,
            actions=None,
            largest_error=0.0,
        )
    assert not (tmp_path / "validation_report.json").exists()
    report = _quality_report(
        tmp_path,
        policy,
        revision=REVISION,
        examples=2,
        decisions=1,
        actions=None,
        largest_error=0.0,
    )
    assert json.loads(report.read_text())["action_accuracy"] is None


def test_official_checkpoint_exports_the_upstream_answers(tmp_path):
    value = os.environ.get("VIS_DECISION2_CHECKPOINT")
    if not value:
        pytest.skip(
            "Set VIS_DECISION2_CHECKPOINT to run the offline real-checkpoint test"
        )
    source = Path(value)
    report = prepare_fp32(
        source,
        tmp_path / "inference",
        license_file=source / "LICENSE",
        revision=REVISION,
    )
    assert report["max_abs_probability_error"] <= 1e-3
    model = provenance(tmp_path / "inference")["model"]
    assert ("score_bias" in provenance(tmp_path / "inference")) == (model == KAI_ID)
    runtime = ort.InferenceSession(
        str(tmp_path / "inference" / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    arguments = make_batch(PROBES, tokenizer)
    logits = runtime.run(
        None,
        {
            name: value.numpy()
            for name, value in zip(
                ("input_ids", "candidate_positions", "query_positions"),
                arguments,
                strict=True,
            )
        },
    )[0]
    # Upstream FP32 answers for the three probes: refund, true and a score level.
    expected = {
        MODEL_ID: [
            [0.9886, 0.0045, 0.0069],
            [0.0292, 0.9708],
            [0.2933, 0.7019, 0.0049],
        ],
        KAI_ID: [[0.8997, 0.0558, 0.0445], [0.3445, 0.6555], [0.6674, 0.2783, 0.0543]],
    }[model]
    for index, probabilities in enumerate(expected):
        count = len(probabilities)
        scores = np.exp(logits[index, :count] - logits[index, :count].max())
        assert np.allclose(scores / scores.sum(), probabilities, atol=2e-3)
