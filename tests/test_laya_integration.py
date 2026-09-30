"""Offline SDK acceptance: local checkpoint → train → export → reopen FP32."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from blockether.vis_decisions import Trainer, TrainingBundle


@pytest.fixture(scope="module")
def base():
    path = os.environ.get("VIS_LAYA_TRAINING_DIR")
    if not path:
        pytest.skip("Set VIS_LAYA_TRAINING_DIR to a verified complete checkpoint")
    return TrainingBundle.open(path)


@pytest.fixture
def labeled(tmp_path):
    data = [
        {
            "state": "Please refund my damaged purchase.",
            "question": {
                "type": "choice",
                "instructions": "Choose a request.",
                "criteria": ["refund", "repair"],
            },
            "target": 0,
            "action": 0,
        },
        {
            "state": "The account is blocked after repeated billing issues.",
            "question": {
                "type": "score",
                "instructions": "Rate urgency.",
                "criteria": ["low", "medium", "high"],
            },
            "target": 2,
            "action": 1,
        },
        {
            "state": "The parcel is still missing.",
            "question": {"type": "noul", "instructions": "Is the parcel missing?"},
            "target": 1,
            "action": 0,
        },
    ]
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text("\n".join(json.dumps(row) for row in data) + "\n")
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(
        json.dumps({**data[0], "state": "A second item arrived damaged."}) + "\n"
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"min_decision_accuracy": 0.0, "min_action_accuracy": 0.0})
    )
    training_config = tmp_path / "config.json"
    training_config.write_text(
        json.dumps(
            {"epochs": 1, "learning_rate": 1e-5, "train_encoder": False, "max_steps": 1}
        )
    )
    return train_file, eval_file, policy, training_config


def _deny_network(*_args, **_kwargs):
    raise AssertionError("Offline Laya SDK validation attempted network access")


def test_prepare_fp32_without_network(base, labeled, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    _, evaluation, policy, _ = labeled
    progress = []
    with Trainer(base) as trainer:
        result = trainer.prepare(
            eval_data=evaluation,
            validation_policy=policy,
            output_dir=tmp_path / "prepared",
            progress=progress.append,
        )
    assert [event["stage"] for event in progress] == [
        "loading",
        "exporting",
        "validated",
    ]
    report = json.loads(result.validation_report.read_text())
    assert report["examples"] == 3
    assert 0 <= report["decision_accuracy"] <= 1
    assert report["status"] == "evaluated_not_approved_for_autonomous_actions"
    assert (result.inference_bundle / "model.onnx").is_file()
    assert (result.inference_bundle / "tokenizer/tokenizer.json").is_file()
    assert not (result.inference_bundle / "model.safetensors").exists()


def test_finetune_resume_and_reopen_fp32_without_network(
    base, labeled, tmp_path, monkeypatch
):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(socket.socket, "connect", _deny_network)
    train, evaluation, policy, config = labeled
    progress = []
    config.write_text(
        json.dumps(
            {
                "epochs": 3,
                "learning_rate": 1e-5,
                "train_encoder": False,
                "max_steps": 3,
                "checkpoint_steps": 1,
            }
        )
    )

    def stop_after_saved_step(event):
        if event["stage"] == "checkpoint_saved" and event["step"] == 1:
            raise InterruptedError("Requested training stop")

    with Trainer(base) as trainer:
        with pytest.raises(InterruptedError, match="Requested training stop"):
            trainer.train(
                train_data=train,
                eval_data=evaluation,
                training_config=config,
                validation_policy=policy,
                output_dir=tmp_path / "interrupted",
                progress=stop_after_saved_step,
            )
    partial = TrainingBundle.open(tmp_path / "interrupted/checkpoint")
    metadata = json.loads((partial.path / "PROVENANCE.json").read_text())
    assert metadata["partial"]["step"] == 1
    assert metadata["partial"]["max_steps"] == 3
    assert not (tmp_path / "interrupted/inference").exists()
    with Trainer(partial) as trainer:
        result = trainer.train(
            train_data=train,
            eval_data=evaluation,
            training_config=config,
            validation_policy=policy,
            output_dir=tmp_path / "resumed",
            progress=progress.append,
        )
    training = [event for event in progress if event["stage"] == "training"]
    assert [event["step"] for event in training] == [1, 2, 3]
    assert "partial" not in json.loads(
        (result.checkpoint_dir / "PROVENANCE.json").read_text()
    )
    assert result.checkpoint_dir != base.path
    resumed = TrainingBundle.open(result.checkpoint_dir)
    # New rows start a new trajectory from the completed checkpoint's weights.
    train.write_text(
        json.dumps(
            {
                "state": "Another delivered item needs repair.",
                "question": {
                    "type": "choice",
                    "instructions": "Choose a request.",
                    "criteria": ["refund", "repair"],
                },
                "target": 1,
                "action": 0,
            }
        )
        + "\n"
    )
    config.write_text(
        json.dumps(
            {
                "epochs": 1,
                "learning_rate": 1e-5,
                "train_encoder": False,
                "max_steps": 1,
            }
        )
    )
    continued_progress = []
    with Trainer(resumed) as trainer:
        continued = trainer.train(
            train_data=train,
            eval_data=evaluation,
            training_config=config,
            validation_policy=policy,
            output_dir=tmp_path / "continued",
            progress=continued_progress.append,
        )
    assert continued_progress[0]["step"] == 0
    child = json.loads((continued.checkpoint_dir / "PROVENANCE.json").read_text())
    parent = json.loads((resumed.path / "PROVENANCE.json").read_text())
    assert child["parent_revision"] == parent["revision"]
    TrainingBundle.open(resumed.path)  # Opening must not rewrite a checkpoint.
    assert (result.inference_bundle / "model.onnx").is_file()
    assert not (result.inference_bundle / "model.safetensors").exists()
    report = json.loads(result.validation_report.read_text())
    assert report["max_abs_logit_error"] < 0.5
    # A new interpreter sees only saved local files, not the trainer's tensors.
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "from blockether.vis_decisions.training import TrainingBundle; "
            "from blockether.vis_decisions._training import load_onnx; "
            "import sys; TrainingBundle.open(sys.argv[1]); "
            "load_onnx(sys.argv[2]).predict('Please refund my order.', "
            "{'request': {'type': 'choice', 'instructions': 'Choose a request.', "
            "'criteria': ['refund', 'repair']}})",
            str(resumed.path),
            str(result.inference_bundle),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert process.returncode == 0, process.stderr[-1000:]
