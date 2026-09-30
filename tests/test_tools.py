"""The extension exports typed tools, explicit activities and no hidden activation."""

import json
import runpy
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import blockether.vis.extension as vis
import pytest
from test_decision_training import checkpoint

from blockether.vis_decisions import tools


@pytest.fixture
def registered(monkeypatch):
    extensions = []
    monkeypatch.setattr(vis, "register_extension", extensions.append)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "extension.py"))
    assert len(extensions) == 1
    return extensions[0]


def test_registration_owns_all_six_typed_methods_and_effects(registered):
    assert (registered.name, registered.alias) == ("vis-decisions", "decisions")
    members = registered.symbols[0].contract["members"]
    tags = {member["name"].rsplit(".", 1)[-1]: member["tag"] for member in members}
    assert tags == {
        "models": "observation",
        "inspect_checkpoint": "observation",
        "fetch": "mutation",
        "train": "mutation",
        "prepare": "mutation",
        "package": "mutation",
    }
    for member in members:
        assert member["returns"]
        name = member["name"].rsplit(".", 1)[-1]
        activity = getattr(tools.DecisionTools, name).__vis_symbol_activity__
        assert activity.label[0].isupper() and "_" not in activity.label
        assert activity.show_start is (name not in {"models", "inspect_checkpoint"})


@pytest.mark.parametrize(
    "method", ["models", "fetch", "inspect_checkpoint", "train", "prepare", "package"]
)
def test_every_activity_keeps_running_failure_cancelled_and_result_detail(
    registered, method
):
    activity = getattr(tools.DecisionTools, method).__vis_symbol_activity__
    running = activity.render(phase="start")
    assert running.summary == "Running"
    failure = activity.render(phase="failure", error=RuntimeError("invalid checksum"))
    assert failure.summary == "Failed"
    assert [part.text for part in failure.content] == ["invalid checksum"]
    assert activity.render(phase="cancelled").summary == "Cancelled"
    if method == "models":
        result = [
            tools.ModelInfo(
                "gliner2.5-decide-1b", "a" * 40, 4_142_645_738, 4_413_546_277
            )
        ]
        rendered = activity.render(phase="success", result=result)
        assert rendered.summary == "1 models"
        assert "4,413,546,277" in rendered.content[0].text
        empty = activity.render(phase="success", result=[])
        assert (empty.summary, empty.content[0].text) == ("0 models", "No models")
    else:
        result = tools.ArchiveInfo("/tmp/inference.zip", "a" * 64, 123)
        rendered = activity.render(phase="success", result=result)
        assert rendered.summary == "Completed"
        assert "/tmp/inference.zip" in rendered.content[0].text
        assert "a" * 64 in rendered.content[0].text


def test_catalog_and_checkpoint_tools_return_frozen_records(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tools,
        "_manifest",
        lambda: [
            {
                "id": "laya-typed-decisions",
                "revision": "pinned",
                "artifacts": {"inference": {"bytes": 10}, "training": {"bytes": 20}},
            }
        ],
    )
    result = tools.DecisionTools().models()
    assert result == [tools.ModelInfo("laya-typed-decisions", "pinned", 10, 20)]
    with pytest.raises(FrozenInstanceError):
        result[0].training_bytes = 0
    root = checkpoint(tmp_path / "checkpoint")
    inspected = tools.DecisionTools().inspect_checkpoint(str(root))
    assert inspected.model_id == "laya-typed-decisions"
    assert (inspected.path, inspected.step, inspected.max_steps) == (
        str(root.resolve()),
        0,
        0,
    )
    (root / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        tools.DecisionTools().inspect_checkpoint(str(root))


def test_fetch_forwards_pin_and_returns_verified_inventory(tmp_path, monkeypatch):
    root = checkpoint(tmp_path / "checkpoint")

    def fetch(model_ref, destination):
        assert (model_ref, destination) == (
            "laya-typed-decisions@pinned",
            str(tmp_path),
        )
        return SimpleNamespace(path=root)

    monkeypatch.setattr(tools.TrainingBundle, "fetch", fetch)
    assert tools.DecisionTools().fetch(
        "laya-typed-decisions@pinned", str(tmp_path)
    ).path == str(root)


@pytest.mark.parametrize("operation", ["train", "prepare"])
def test_training_tools_close_trainer_and_keep_validation_metrics(
    tmp_path, monkeypatch, operation
):
    root = checkpoint(tmp_path / "checkpoint")
    report = tmp_path / "validation.json"
    report.write_text(json.dumps({"decision_accuracy": 0.75, "action_accuracy": 1.0}))
    result = SimpleNamespace(
        checkpoint_dir=root,
        inference_bundle=tmp_path / "inference",
        validation_report=report,
    )
    calls = []

    class NativeTrainer:
        def __init__(self, bundle):
            assert bundle.path == root

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            calls.append("closed")

        def train(self, **kwargs):
            calls.append(kwargs)
            return result

        def prepare(self, **kwargs):
            calls.append(kwargs)
            return result

    monkeypatch.setattr(tools, "Trainer", NativeTrainer)
    kwargs = dict(
        eval_data="eval.jsonl", validation_policy="policy.json", output_dir="output"
    )
    if operation == "train":
        kwargs.update(train_data="train.jsonl", training_config="config.json")
    outcome = getattr(tools.DecisionTools(), operation)(str(root), **kwargs)
    assert (outcome.decision_accuracy, outcome.action_accuracy) == (0.75, 1.0)
    assert calls == [kwargs, "closed"]


def test_package_keeps_digest_and_size(tmp_path, monkeypatch):
    archive = tmp_path / "inference.zip"
    monkeypatch.setattr(tools, "package", lambda source, target: ("a" * 64, 123))
    result = tools.DecisionTools().package(str(tmp_path / "inference"), str(archive))
    assert result == tools.ArchiveInfo(str(archive), "a" * 64, 123)
