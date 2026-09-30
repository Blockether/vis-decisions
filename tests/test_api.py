"""One public API selects both families and preserves native trainer semantics."""

import json
from types import SimpleNamespace

import pytest
from test_decision_training import checkpoint
from test_gliner_decision_training import source_checkpoint

from blockether.vis_decisions import Trainer, TrainingBundle, api


@pytest.mark.parametrize("model_id", ["laya-typed-decisions", "gliner2.5-decide-1b"])
def test_fetch_forwards_keyword_only_checkpoint_contract(
    tmp_path, monkeypatch, model_id
):
    calls = []
    expected = object()
    family = (
        api.LayaTrainingBundle
        if model_id == "laya-typed-decisions"
        else api.GlinerTrainingBundle
    )

    def fetch(cls, *, model_ref, cache_dir):
        calls.append((cls, model_ref, cache_dir))
        return expected

    monkeypatch.setattr(family, "fetch", classmethod(fetch))
    reference = f"{model_id}@{'a' * 40}"
    assert TrainingBundle.fetch(reference, tmp_path) is expected
    assert calls == [(family, reference, tmp_path)]


@pytest.mark.parametrize("model_id", ["laya-typed-decisions", "gliner2.5-decide-1b"])
def test_open_selects_verified_family_and_trainer_delegates(
    tmp_path, monkeypatch, model_id
):
    if model_id == "laya-typed-decisions":
        path = checkpoint(tmp_path / "checkpoint")
        implementation = "ModernBertTrainer"
    else:
        source = source_checkpoint(tmp_path / "source", model_id)
        license_file = tmp_path / "LICENSE"
        license_file.write_text("Apache-2.0")
        path = api.GlinerTrainingBundle.from_local(
            source,
            tmp_path / "checkpoint",
            model_id=model_id,
            revision="a" * 40,
            license_file=license_file,
        ).path
        implementation = "GlinerTrainer"
    bundle = TrainingBundle.open(path)
    calls = []
    result = object()

    class NativeTrainer:
        def __init__(self, selected):
            assert selected is bundle

        def finetune(self, **kwargs):
            calls.append(("train", kwargs))
            return result

        def prepare_fp32(self, **kwargs):
            calls.append(("prepare", kwargs))
            return result

        def close(self):
            calls.append(("close", {}))

    monkeypatch.setattr(api, implementation, NativeTrainer)
    arguments = dict(
        eval_data="eval.jsonl", validation_policy="policy.json", output_dir="out"
    )
    with Trainer(bundle) as trainer:
        assert (
            trainer.train(
                **arguments, train_data="train.jsonl", training_config="config.json"
            )
            is result
        )
        assert trainer.prepare(**arguments) is result
    assert [operation for operation, _ in calls] == ["train", "prepare", "close"]
    assert calls[0][1]["train_data"] == "train.jsonl"
    assert calls[1][1]["eval_data"] == "eval.jsonl"
    assert calls[1][1]["progress"] is None
    (path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="inventory|checksum|digest"):
        TrainingBundle.open(path)


def test_unknown_models_and_unverified_objects_are_rejected(tmp_path):
    (tmp_path / "PROVENANCE.json").write_text(json.dumps({"model": "unsupported"}))
    with pytest.raises(ValueError, match="Unsupported"):
        TrainingBundle.open(tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        TrainingBundle.fetch("unsupported", tmp_path)
    with pytest.raises(TypeError, match="TrainingBundle"):
        Trainer(SimpleNamespace(path=tmp_path))
