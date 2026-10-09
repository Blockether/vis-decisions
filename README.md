# Vis Decisions

Train Laya, GLiNER and Decision 2.0 decision models with one Python environment and one Vis extension.
Keep your training rows local. Export verified FP32 inference bundles for Vis.

## When to use

- To train from chat, [enable the extension](#use-in-vis).
- To train from Python, [open a complete checkpoint](#train-with-python).
- To continue an interrupted run, [resume a saved checkpoint](#resume-or-continue-training).
- To classify with OpenAI's GPT-6 Luna instead of a local model, [ask an OpenAI decision model](#ask-openai-decision-models).
- To select a model or configure live decisions, use the [Vis decision-model guide](https://github.com/Blockether/vis/blob/main/resources/vis-docs/decision-models.md).

## Use in Vis

Add the extension to your project `vis.yml`:

```yaml
extensions:
  vis-decisions:
    source: https://github.com/Blockether/vis-decisions
    version: '0.4.0'
```

Reload the project. The first installation downloads the pinned Python dependencies.
Local training needs Python 3.12 or newer, enough memory, and space for complete model checkpoints.
Decide-1B and Decision 2.0 need several gigabytes for each checkpoint or export.

Ask Vis to show the decision models and their download sizes.
Then ask it to train a selected checkpoint with your training and evaluation files.
Give it a new output directory and a quality policy.

The extension exposes these tools:

| Tool | Result |
| --- | --- |
| `decisions.models()` | Canonical model revisions and archive sizes from the Vis release |
| `decisions.fetch(model_ref, destination)` | A verified complete checkpoint for `<id>@<revision>` |
| `decisions.inspect_checkpoint(path)` | Local model identity, revision and resumable progress |
| `decisions.train(...)` | Checkpoint, FP32 bundle and held-out quality report |
| `decisions.prepare(...)` | Validated FP32 export without training |
| `decisions.package(inference_bundle, destination)` | An inference-only archive, SHA-256 digest and size |

Training, export and packaging do not publish a model or activate an alias.
Review the validation report before importing the model into a gateway.

## Ask OpenAI decision models

The `Decisions` client also reaches OpenAI classifier models, such as GPT-6 Luna, through your gateway.
Use the same questions and the same answer shapes as for a local model.
Name the model as `openai/<id>`:

```python
answer = decisions.infer(
    model="openai/gpt-6-luna",
    state="A damaged item needs a refund",
    questions={
        "intent": {
            "type": "choice",
            "instructions": "Choose a request",
            "criteria": ["refund", "repair"],
        }
    },
)
# Prints: refund openai
print(answer["answers"]["intent"]["choice"], answer["routing"]["provider"])
```

The gateway needs an OpenAI API key, from the `openai` provider or `OPENAI_API_KEY`.
A ChatGPT (Codex) sign-in does not work with the OpenAI Decisions API.
`decisions.list_models()` shows `"available": true` when the gateway has a key.
OpenAI answers have no `action` head. If OpenAI declines a question, its answer is `{"type": "refusal"}`.
Each request is a paid OpenAI call, and the state goes to OpenAI.

## Train with Python

Clone this repository. Install the locked environment:

```sh
uv sync --frozen --python 3.12
```

The same environment trains all three families.
It pins Transformers 5.17.0, GLiNER2 2.0.0, Laya 0.3.22 and PyTorch 2.14.0.
The model releases contain weights and metadata, not Python dependency archives.

Find a pinned model revision with `decisions.models()` or the Vis model catalog.
Download the complete training checkpoint with `TrainingBundle.fetch(model_ref, destination)`.
This explicit call downloads model assets. Local checkpoint loading does not access the network.

```python
from blockether.vis_decisions import Trainer, TrainingBundle

checkpoint = TrainingBundle.open("./models/training")
with Trainer(checkpoint) as trainer:
    result = trainer.train(
        train_data="./data/train.jsonl",
        eval_data="./data/eval.jsonl",
        training_config="./data/training.json",
        validation_policy="./data/quality.json",
        output_dir="./runs/customer-support-v1",
        progress=print,
    )
print(result.validation_report)
```

Use complete checkpoints, not encoder-only weights or ONNX inference bundles.
The loader verifies each file and selects the family from model provenance.
Training rows and evaluation rows must be disjoint.
Keep the output directory new for each run.
Training uses one CPU thread for each physical core, or for each performance core on Apple silicon.
To choose another number, set `OMP_NUM_THREADS` before you start Python.

Decision 2.0 has no action head. Its quality policy sets only `min_decision_accuracy`.
Its validation report gives no action accuracy.

The [Vis guide](https://github.com/Blockether/vis/blob/main/resources/vis-docs/decision-models.md#train-locally-with-the-python-sdk)
explains the row formats, training settings and quality policy.
The Python package also provides `Decisions(gateway)` for explicit gateway import and activation.

## Resume or continue training

Set `checkpoint_steps` in the training configuration to save partial checkpoints.
Stop after a checkpoint has been saved. Reopen the saved `checkpoint` directory.
Call `Trainer.train` with the same ordered rows and settings, using a new output directory.

A matching partial run resumes its saved step and batch order.
The optimizer is recreated, so this is not an exact optimizer-state continuation.
Changed rows or settings start a new run from the saved weights.
A completed checkpoint can start another fine-tuning run on new rows.

## Model assets and precision

Model assets remain in [Vis assets-pack](https://github.com/Blockether/vis/releases/tag/assets-pack).
The canonical catalog is maintained in Vis and published as `decisions.json`.
This repository does not keep a second catalog.

Decide-1B FP32 and training archives exceed GitHub's per-file limit.
They use ordered parts of at most 2,000,000,000 bytes.
The downloader verifies every part, joins them, and verifies the complete archive before extraction.
No conversion to FP16, BF16 or quantized weights is used to reduce their size.

To prepare the Decision 2.0 FP32 bundle from the upstream checkpoint, run `scripts/export_decision2.py`.
To pack prepared bundles for the release, run `scripts/build_release.py`.
It verifies each file, writes reproducible archives and prints their sizes and digests for the catalog.

## Development

```sh
uv sync --frozen --python 3.12
uv run ruff format --check .
uv run ruff check .
uv run python -m pytest -q
uv build
```

Most tests do not download model weights.
To run the offline Laya acceptance tests, set `VIS_LAYA_TRAINING_DIR` to a verified complete checkpoint.
To run the Decide-1B acceptance test, set `VIS_GLINER_DECIDE_1B_CHECKPOINT` and `VIS_GLINER_LICENSE`.
To run the Decision 2.0 export test, set `VIS_DECISION2_CHECKPOINT` to the upstream checkpoint.
The real-model tests cover train, stop, resume, export and reopening in a new process.
