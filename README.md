# Vis Decisions

Train, resume and validate Laya, GLiNER, Decision 2.0 and D3 decision models in Vis. Keep training rows local and prepare verified FP32 bundles for publication.

## Install

Add the extension to `vis.yml` and reload Vis. Local training requires Python 3.12 or newer and space for complete checkpoints.

```yaml
extensions:
  vis-decisions:
    source: https://github.com/Blockether/vis-decisions
    version: '0.6.4'
```

## Use

Ask Vis to list model revisions and sizes. Select a complete checkpoint, then give Vis your training file, evaluation file, a new output directory and a quality policy. Use different rows in the two files.

| Tool | Result |
| --- | --- |
| `decisions.models()` | Model revisions and download sizes. |
| `decisions.fetch(model_ref, destination)` | A verified training checkpoint. |
| `decisions.inspect_checkpoint(path)` | The checkpoint identity, revision and saved step. |
| `decisions.train(...)` | A checkpoint, FP32 export and quality report. |
| `decisions.prepare(...)` | A validated FP32 export without training. |
| `decisions.package(inference_bundle, destination)` | An inference archive, SHA-256 digest and size. |

Training does not publish the model or activate an alias. Review the quality report before you import the model. To train in Python, resume a run, or ask an OpenAI classifier through the gateway, see the [training and inference guide](docs/training-and-inference.md). Use the [Vis decision-model guide](https://github.com/Blockether/vis/blob/main/resources/vis-docs/decision-models.md) to configure live decisions.

## Configuration

| Training key | Use |
| --- | --- |
| `device` | Select `auto`, `cpu`, `mps` or `cuda`. |
| `checkpoint_steps` | Save partial checkpoints at an interval. |
| `lora` | Train low-rank adapters with `rank`, `alpha` and `dropout`. |
| `train_encoder` | Train all Laya encoder weights instead of LoRA. |
| `max_input_tokens` | Limit the input tokens for Decision 2.0 and D3. |

## Development

```bash
uv sync --frozen --python 3.12
uv run ruff format --check . && uv run ruff check .
uv run python -m pytest -q
```
