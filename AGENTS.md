# Vis Decisions

- `src/blockether/vis_decisions/` owns the ordinary Python library. Only `extension.py` imports the Vis extension host.
- One environment runs Laya, GLiNER and Decision 2.0 with the pinned Transformers 5 version in `pyproject.toml` and `uv.lock`.
- `Blockether/vis` owns model catalog metadata and model release assets. This repository consumes that catalog; do not copy it.
- Preserve FP32 inference parity, immutable model versions, disjoint validation rows and private training data.
- Every extension binding owns an Activity presentation and tests for running, success, failure and empty states.
- Formatting and lint use ruff. Tests use pytest. Publish only verified changes, with the configured human Git identity.
