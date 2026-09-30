# Unified decision-model training

One runtime, one extension, model assets stay in Vis releases.

## Context

Move `packages/vis-agent/src/blockether/vis/decisions/` from Vis into
`src/blockether/vis_decisions/`. Vis keeps ONNX inference, the canonical
`resources/vis-models/decisions.json`, verified downloads and durable gateway jobs.
Remove wheelhouse artifacts and the two conflicting training environments.
Do not replace GLiNER task heads with a bare Transformers encoder.
Do not reduce weight precision to meet GitHub's per-file limit without a new
validated model variant. Preserve existing split Decide-1B downloads.

## 1. Runtime and unified library

Rationale: both families need the same public training operations and one environment.
Data: Transformers 5.17.0, GLiNER2 2.0.0, Laya 0.3.22, migrated SDK tests.
Acceptance criteria: train, stop, resume, validate and export both families; FP32 parity.
Unknowns: resolved by offline Laya and Decide-1B training, resume and FP32 parity tests.

## 2. Extension and Vis consumers

Rationale: move domain-specific SDK code out without duplicating the catalog.
Data: the Vis extension API, gateway worker, model CLI and native smoke tests.
Acceptance criteria: registered tools execute; jobs use the new worker; no wheelhouse path remains.
Unknowns: resolved by the real registered extension call and durable gateway worker tests.

## 3. Assets, verification and publication

Rationale: models stay usable from GitHub Releases without dependency archives.
Data: release asset sizes and digests, canonical catalog, affected JVM/Python/native tests.
Acceptance criteria: Decide-1B fits release assets, verified downloads, new repository and scoped commits pushed.
Unknowns: a single-file lossless archive still exceeds the limit; retain verified ordered parts.

## Plan state

All three phases are complete.
Real Laya and Decide-1B training, partial resume and FP32 exports pass.
The native SDK uploaded the 3.84 GB Decide-1B bundle and inferred with it.
The public repository and v0.1.0 release are published.
Extension Center pins the reviewed release revision.
The live Vis catalog verifies both pinned training families.
Four wheelhouse archives are removed, and release checksums include the canonical catalog.
All original model assets keep their sizes and digests, including all six Decide-1B parts.
The release workflow selects only distribution archives, with a regression test.
The broader gateway suite has an unrelated SIGINT shell-control failure outside this migration.
