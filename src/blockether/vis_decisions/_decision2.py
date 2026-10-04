"""Offline Decision 2.0 export: a Qwen3 or Qwen3.5 backbone with a candidate head.

This module is optional: importing the gateway client never loads Torch. The ONNX
graph maps each Qwen3.5 gated delta rule layer to the ONNX Runtime
``com.microsoft::LinearAttention`` CPU kernel. No checkpoint is downloaded here.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from transformers import Qwen3_5TextModel, Qwen3Model
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_rotary_pos_emb,
    l2norm,
    torch_chunk_gated_delta_rule,
)

from ._models import DECISION2, DECISION2_BACKBONES, DECISION2_PROMPT_VERSION
from .decision2_training import _checkpoint_model
from .training import _inventory, _sha256

FAMILY = "decision2"
PROMPT_VERSION = DECISION2_PROMPT_VERSION
INPUT_NAMES = ("input_ids", "candidate_positions", "query_positions")
MAX_INPUT_TOKENS = 4096
# Each backbone type gives its class, rotary function and attention output gate.
_BACKBONES = {
    "qwen3_5_text": (Qwen3_5TextModel, apply_rotary_pos_emb, True),
    "qwen3": (Qwen3Model, modeling_qwen3.apply_rotary_pos_emb, False),
}
_SCORE_BIAS_FORMAT = "dev2-score-bias-v1"
_SCORED_FILES = {
    "decision_config.json",
    "decision_head.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
}
_FINGERPRINT_SUFFIXES = {".json", ".safetensors", ".bin", ".model", ".txt"}
PROBES = (
    {
        "state": "Customer: my parcel arrived broken, I want my money back.",
        "task_type": "choice",
        "instructions": "Pick the support queue for this message.",
        "options": [
            {"key": "refund", "description": "Refunds and returns"},
            {"key": "shipping", "description": "Delivery status questions"},
            {"key": "other", "description": None},
        ],
    },
    {
        "state": {"transfer": "pending", "requested_by": "account owner"},
        "task_type": "noul",
        "instructions": "Did the account owner request this transfer?",
        "options": [
            {"key": "false", "description": "No"},
            {"key": "true", "description": "Yes"},
        ],
    },
    {
        "state": "The answer is correct but leaves out one of the three steps.",
        "task_type": "score",
        "instructions": "Rate the completeness of the answer.",
        "options": [
            {"key": "0", "description": "Missing"},
            {"key": "1", "description": "Partial"},
            {"key": "2", "description": "Complete"},
        ],
    },
)


def canonical(value: Any) -> str:
    """Serialize JSON exactly as the Decision 2.0 prompt contract requires."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _payload(value: Any) -> str:
    return value if isinstance(value, str) else canonical(value)


def segments(row: Mapping[str, Any]) -> tuple[str, list[str], str]:
    """Return the prompt prefix, one segment for each option and the query suffix."""
    prefix = (
        f"Context:\n{_payload(row['state'])}\n\n"
        f"Task type: {row['task_type']}\nQuestion:\n"
        f"{_payload(row['instructions'])}\nOptions:"
    )
    options = [
        "\n<option>\n"
        + canonical({"key": option["key"], "description": option["description"]})
        + "\n</option>"
        for option in row["options"]
    ]
    suffix = (
        "\n\nSelect the single option best supported by the context and "
        "instructions.\nDecision:"
    )
    return prefix, options, suffix


def encode(
    row: Mapping[str, Any], tokenizer: Tokenizer, max_length: int = MAX_INPUT_TOKENS
) -> dict[str, Any]:
    """Tokenize each segment without special tokens; never truncate."""
    prefix, options, suffix = segments(row)
    ids = list(tokenizer.encode(prefix, add_special_tokens=False).ids)
    endpoints = []
    for option in options:
        part = tokenizer.encode(option, add_special_tokens=False).ids
        if not part:
            raise ValueError("Decision 2.0 option has no tokens")
        ids.extend(part)
        endpoints.append(len(ids) - 1)
    ids.extend(tokenizer.encode(suffix, add_special_tokens=False).ids)
    if len(ids) > max_length:
        raise ValueError(f"{len(ids)} tokens exceed the {max_length} token limit")
    return {
        "ids": ids,
        "candidate_positions": endpoints,
        "query_position": len(ids) - 1,
        "token_ids_sha256": hashlib.sha256(canonical(ids).encode("utf-8")).hexdigest(),
    }


class CandidateHead(torch.nn.Module):
    """Score each option endpoint against the global query endpoint."""

    def __init__(self, hidden_size: int, head_dim: int = 256) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.candidate_norm = torch.nn.LayerNorm(hidden_size)
        self.query_norm = torch.nn.LayerNorm(hidden_size)
        self.key = torch.nn.Linear(hidden_size, head_dim, bias=False)
        self.query = torch.nn.Linear(hidden_size, head_dim, bias=False)
        self.candidate_mlp = torch.nn.Linear(hidden_size, head_dim)
        self.query_mlp = torch.nn.Linear(hidden_size, head_dim, bias=False)
        self.scalar = torch.nn.Linear(head_dim, 1, bias=False)

    def forward(self, candidates: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        candidate = self.candidate_norm(candidates.float())
        global_query = self.query_norm(query.float())
        bilinear = (self.key(candidate) * self.query(global_query)[:, None, :]).sum(
            -1
        ) / math.sqrt(self.head_dim)
        nonlinear = self.scalar(
            F.gelu(
                self.candidate_mlp(candidate) + self.query_mlp(global_query)[:, None, :]
            )
        ).squeeze(-1)
        return bilinear + nonlinear


class Decision2Model(torch.nn.Module):
    """The pinned FP32 backbone and head, with the upstream reference forward."""

    def __init__(self, backbone: torch.nn.Module, head: CandidateHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        input_ids: torch.Tensor,
        candidate_positions: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, use_cache=False).last_hidden_state
        return _score(self.head, hidden, candidate_positions, query_positions)


def _score(head, hidden, candidate_positions, query_positions):
    width = hidden.shape[-1]
    candidates = torch.gather(
        hidden, 1, candidate_positions.unsqueeze(-1).expand(-1, -1, width)
    )
    query = torch.gather(
        hidden, 1, query_positions.view(-1, 1, 1).expand(-1, 1, width)
    ).squeeze(1)
    return head(candidates, query)


def load_checkpoint(source: str | Path) -> tuple[Decision2Model, Tokenizer]:
    """Load a local full checkpoint in FP32 without remote code or network access."""
    source = Path(source).expanduser().resolve()
    backbone_class = _BACKBONES[DECISION2_BACKBONES[_checkpoint_model(source)]][0]
    metadata = json.loads((source / "decision_config.json").read_text("utf-8"))
    backbone = backbone_class.from_pretrained(
        source / "backbone", dtype=torch.float32, attn_implementation="sdpa"
    )
    head = CandidateHead(backbone.config.hidden_size, metadata.get("head_dim", 256))
    head.load_state_dict(load_file(source / "decision_head.safetensors"))
    model = Decision2Model(backbone, head.float()).eval()
    return model, Tokenizer.from_file(str(source / "tokenizer.json"))


class _GatedDeltaRule(torch.autograd.Function):
    """Apply the gated delta rule to L2-normalized queries and keys."""

    @staticmethod
    def forward(ctx, query, key, value, state, decay, beta, heads, scale):
        batch, length, width = query.shape
        output, _ = torch_chunk_gated_delta_rule(
            query.reshape(batch, length, heads, width // heads),
            key.reshape(batch, length, heads, width // heads),
            value.reshape(batch, length, heads, -1),
            g=decay,
            beta=beta,
            use_qk_l2norm_in_kernel=False,
        )
        return output.reshape(batch, length, -1)

    @staticmethod
    def symbolic(graph, query, key, value, state, decay, beta, heads, scale):
        output, present = graph.op(
            "com.microsoft::LinearAttention",
            query,
            key,
            value,
            state,
            decay,
            beta,
            q_num_heads_i=heads,
            kv_num_heads_i=heads,
            update_rule_s="gated_delta",
            scale_f=scale,
            outputs=2,
        )
        output.setType(value.type())
        present.setType(state.type())
        return output


def _linear_attention(module, hidden: torch.Tensor) -> torch.Tensor:
    batch, length, _ = hidden.shape
    mixed = module.in_proj_qkv(hidden).transpose(1, 2)
    mixed = F.silu(module.conv1d(mixed)[:, :, :length]).transpose(1, 2)
    query, key, value = torch.split(
        mixed, [module.key_dim, module.key_dim, module.value_dim], dim=-1
    )
    heads = module.num_v_heads
    repeat = heads // module.num_k_heads
    query = l2norm(query.reshape(batch, length, module.num_k_heads, -1))
    key = l2norm(key.reshape(batch, length, module.num_k_heads, -1))
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    beta = module.in_proj_b(hidden).sigmoid()
    decay = -module.A_log.float().exp() * F.softplus(
        module.in_proj_a(hidden).float() + module.dt_bias
    )
    state = hidden.new_zeros((batch, heads, module.head_k_dim, module.head_v_dim))
    core = _GatedDeltaRule.apply(
        query.reshape(batch, length, -1),
        key.reshape(batch, length, -1),
        value,
        state,
        decay,
        beta,
        heads,
        module.head_k_dim**-0.5,
    )
    gate = module.in_proj_z(hidden)
    core = module.norm(
        core.reshape(-1, module.head_v_dim), gate.reshape(-1, module.head_v_dim)
    )
    return module.out_proj(core.reshape(batch, length, -1))


def _full_attention(module, hidden, cos, sin, rotate, gated) -> torch.Tensor:
    batch, length, _ = hidden.shape
    size = module.head_dim
    if gated:
        query, gate = torch.chunk(
            module.q_proj(hidden).view(batch, length, -1, size * 2), 2, dim=-1
        )
        gate = gate.reshape(batch, length, -1)
    else:
        query = module.q_proj(hidden).view(batch, length, -1, size)
    query = module.q_norm(query).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden).view(batch, length, -1, size))
    value = module.v_proj(hidden).view(batch, length, -1, size).transpose(1, 2)
    query, key = rotate(query, key.transpose(1, 2), cos, sin)
    groups = module.num_key_value_groups
    heads = key.shape[1]
    key = (
        key[:, :, None]
        .expand(batch, heads, groups, length, size)
        .reshape(batch, heads * groups, length, size)
    )
    value = (
        value[:, :, None]
        .expand(batch, heads, groups, length, size)
        .reshape(batch, heads * groups, length, size)
    )
    output = F.scaled_dot_product_attention(
        query, key, value, is_causal=True, scale=module.scaling
    )
    output = output.transpose(1, 2).reshape(batch, length, -1)
    if gated:
        output = output * torch.sigmoid(gate)
    return module.o_proj(output)


class DecisionGraph(torch.nn.Module):
    """Export-safe forward of the same weights: token IDs to option logits."""

    def __init__(self, model: Decision2Model) -> None:
        super().__init__()
        self.model = model
        config = model.backbone.config
        _, self.rotate, self.gated = _BACKBONES[config.model_type]
        self.layer_types = list(config.layer_types)
        if not set(self.layer_types) <= {"linear_attention", "full_attention"}:
            raise ValueError("Decision 2.0 backbone has an unsupported layer type")

    def forward(self, input_ids, candidate_positions, query_positions):
        backbone = self.model.backbone
        hidden = backbone.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32)
        frequencies = positions[:, None] * backbone.rotary_emb.inv_freq[None, :].float()
        embedding = torch.cat([frequencies, frequencies], dim=-1)[None]
        cos, sin = embedding.cos(), embedding.sin()
        for layer, kind in zip(backbone.layers, self.layer_types, strict=True):
            normed = layer.input_layernorm(hidden)
            if kind == "linear_attention":
                hidden = hidden + _linear_attention(layer.linear_attn, normed)
            else:
                hidden = hidden + _full_attention(
                    layer.self_attn, normed, cos, sin, self.rotate, self.gated
                )
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = backbone.norm(hidden)
        return _score(self.model.head, hidden, candidate_positions, query_positions)


def collate(
    encoded: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad encoded rows; causal layers keep padded positions out of real logits.

    A padded option repeats the last option endpoint. Mask its logit in a loss.
    """
    length = max(len(item["ids"]) for item in encoded)
    width = max(len(item["candidate_positions"]) for item in encoded)
    ids = torch.zeros((len(encoded), length), dtype=torch.int64)
    candidates = torch.zeros((len(encoded), width), dtype=torch.int64)
    for index, item in enumerate(encoded):
        ids[index, : len(item["ids"])] = torch.tensor(item["ids"])
        positions = item["candidate_positions"]
        candidates[index] = torch.tensor(
            positions + [positions[-1]] * (width - len(positions))
        )
    queries = torch.tensor([item["query_position"] for item in encoded])
    return ids, candidates, queries


def make_batch(
    rows: Sequence[Mapping[str, Any]], tokenizer: Tokenizer
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode prompt rows and right-pad them into one batch."""
    return collate([encode(row, tokenizer) for row in rows])


def save_checkpoint(
    model: Decision2Model, source: str | Path, destination: str | Path
) -> Path:
    """Save full FP32 weights with the settings and tokenizer of ``source``."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True)
    model.backbone.save_pretrained(destination / "backbone")
    save_file(
        {
            name: tensor.detach().contiguous()
            for name, tensor in model.head.state_dict().items()
        },
        str(destination / "decision_head.safetensors"),
    )
    for name in ("decision_config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(source / name, destination / name)
    return destination


def export_graph(
    model: Decision2Model, tokenizer: Tokenizer, output: str | Path
) -> Path:
    """Export one dynamic FP32 graph with its weights in ``model.onnx.data``."""
    graph = DecisionGraph(model).eval()
    arguments = make_batch(PROBES[:1], tokenizer)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".decision2-graph-", dir=output.parent
    ) as temporary:
        exported = Path(temporary) / output.name
        with torch.no_grad():
            torch.onnx.export(
                graph,
                arguments,
                str(exported),
                input_names=list(INPUT_NAMES),
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "candidate_positions": {0: "batch", 1: "options"},
                    "query_positions": {0: "batch"},
                    "logits": {0: "batch", 1: "options"},
                },
                opset_version=17,
                custom_opsets={"com.microsoft": 1},
                dynamo=False,
                external_data=True,
            )
        onnx.save_model(
            onnx.load(str(exported)),
            str(output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{output.name}.data",
        )
    onnx.checker.check_model(str(output))
    return output


def _softmax(values: np.ndarray) -> np.ndarray:
    scores = np.exp(values - values.max())
    return scores / scores.sum()


def validate_graph(
    model: Decision2Model, tokenizer: Tokenizer, output: str | Path
) -> dict[str, float | int]:
    """Compare ONNX logits with the PyTorch reference, one row and one padded batch."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    runtime = ort.InferenceSession(
        str(output), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if (
        {entry.name for entry in runtime.get_inputs()} != set(INPUT_NAMES)
        or [entry.name for entry in runtime.get_outputs()] != ["logits"]
        or runtime.get_outputs()[0].type != "tensor(float)"
    ):
        raise ValueError("Decision 2.0 graph inputs or FP32 output do not match")
    largest_logit, largest_probability = 0.0, 0.0
    batches = [[row] for row in PROBES] + [list(PROBES)]
    for rows in batches:
        arguments = make_batch(rows, tokenizer)
        with torch.inference_mode():
            expected = model(*arguments).numpy()
        actual = runtime.run(
            None,
            {
                name: value.numpy()
                for name, value in zip(INPUT_NAMES, arguments, strict=True)
            },
        )[0]
        for index, row in enumerate(rows):
            count = len(row["options"])
            want, got = expected[index, :count], actual[index, :count]
            if not np.isfinite(got).all() or not np.allclose(
                want, got, rtol=1e-4, atol=1e-3
            ):
                raise ValueError("Decision 2.0 FP32 export disagrees with checkpoint")
            if int(want.argmax()) != int(got.argmax()):
                raise ValueError("Decision 2.0 export changes the selected option")
            largest_logit = max(largest_logit, float(np.abs(want - got).max()))
            largest_probability = max(
                largest_probability,
                float(np.abs(_softmax(want) - _softmax(got)).max()),
            )
    return {
        "examples": len(PROBES),
        "max_abs_logit_error": largest_logit,
        "max_abs_probability_error": largest_probability,
    }


def _score_offsets(value: Any) -> bool:
    """Accept Score offsets that map each level count L in 2..255 to L numbers."""
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(key, str)
            and key.isascii()
            and key.isdigit()
            and str(int(key)) == key
            and 2 <= int(key) <= 255
            and isinstance(row, list)
            and len(row) == int(key)
            and all(type(item) in (int, float) and math.isfinite(item) for item in row)
            for key, row in value.items()
        )
    )


def _score_bias(source: Path) -> dict[str, list[float]] | None:
    """Return upstream Score offsets only with the unchanged weights that they fit."""
    path = source / "MODEL_MANIFEST.json"
    manifest = json.loads(path.read_text("utf-8")) if path.is_file() else {}
    entry = manifest.get("score_bias")
    if entry is None:
        return None
    identity = manifest.get("identity", {})
    expected = identity.get("fingerprint_files", {})
    files = {
        name: _sha256(source / name)
        for name in expected
        if "/" not in name and (source / name).is_file()
    }
    files.update(
        (item.relative_to(source).as_posix(), _sha256(item))
        for item in (source / "backbone").rglob("*")
        if item.is_file() and item.suffix in _FINGERPRINT_SUFFIXES
    )
    canonical = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path = source / entry.get("file", "")
    bias = (
        json.loads(path.read_text("utf-8"))
        if path.is_file() and _sha256(path) == entry.get("sha256")
        else {}
    )
    if (
        files != expected
        or not _SCORED_FILES <= files.keys()
        or digest != identity.get("model_sha256")
        or bias.get("format") != _SCORE_BIAS_FORMAT
        or bias.get("model_sha256") != digest
        or bias.get("offsets") != entry.get("offsets")
        or not _score_offsets(bias["offsets"])
    ):
        raise ValueError("Score offsets do not match the unchanged checkpoint weights")
    return {key: [float(item) for item in row] for key, row in bias["offsets"].items()}


def prepare_fp32(
    source: str | Path,
    destination: str | Path,
    *,
    license_file: str | Path,
    revision: str,
    model_id: str | None = None,
) -> dict[str, float | int]:
    """Atomically prepare an offline, inventoried bundle from a local checkpoint.

    The bundle keeps upstream Score offsets only when the manifest fingerprint
    proves that the checkpoint has the unchanged weights that the offsets fit.
    """
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    license_file = Path(license_file).expanduser().resolve()
    if not license_file.is_file():
        raise FileNotFoundError("An Apache-2.0 license file is required")
    checkpoint = _checkpoint_model(source)
    if model_id not in {None, checkpoint}:
        raise ValueError(f"Checkpoint is not a {model_id} model")
    score_bias = _score_bias(source)
    model, tokenizer = load_checkpoint(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".decision2-export-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "inference"
        staging.mkdir()
        export_graph(model, tokenizer, staging / "model.onnx")
        report = validate_graph(model, tokenizer, staging / "model.onnx")
        shutil.copyfile(
            source / "decision_config.json", staging / "decision_config.json"
        )
        (staging / "tokenizer").mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(source / name, staging / "tokenizer" / name)
        shutil.copyfile(license_file, staging / "LICENSE.txt")
        metadata = {
            "schema_version": 1,
            "kind": "inference",
            "format": "onnx",
            "precision": "fp32",
            "family": FAMILY,
            "architecture": DECISION2[checkpoint],
            "prompt_version": PROMPT_VERSION,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "model": checkpoint,
            "revision": revision,
            "license": "Apache-2.0",
            "files": _inventory(staging),
        }
        if score_bias is not None:
            metadata["score_bias"] = score_bias
        (staging / "PROVENANCE.json").write_text(json.dumps(metadata, indent=2) + "\n")
        staging.rename(destination)
    return report
