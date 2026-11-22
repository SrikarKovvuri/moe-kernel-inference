"""Portable routing traces with explicit provenance and strict validation."""

import json
from pathlib import Path

import torch

from .benchmark import write_json
from .routing import Routing, occupancy


def routing_from_logits(logits, *, top_k, norm_topk_prob=False):
    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("router logits must be floating point [tokens, experts]")
    if not 1 <= top_k <= logits.shape[1]:
        raise ValueError("top_k must be in [1, num_experts]")
    if not torch.isfinite(logits).all().item():
        raise ValueError("router logits must be finite")
    probabilities = logits.float().softmax(-1)
    weights, ids = probabilities.topk(top_k, dim=-1)
    if norm_topk_prob:
        weights = weights / weights.sum(-1, keepdim=True)
    return Routing(ids.contiguous(), weights.contiguous(), logits.shape[1]).validate()


def save_trace(path, routing, *, provenance):
    routing.validate()
    required = ("source", "model", "model_revision", "layer", "phase", "norm_topk_prob")
    if any(key not in provenance for key in required):
        raise ValueError(f"trace provenance must contain {required}")
    if provenance["source"] not in ("pretrained_model", "synthetic", "external"):
        raise ValueError("trace source must identify pretrained_model, synthetic, or external")
    data = {
        "schema_version": 1,
        "kind": "routing_trace",
        "provenance": provenance,
        "num_experts": routing.num_experts,
        "num_tokens": routing.num_tokens,
        "top_k": routing.top_k,
        "expert_ids": routing.expert_ids.cpu().tolist(),
        "weights": routing.weights.float().cpu().tolist(),
        "occupancy": occupancy(routing),
    }
    write_json(path, data)


def load_trace(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("kind") != "routing_trace":
        raise ValueError("unsupported routing trace format")
    tokens, k = data["num_tokens"], data["top_k"]
    if not isinstance(tokens, int) or tokens < 0 or not isinstance(k, int) or k < 1:
        raise ValueError("invalid trace dimensions")
    # Reject fractional IDs before converting; torch.tensor(dtype=int64) would truncate them.
    ids = torch.tensor(data["expert_ids"])
    if tokens and ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("trace expert IDs must be integers")
    ids = ids.to(dtype=torch.int64).reshape(tokens, k)
    weights = torch.tensor(data["weights"], dtype=torch.float32).reshape(tokens, k)
    routing = Routing(ids, weights, data["num_experts"]).validate()
    return routing, data["provenance"]
