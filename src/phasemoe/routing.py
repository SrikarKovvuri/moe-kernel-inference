"""Explicit routing contracts and deterministic synthetic workloads."""

from dataclasses import dataclass

import torch

from .config import MoEConfig


@dataclass
class Routing:
    expert_ids: torch.Tensor  # [tokens, top_k], unique within a token
    weights: torch.Tensor  # [tokens, top_k], nonnegative; not necessarily sum=1
    num_experts: int

    def validate(self):
        if self.expert_ids.ndim != 2 or self.weights.shape != self.expert_ids.shape:
            raise ValueError("routing ids and weights must both have shape [tokens, top_k]")
        if self.expert_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("expert_ids must be int32 or int64")
        if not self.weights.is_floating_point():
            raise ValueError("routing weights must be floating point")
        if self.weights.device != self.expert_ids.device:
            raise ValueError("routing ids and weights must share a device")
        if self.num_experts < 1 or not 1 <= self.expert_ids.shape[1] <= self.num_experts:
            raise ValueError("invalid expert count or top_k")
        if self.expert_ids.numel():
            if self.expert_ids.min().item() < 0 or self.expert_ids.max().item() >= self.num_experts:
                raise ValueError("expert id out of range")
            ordered = self.expert_ids.sort(dim=1).values
            if (ordered[:, 1:] == ordered[:, :-1]).any().item():
                raise ValueError("a token cannot select an expert more than once")
        if not torch.isfinite(self.weights).all().item() or (self.weights < 0).any().item():
            raise ValueError("routing weights must be finite and nonnegative")
        return self

    @property
    def num_tokens(self):
        return self.expert_ids.shape[0]

    @property
    def top_k(self):
        return self.expert_ids.shape[1]

    def to(self, device):
        return Routing(self.expert_ids.to(device), self.weights.to(device), self.num_experts)


@dataclass
class PackedRouting:
    assignment_ids: torch.Tensor  # packed row -> original token * top_k + slot
    inverse: torch.Tensor  # original assignment -> packed row
    counts: torch.Tensor  # [experts]
    offsets: torch.Tensor  # [experts + 1], exclusive prefix sum


def pack_routes(routing: Routing) -> PackedRouting:
    """Device-side metadata preparation; included in full-block benchmarks.

    Validate public inputs before timing. No CPU copies or value-dependent Python
    branches occur here; stable ordering makes packing deterministic.
    """
    ids = routing.expert_ids.reshape(-1)
    order = torch.argsort(ids, stable=True)
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(order.numel(), device=order.device))
    # scatter_add avoids bincount's dynamic output-size synchronization on CUDA.
    counts = torch.zeros(routing.num_experts, dtype=torch.int64, device=ids.device)
    counts.scatter_add_(0, ids.long(), torch.ones_like(ids, dtype=torch.int64))
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return PackedRouting(order, inverse, counts, offsets)


def synthetic_routing(config: MoEConfig, num_tokens: int, distribution="uniform", seed=0):
    """Generate routes on CPU so a seed is portable across CPU/CUDA runners.

    Uniform = iid normal logits. Skewed = normal logits with a fixed expert bias.
    Balanced = cyclic unique assignments (control, not a learned top-k router).
    All weights come from a full-expert FP32 softmax and optional top-k renormalization.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens cannot be negative")
    if distribution not in ("uniform", "balanced", "skewed"):
        raise ValueError(f"unknown routing distribution: {distribution}")
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_tokens, config.num_experts, generator=generator)
    if distribution == "skewed":
        logits -= torch.arange(config.num_experts).float() * (8.0 / config.num_experts)
    probs = logits.softmax(dim=-1)
    if distribution == "balanced":
        ids = torch.arange(num_tokens * config.top_k).reshape(num_tokens, config.top_k)
        ids = ids.remainder(config.num_experts)
        weights = probs.gather(1, ids)
    else:
        weights, ids = probs.topk(config.top_k, dim=-1)
    if config.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return Routing(ids.contiguous(), weights.contiguous(), config.num_experts).validate()


def occupancy(routing: Routing):
    counts = torch.bincount(routing.expert_ids.flatten().cpu(), minlength=routing.num_experts)
    assignments = int(counts.sum())
    return {
        "counts": counts.tolist(),
        "histogram": torch.bincount(counts).tolist(),
        "assignments": assignments,
        "empty_experts": int((counts == 0).sum()),
        "singleton_experts": int((counts == 1).sum()),
        "active_experts": int((counts > 0).sum()),
        "max_occupancy": int(counts.max()),
    }
