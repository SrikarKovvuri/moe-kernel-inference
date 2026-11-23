"""Transparent math oracle and an expert-batched PyTorch baseline.

The latter launches separate PyTorch GEMMs per active expert: it is deliberately
NOT labeled grouped-GEMM. Both implementations accumulate weighted outputs in FP32.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import MoEConfig
from .routing import Routing


@dataclass
class ExpertWeights:
    gate: torch.Tensor  # [experts, hidden, intermediate]
    up: torch.Tensor  # [experts, hidden, intermediate]
    down: torch.Tensor  # [experts, intermediate, hidden]

    @classmethod
    def random(cls, config: MoEConfig, *, seed=0, dtype=torch.float32, device="cpu"):
        # Initialize on CPU for reproducible data across devices. Each projection
        # is transferred independently to avoid a second full GPU weight copy.
        gen = torch.Generator().manual_seed(seed)
        e, h, i = config.num_experts, config.hidden_size, config.intermediate_size

        def sample(shape, fan_in):
            return (torch.randn(shape, generator=gen) / fan_in**0.5).to(device=device, dtype=dtype)

        return cls(sample((e, h, i), h), sample((e, h, i), h), sample((e, i, h), i))

    def to(self, *, device=None, dtype=None):
        return ExpertWeights(*(w.to(device=device, dtype=dtype) for w in self.tensors()))

    def tensors(self):
        return self.gate, self.up, self.down


def validate_inputs(x, weights: ExpertWeights, routing: Routing):
    routing.validate()
    if x.ndim != 2 or x.shape[0] != routing.num_tokens or not x.is_floating_point():
        raise ValueError("x must be floating point with shape [tokens, hidden]")
    if weights.gate.ndim != 3:
        raise ValueError("gate weights must have shape [experts, hidden, intermediate]")
    e, h, i = weights.gate.shape
    if e != routing.num_experts or h != x.shape[1]:
        raise ValueError("input, routing, and expert weight shapes disagree")
    if weights.up.shape != (e, h, i) or weights.down.shape != (e, i, h):
        raise ValueError("invalid up/down weight shape")
    if routing.weights.device != x.device:
        raise ValueError("input and routing must share a device")
    if any(w.dtype != x.dtype or w.device != x.device for w in weights.tensors()):
        raise ValueError("input and all expert weights must share dtype and device")


@torch.inference_mode()
def moe_oracle(x, weights: ExpertWeights, routing: Routing):
    """Independent token/slot loop in FP32; for correctness, never timing."""
    validate_inputs(x, weights, routing)
    # Explicit .float() inputs alone do not prevent autocast from lowering GEMMs.
    with torch.autocast(device_type=x.device.type, enabled=False):
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for token in range(routing.num_tokens):
            for slot in range(routing.top_k):
                expert = int(routing.expert_ids[token, slot])
                xi = x[token].float()
                gate = xi @ weights.gate[expert].float()
                up = xi @ weights.up[expert].float()
                value = (F.silu(gate) * up) @ weights.down[expert].float()
                out[token] += routing.weights[token, slot].float() * value
    return out


@torch.inference_mode()
def moe_torch(x, weights: ExpertWeights, routing: Routing):
    """Expert-wise GEMMs including dispatch, activation, and combine.

    BF16 projections are rounded to BF16 at each projection boundary. Activation
    multiplication and accumulation use FP32; activated values are stored in x.dtype.
    Data-dependent nonzero and Python expert loops are part of this simple baseline.
    Ambient autocast must not override this explicit dtype policy.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for expert in range(routing.num_experts):
            tokens, slots = torch.where(routing.expert_ids == expert)
            if tokens.numel() == 0:
                continue
            xi = x[tokens]
            gate = (xi @ weights.gate[expert]).float()
            up = (xi @ weights.up[expert]).float()
            hidden = (F.silu(gate) * up).to(x.dtype)
            values = (hidden @ weights.down[expert]).float()
            values *= routing.weights[tokens, slots, None].float()
            out.index_add_(0, tokens, values)
    return out.to(x.dtype)


def error_metrics(actual, expected, *, atol, rtol):
    if actual.shape != expected.shape:
        raise ValueError("correctness comparison shapes must match")
    a, b = actual.float(), expected.float()
    error = (a - b).abs()
    allowed = atol + rtol * b.abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {
        "passed": finite and bool((error <= allowed).all()),
        "finite": finite,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": float(error.max()) if error.numel() else 0.0,
        "rmse": float(error.square().mean().sqrt()) if error.numel() else 0.0,
        "relative_l2_error": float(
            torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12)
        ),
        "fraction_outside_tolerance": float((error > allowed).float().mean())
        if error.numel()
        else 0.0,
    }
