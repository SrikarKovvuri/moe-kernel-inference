"""Native PyTorch grouped-mm API baseline with explicit architecture caveats.

PyTorch's native implementation may fall back to a host loop of per-expert GEMMs
on some architectures/dtypes. Calling the API does not prove a single grouped GPU
kernel executed. Inspect a profiler trace before interpreting this as a fast backend.
"""

import torch
import torch.nn.functional as F

from .routing import pack_routes


def native_grouped_mm():
    if hasattr(F, "grouped_mm"):
        return F.grouped_mm
    if hasattr(torch, "_grouped_mm"):
        return torch._grouped_mm
    raise RuntimeError("native grouped-mm requires a PyTorch build with grouped_mm support")


@torch.inference_mode()
def moe_torch_grouped(x, weights, routing):
    op = native_grouped_mm()
    if routing.num_tokens == 0:
        return torch.empty_like(x)
    alignment = 16 // x.element_size()
    if weights.gate.shape[1] % alignment or weights.gate.shape[2] % alignment:
        raise ValueError(f"native grouped-mm requires hidden/intermediate multiples of {alignment}")
    packed = pack_routes(routing)
    dispatched = x[packed.assignment_ids // routing.top_k].contiguous()
    ends = packed.offsets[1:].to(torch.int32)
    gate = op(dispatched, weights.gate, offs=ends).float()
    up = op(dispatched, weights.up, offs=ends).float()
    hidden = (F.silu(gate) * up).to(x.dtype)
    projected = op(hidden, weights.down, offs=ends)
    # Inverse permutation restores token/slot order; FP32 sum matches our contract.
    restored = projected[packed.inverse].reshape(routing.num_tokens, routing.top_k, x.shape[1])
    return (restored.float() * routing.weights.float().unsqueeze(-1)).sum(1).to(x.dtype)
