"""Fixed grouped-GEMM baseline extracted from the existing grouped execution path.

Every active expert uses grouped projections. Counts delimit rows and GEMM tiles;
they never select an alternative computation. All down-projection rows are stored.
"""

from dataclasses import dataclass

import torch

from .reference import ExpertWeights
from .routing import Routing, pack_routes

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


@dataclass(frozen=True)
class GroupedConfig:
    block_m: int = 16
    block_n: int = 64
    block_k: int = 32
    combine_n: int = 32
    num_warps: int = 4


CONFIG = GroupedConfig()


if triton is not None:

    @triton.jit
    def _expert_tile(TileEnds, Offsets, E: tl.constexpr, PAD_E: tl.constexpr):
        tile = tl.program_id(0)
        es = tl.arange(0, PAD_E)
        ends = tl.load(TileEnds + es, es < E, other=2147483647)
        expert = tl.sum((tile >= ends).to(tl.int32), axis=0)
        previous = tl.load(TileEnds + expert - 1, (expert > 0) & (expert < E), other=0)
        start = tl.load(Offsets + expert, expert < E, other=0)
        stop = tl.load(Offsets + expert + 1, expert < E, other=0)
        return expert, tile - previous, start, stop

    @triton.jit
    def _gate_up(
        X,
        Gate,
        Up,
        Assignment,
        Offsets,
        TileEnds,
        Hidden,
        H: tl.constexpr,
        I: tl.constexpr,
        E: tl.constexpr,
        TOP_K: tl.constexpr,
        PAD_E: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        expert, tile_m, start, stop = _expert_tile(TileEnds, Offsets, E, PAD_E)
        if expert < E:
            rows = start + tile_m * BM + tl.arange(0, BM)
            ns = tl.program_id(1) * BN + tl.arange(0, BN)
            ks = tl.arange(0, BK)
            assignment = tl.load(Assignment + rows, rows < stop, other=0)
            tokens = assignment // TOP_K
            acc_g = tl.full((BM, BN), 0, tl.float32)
            acc_u = tl.full((BM, BN), 0, tl.float32)
            for block in range(tl.cdiv(H, BK)):
                kk = block * BK + ks
                xv = tl.load(
                    X + tokens[:, None] * H + kk[None, :],
                    (rows[:, None] < stop) & (kk[None, :] < H),
                    other=0,
                )
                pos = expert * H * I + kk[:, None] * I + ns[None, :]
                mask = (kk[:, None] < H) & (ns[None, :] < I)
                gv = tl.load(Gate + pos, mask, other=0)
                uv = tl.load(Up + pos, mask, other=0)
                acc_g += tl.dot(xv, gv, input_precision="ieee")
                acc_u += tl.dot(xv, uv, input_precision="ieee")
            g = acc_g.to(X.dtype.element_ty).to(tl.float32)
            u = acc_u.to(X.dtype.element_ty).to(tl.float32)
            tl.store(
                Hidden + rows[:, None] * I + ns[None, :],
                (g * tl.sigmoid(g)) * u,
                (rows[:, None] < stop) & (ns[None, :] < I),
            )

    @triton.jit
    def _down(
        Hidden,
        Down,
        Offsets,
        TileEnds,
        Projected,
        H: tl.constexpr,
        I: tl.constexpr,
        E: tl.constexpr,
        PAD_E: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        expert, tile_m, start, stop = _expert_tile(TileEnds, Offsets, E, PAD_E)
        if expert < E:
            rows = start + tile_m * BM + tl.arange(0, BM)
            ns = tl.program_id(1) * BN + tl.arange(0, BN)
            ks = tl.arange(0, BK)
            acc = tl.full((BM, BN), 0, tl.float32)
            for block in range(tl.cdiv(I, BK)):
                kk = block * BK + ks
                av = tl.load(
                    Hidden + rows[:, None] * I + kk[None, :],
                    (rows[:, None] < stop) & (kk[None, :] < I),
                    other=0,
                )
                bv = tl.load(
                    Down + expert * I * H + kk[:, None] * H + ns[None, :],
                    (kk[:, None] < I) & (ns[None, :] < H),
                    other=0,
                )
                acc += tl.dot(av, bv, input_precision="ieee")
            tl.store(
                Projected + rows[:, None] * H + ns[None, :],
                acc,
                (rows[:, None] < stop) & (ns[None, :] < H),
            )

    @triton.jit
    def _reduce(
        Projected,
        RouteWeights,
        Inverse,
        Out,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        BN: tl.constexpr,
    ):
        token = tl.program_id(0)
        ns = tl.program_id(1) * BN + tl.arange(0, BN)
        result = tl.full((BN,), 0, tl.float32)
        for slot in range(TOP_K):
            assignment = token * TOP_K + slot
            row = tl.load(Inverse + assignment)
            weight = tl.load(RouteWeights + assignment).to(tl.float32)
            value = tl.load(Projected + row * H + ns, ns < H, other=0).to(tl.float32)
            result += value * weight
        tl.store(Out + token * H + ns, result, ns < H)


def prepare(routing):
    packed = pack_routes(routing)
    ends = ((packed.counts + CONFIG.block_m - 1) // CONFIG.block_m).cumsum(0)
    return packed, ends


def project_gate_up(x, weights, routing, packed, ends):
    h, i = weights.gate.shape[1:]
    assignments = routing.expert_ids.numel()
    hidden = torch.empty((assignments, i), device=x.device, dtype=x.dtype)
    capacity = triton.cdiv(assignments, CONFIG.block_m) + routing.num_experts - 1
    _gate_up[(capacity, triton.cdiv(i, CONFIG.block_n))](
        x,
        weights.gate,
        weights.up,
        packed.assignment_ids,
        packed.offsets,
        ends,
        hidden,
        h,
        i,
        routing.num_experts,
        routing.top_k,
        triton.next_power_of_2(routing.num_experts),
        CONFIG.block_m,
        CONFIG.block_n,
        CONFIG.block_k,
        num_warps=CONFIG.num_warps,
    )
    return hidden


def project_down(hidden, weights, routing, packed, ends):
    e, i, h = weights.down.shape
    projected = torch.empty((hidden.shape[0], h), device=hidden.device, dtype=hidden.dtype)
    capacity = triton.cdiv(hidden.shape[0], CONFIG.block_m) + e - 1
    _down[(capacity, triton.cdiv(h, CONFIG.block_n))](
        hidden,
        weights.down,
        packed.offsets,
        ends,
        projected,
        h,
        i,
        e,
        triton.next_power_of_2(e),
        CONFIG.block_m,
        CONFIG.block_n,
        CONFIG.block_k,
        num_warps=CONFIG.num_warps,
    )
    return projected


def reduce_output(projected, routing, packed):
    h = projected.shape[1]
    out = torch.empty((routing.num_tokens, h), device=projected.device, dtype=projected.dtype)
    _reduce[(routing.num_tokens, triton.cdiv(h, CONFIG.combine_n))](
        projected,
        routing.weights,
        packed.inverse,
        out,
        h,
        routing.top_k,
        CONFIG.combine_n,
        num_warps=CONFIG.num_warps,
    )
    return out


@torch.inference_mode()
def moe_grouped(x, weights: ExpertWeights, routing: Routing):
    """Baseline including dispatch; validate shapes/route values before timing."""
    if triton is None:
        raise RuntimeError("Install the GPU extra to run the grouped baseline")
    if x.device.type != "cuda":
        raise ValueError("grouped baseline requires CUDA tensors")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("grouped baseline supports BF16, FP16, or FP32")
    tensors = (x, *weights.tensors(), routing.expert_ids, routing.weights)
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("grouped baseline requires contiguous tensors")
    if routing.num_tokens == 0:
        return torch.empty_like(x)
    packed, ends = prepare(routing)
    hidden = project_gate_up(x, weights, routing, packed, ends)
    projected = project_down(hidden, weights, routing, packed, ends)
    return reduce_output(projected, routing, packed)
