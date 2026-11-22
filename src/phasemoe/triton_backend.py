"""Grouped projections and occupancy-selective down/combine fusion.

This is a research baseline, not a production vLLM/MegaBlocks replacement. A single
launch groups experts using a device-built tile schedule. Padded tile capacity is
bounded by assignments / BLOCK_M + experts; unused programs exit before GEMM work.
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
class KernelConfig:
    block_m: int = 16
    block_n: int = 64
    block_k: int = 32
    combine_n: int = 32
    combine_k: int = 128
    num_warps: int = 4

    def __post_init__(self):
        for name in ("block_m", "block_n", "block_k", "combine_n", "combine_k"):
            value = getattr(self, name)
            if value < 16 or value & (value - 1):
                raise ValueError(f"{name} must be a power of two >= 16")
        if self.num_warps not in (4, 8):
            raise ValueError("num_warps must be 4 or 8")


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
            # Match the projection rounding boundaries of the PyTorch baseline.
            g = acc_g.to(X.dtype.element_ty).to(tl.float32)
            u = acc_u.to(X.dtype.element_ty).to(tl.float32)
            activated = (g * tl.sigmoid(g)) * u
            tl.store(
                Hidden + rows[:, None] * I + ns[None, :],
                activated,
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
        THRESHOLD: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        expert, tile_m, start, stop = _expert_tile(TileEnds, Offsets, E, PAD_E)
        if (expert < E) & (stop - start > THRESHOLD):
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
    def _combine(
        Hidden,
        Down,
        Projected,
        Ids,
        RouteWeights,
        Inverse,
        Counts,
        Out,
        H: tl.constexpr,
        I: tl.constexpr,
        TOP_K: tl.constexpr,
        THRESHOLD: tl.constexpr,
        ALL_FUSED: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        token = tl.program_id(0)
        ns = tl.program_id(1) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        result = tl.full((BN,), 0, tl.float32)
        for slot in range(TOP_K):
            assignment = token * TOP_K + slot
            row = tl.load(Inverse + assignment)
            route_weight = tl.load(RouteWeights + assignment).to(tl.float32)
            if THRESHOLD > 0 or ALL_FUSED:
                expert = tl.load(Ids + assignment)
                count = tl.load(Counts + expert)
                if ALL_FUSED or count <= THRESHOLD:
                    value = tl.full((BN,), 0, tl.float32)
                    for block in range(tl.cdiv(I, BK)):
                        kk = block * BK + ks
                        av = tl.load(Hidden + row * I + kk, kk < I, other=0).to(tl.float32)
                        bv = tl.load(
                            Down + expert * I * H + kk[:, None] * H + ns[None, :],
                            (kk[:, None] < I) & (ns[None, :] < H),
                            other=0,
                        ).to(tl.float32)
                        value += tl.sum(av[:, None] * bv, axis=0)
                    value = value.to(Out.dtype.element_ty).to(tl.float32)
                else:
                    value = tl.load(Projected + row * H + ns, ns < H, other=0).to(tl.float32)
            else:
                value = tl.load(Projected + row * H + ns, ns < H, other=0).to(tl.float32)
            result += value * route_weight
        tl.store(Out + token * H + ns, result, ns < H)


def require_triton(x):
    if triton is None:
        raise RuntimeError("Triton is unavailable; install the GPU extra in a CUDA environment")
    if x.device.type != "cuda":
        raise ValueError("Triton backend requires CUDA tensors")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Triton backend supports BF16, FP16, or FP32")


def prepare(routing: Routing, config: KernelConfig):
    packed = pack_routes(routing)
    tile_ends = ((packed.counts + config.block_m - 1) // config.block_m).cumsum(0)
    return packed, tile_ends


def project_gate_up(x, weights, routing, packed, tile_ends, config):
    h, i = weights.gate.shape[1:]
    assignments = routing.expert_ids.numel()
    hidden = torch.empty((assignments, i), device=x.device, dtype=x.dtype)
    capacity = triton.cdiv(assignments, config.block_m) + routing.num_experts - 1
    _gate_up[(capacity, triton.cdiv(i, config.block_n))](
        x,
        weights.gate,
        weights.up,
        packed.assignment_ids,
        packed.offsets,
        tile_ends,
        hidden,
        h,
        i,
        routing.num_experts,
        routing.top_k,
        triton.next_power_of_2(routing.num_experts),
        config.block_m,
        config.block_n,
        config.block_k,
        num_warps=config.num_warps,
    )
    return hidden


def project_down(hidden, weights, routing, packed, tile_ends, config, threshold):
    e, i, h = weights.down.shape
    projected = torch.empty((hidden.shape[0], h), device=hidden.device, dtype=hidden.dtype)
    capacity = triton.cdiv(hidden.shape[0], config.block_m) + e - 1
    _down[(capacity, triton.cdiv(h, config.block_n))](
        hidden,
        weights.down,
        packed.offsets,
        tile_ends,
        projected,
        h,
        i,
        e,
        triton.next_power_of_2(e),
        threshold,
        config.block_m,
        config.block_n,
        config.block_k,
        num_warps=config.num_warps,
    )
    return projected


def combine(hidden, weights, projected, routing, packed, config, threshold, all_fused=False):
    _, i, h = weights.down.shape
    out = torch.empty((routing.num_tokens, h), device=hidden.device, dtype=hidden.dtype)
    _combine[(routing.num_tokens, triton.cdiv(h, config.combine_n))](
        hidden,
        weights.down,
        projected,
        routing.expert_ids,
        routing.weights,
        packed.inverse,
        packed.counts,
        out,
        h,
        i,
        routing.top_k,
        threshold,
        all_fused,
        config.combine_n,
        config.combine_k,
        num_warps=config.num_warps,
    )
    return out


@torch.inference_mode()
def moe_triton(
    x,
    weights: ExpertWeights,
    routing: Routing,
    *,
    threshold=1,
    strategy="hybrid",
    config=None,
    prepared=None,
):
    """Inference on validated contiguous inputs; validation belongs outside timing.

    `prepared` is reserved for explicit prepacked microbenchmarks. The default
    recomputes metadata, so full-block timing includes occupancy/dispatch overhead.
    No value-dependent host synchronization is used in the GPU execution path.
    """
    require_triton(x)
    if strategy not in ("grouped", "hybrid", "fused"):
        raise ValueError(f"unknown strategy: {strategy}")
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    tensors = (x, *weights.tensors(), routing.expert_ids, routing.weights)
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("Triton backend requires contiguous input, weights, and routing")
    if routing.num_tokens == 0:
        return torch.empty_like(x)
    config = config or KernelConfig()
    packed, tile_ends = prepared if prepared is not None else prepare(routing, config)
    if strategy == "grouped":
        threshold = 0
    hidden = project_gate_up(x, weights, routing, packed, tile_ends, config)
    if strategy == "fused":
        # Unused pointer argument; the constexpr all-fused branch never loads it.
        projected = hidden
    else:
        projected = project_down(hidden, weights, routing, packed, tile_ends, config, threshold)
    return combine(
        hidden,
        weights,
        projected,
        routing,
        packed,
        config,
        threshold,
        all_fused=strategy == "fused",
    )
