"""Two-launch stable dispatch for small assignment batches (no host count reads).

Each expert scans all assignment IDs, then uses a prefix sum to scatter its rows.
Work is O(experts * assignments), so this is an explicitly bounded decode-oriented
alternative to sorting, not an asymptotically better general-purpose dispatcher.
"""

import torch

from .routing import PackedRouting

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _pack(
        Ids,
        Assignment,
        Inverse,
        Counts,
        Offsets,
        A: tl.constexpr,
        E: tl.constexpr,
        BLOCK_A: tl.constexpr,
    ):
        expert = tl.program_id(0)
        assignments = tl.arange(0, BLOCK_A)
        ids = tl.load(Ids + assignments, assignments < A, other=E)
        belongs = (ids == expert) & (assignments < A)
        count = tl.sum(belongs.to(tl.int32), axis=0)
        start = tl.sum((ids < expert).to(tl.int32), axis=0)
        rank = tl.cumsum(belongs.to(tl.int32), axis=0) - 1
        rows = start + rank
        tl.store(Assignment + rows, assignments.to(tl.int64), belongs)
        tl.store(Inverse + assignments, rows.to(tl.int64), belongs)
        tl.store(Counts + expert, count.to(tl.int64))
        tl.store(Offsets + expert, start.to(tl.int64))
        if expert == E - 1:
            tl.store(Offsets + E, A)

    @triton.jit
    def _tiles(Counts, Ends, E: tl.constexpr, PAD_E: tl.constexpr, BM: tl.constexpr):
        experts = tl.arange(0, PAD_E)
        counts = tl.load(Counts + experts, experts < E, other=0)
        ends = tl.cumsum((counts + BM - 1) // BM, axis=0)
        tl.store(Ends + experts, ends, experts < E)


def pack_routes_triton(routing, *, block_m=16):
    if triton is None or routing.expert_ids.device.type != "cuda":
        raise ValueError("Triton dispatch requires CUDA and Triton")
    if not routing.expert_ids.is_contiguous():
        raise ValueError("routing IDs must be contiguous")
    a, e = routing.expert_ids.numel(), routing.num_experts
    if a < 1 or a > 4096:
        raise ValueError("Triton scan dispatch supports 1..4096 assignments")
    if block_m < 1:
        raise ValueError("block_m must be positive")
    options = dict(dtype=torch.int64, device=routing.expert_ids.device)
    assignment = torch.empty(a, **options)
    inverse = torch.empty(a, **options)
    counts = torch.empty(e, **options)
    offsets = torch.empty(e + 1, **options)
    ends = torch.empty(e, **options)
    _pack[(e,)](
        routing.expert_ids,
        assignment,
        inverse,
        counts,
        offsets,
        a,
        e,
        triton.next_power_of_2(a),
        num_warps=4,
    )
    _tiles[(1,)](counts, ends, e, triton.next_power_of_2(e), block_m, num_warps=4)
    return PackedRouting(assignment, inverse, counts, offsets), ends
