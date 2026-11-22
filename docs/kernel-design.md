# Selective down-projection fusion

## Tensor contract

For T tokens, E experts, top-k K, hidden width H, and intermediate width I:

- Inputs: X[T,H], gate/up weights [E,H,I], down weights [E,I,H].
- Routing: expert IDs [T,K] and floating-point weights [T,K]. Synthetic routes use
  FP32 weights; pretrained block execution preserves the upstream BF16 cast.
  IDs are unique per token.
- Packed assignment rows: A=T*K, sorted by expert; each row retains its token/slot ID.
- Offsets [E+1] and counts [E] delimit experts. An inverse permutation maps each
  original assignment to its packed row.
- Gate/up kernel produces hidden activations [A,I].
- Grouped down kernel produces projected rows [A,H] for heavy experts only.
- Combine kernel owns one token/output-column tile and writes the final output once.

Every implementation computes the same weighted sum of SiLU-gated expert MLPs.
Selected top-k weights are not renormalized by default. The implementation uses
BF16 projection/activation storage and FP32 dot accumulation, activation arithmetic,
and output reduction. The FP32 oracle does not round intermediate projections.
This numerical policy is explicit and is not claimed to be bitwise identical to
Hugging Face's BF16 expert accumulation order.

## Device scheduling

The packed route builder uses sort, scatter, and prefix sums on the device. For
BLOCK_M rows per GEMM tile, the per-expert tile counts are ceil(count/BLOCK_M).
Their prefix sums give a compact logical tile schedule. Each program locates its
expert from those prefixes. The host launches an upper bound of
ceil(A/BLOCK_M)+E-1 tiles, so no GPU count needs to be copied back to the CPU.
Excess programs exit before matrix multiplication. This simple schedule is a
research baseline; persistent scheduling and metadata lookup costs remain tuning
opportunities.

An optional small-batch dispatcher scans assignment IDs once per expert. A prefix
sum places each expert's assignments in stable order, and a second launch computes
the GEMM tile ends. Tests require exact equality with sort/scatter metadata. This
has O(E × A) work and is deliberately restricted to 1–4096 assignments. It is a
separate optimization from down/combine fusion and is not enabled by default.

## Three strategies

1. **Grouped**: all active experts use grouped down GEMM; combine reads projected rows.
2. **Hybrid**: experts with count <= threshold skip down GEMM. During combination,
   their down projection is a chunked matrix-vector reduction using the hidden row
   and expert weights; heavy experts load their precomputed projected row.
3. **Uniformly fused**: skip down GEMM entirely and compute every projection in combine.

The gate/up/activation kernel is identical across strategies. It combines gate and
up projections in one launch and avoids their separate intermediate tensors.
The controlled comparison is the down/combine strategy, not this shared fusion.

## Correctness and memory safety

- Every output element has one writer, so output aggregation needs no atomics.
- The hybrid combine branch never reads skipped/uninitialized projected rows.
- Input/output tails and expert-row tails are masked independently.
- BF16 results are rounded at matching projection boundaries before weighting.
- Empty batches bypass all kernels. Zero-occupancy experts produce no work.
- The GPU suite covers FP32, FP16, BF16, non-power-of-two expert counts, unaligned
  dimensions, singleton-only, all-heavy, mixed, and zero-token workloads.

## Limits and hypothesis risks

Fusion avoids projected-row stores/loads for light experts, but does not avoid
reading their down weights. The hybrid prototype still reserves the full projected
buffer; **traffic savings do not imply allocation savings**. Direct matrix-vector
reductions may have worse reuse, instruction efficiency, or parallelism than tiled
GEMM. Combine serializes the selected expert slots within a token/output tile.
Sorting, tile preparation, empty programs, and the remaining grouped launch can
outweigh saved traffic. Full-block measurements must include these costs.

No claim is made that this grouped implementation matches the strongest available
production grouped-GEMM backend. External backend comparison and target-cluster
profiling remain required before a broader performance claim.
