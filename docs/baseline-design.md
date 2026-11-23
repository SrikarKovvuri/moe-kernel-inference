# Grouped baseline design

## Purpose

A fixed, inspectable baseline for correctness, latency and expert-occupancy studies.
It is extracted from the repository's existing grouped path. It is not a proposed
research contribution, an optimal design, or a replacement for a production baseline.

## Data and numerical contract

Inputs are X[T,H]; gate/up weights are [E,H,I]; down weights are [E,I,H].
Routes contain unique expert IDs and nonnegative weights per token, [T,K].
Default OLMoE dimensions are E=64, K=8, H=2048, I=1024. Synthetic selected
probabilities are FP32 and unnormalized after top-k by default.

The PyTorch reference performs independent GEMMs per active expert. A separate
FP32 token/slot oracle checks indexing, weight orientation and reduction on small shapes.
BF16/FP16 projection results and activated values are stored in the input dtype;
activation arithmetic, dot accumulation and weighted output reduction use FP32.
The final output is cast to the input dtype. This is not bitwise upstream-model parity.

## Fixed GPU execution

1. Stable sort packs token/expert assignments; counts and prefix sums delimit rows.
2. Grouped gate/up GEMMs and SiLU multiplication produce all hidden rows.
3. Grouped down GEMM produces and stores all projected rows.
4. A separate weighted reduction reads those rows and writes one output per token.

The gate/up stage already shares a launch in this inherited baseline. That detail
is explicitly part of the comparison baseline, not evidence that the team's
planned optimization is complete. Down projection and output reduction are separate.

GEMM tiles use fixed M=16, N=64, K=32, four warps; output reduction uses N=32.
The device computes tile ranges from row counts. The launch uses a conservative
upper bound and excess programs exit. Counts only delimit normal grouped work;
there is no expert-dependent algorithm selection. All active experts follow the same path.
Zero-token batches bypass GPU launches; row/column tails are masked.

## Measurement boundaries

Validation and JIT compilation precede warmup and timing. Full expert-block timings
include packing, allocations, projections, activation and reduction. Router selection,
attention, normalization and checkpoint loading are excluded. Stage profiling uses
precomputed inputs; stage sums do not estimate block latency. Profiler trace durations
are diagnostic and are not substituted for repeated latency samples.

## Research remains open

The team will independently evaluate candidate layouts, kernel boundaries, wider or
narrower fusion, execution strategies, reuse opportunities and occupancy-aware
scheduling. No singleton threshold or down/combine design is prescribed. Occupancy
statistics identify workload variation; they do not prove which design will be faster.
Use target-GPU profiling and strong external baselines before choosing an approach.
