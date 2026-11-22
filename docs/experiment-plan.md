# Experiment plan

## Claim under test

Occupancy-aware fusion of expert down projection and output aggregation may reduce
single-layer latency at low token counts. A slowdown is a valid result. Compare
against the same grouped projection implementation with only the down/combine
strategy changed, and separately report the simple PyTorch baseline.

## Milestones

1. Tested routing semantics, independent reference, and PyTorch expert baseline.
2. Reproducible CPU/GPU benchmark output with device, dtype, software, dimensions,
   routing distribution/seed, warmup, repetitions, raw samples, and source revision.
3. Grouped Triton projection baseline and BF16 correctness tests.
4. Singleton fused down/combine kernel, then hybrid integration and threshold sweep.
5. Actual development measurements, figures, profiling, and a conservative midpoint draft.
6. L40S/A6000 rerun and real model routing traces when those resources are available.

## Measurement protocol

- Synthetic token counts: 1, 8, 32, 128, 512; distributions: uniform, balanced, skewed.
- Default target configuration: OLMoE dimensions, BF16, thresholds 0, 1, 2, 4.
- Use identical inputs, routes, and weights for all backends in a case.
- Validate outside the timed region, using explicit absolute/relative tolerances.
- Compile and warm up before measurement. Record median, p10/p90, and raw samples.
- CUDA events measure stream elapsed time; synchronized wall time measures Python
  launch/dispatch overhead as well. Do not equate isolated kernel sums with full latency.
- Record whether routing preparation and allocations are included and whether timing
  uses eager launches or CUDA graphs. Never compare different timing scopes as speedups.
- Synthetic routes and random weights do not establish model quality or real routing statistics.
- Keep full-scale GPU validation separate from small-shape CPU logic checks.

## Midpoint requirements still needing team input

UNIs, confirmed roles, experiment-dashboard URL, and confirmation that the approved
three-person team satisfies the course's stated four-to-five-person template.
Do not invent these fields or claim cluster access that has not been exercised.
