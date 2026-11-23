# Midpoint status and second-half research plan

## Established at midpoint

- OLMoE dimensions, routing semantics and deterministic synthetic workloads.
- Independent mathematical reference and a working fixed grouped-GEMM baseline.
- Baseline correctness checks, eager latency measurements and basic stage profiling.
- Expert-occupancy counts, histograms and baseline visualizations.

The current research question is how uneven expert workloads affect GPU execution,
and whether a different execution design can reduce complete MoE block latency.
No optimization implementation or positive/negative optimization conclusion is included.

## Team-owned research after midpoint

1. Reproduce the baseline on the intended L40S/A6000 under stable power/clocks.
   Check wall/event agreement and inspect actual kernels before interpreting timings.
2. Establish at least one optimized external grouped-GEMM comparison with matched
   dtype, input/routes, numerical tolerance and timing boundaries.
3. Collect real model routes across layers, prompt lengths and concurrent decode
   sequences; distinguish actual generation from synthetic token-count tests.
4. Independently design candidates. Explore alternative layouts and execution
   strategies, fusion over different or more extensive stages, occupancy-aware
   scheduling, weight reuse, and resource tradeoffs. Select designs from evidence;
   do not assume any previous strategy was optimal or make a singleton policy mandatory.
5. Implement selected candidates, add their correctness checks, and evaluate total
   block latency including dispatch/scheduling. Add ablations only when needed to
   distinguish mechanisms. Measure memory traffic and GPU utilization with appropriate tools.
6. Validate model integration/quality where numerical or execution changes require it.
   Report limitations and negative findings as carefully as improvements.

## Baseline protocol

Use identical seeded inputs/weights/routes for the PyTorch and grouped baselines.
At full OLMoE shape, check BF16 against FP32 at atol=0.02, rtol=0.03; small-shape
FP32 tests also use the independent oracle. Report numerical errors, not just pass/fail.
Compile/check first, then 20 warmups and 100 eager samples. Keep raw wall and
CUDA-event measurements and source hashes. Record every shape, dtype, device,
routing distribution, seed and scope. Repeat independent runs before interpreting
small differences; the current fixed-order synthetic sweep is preliminary.

Synthetic balanced routing is an even-load control, not a learned router.
Uniform and skewed routes are controlled synthetic distributions, not corpus statistics.

## Administrative fields

Team UNIs, confirmed roles, an accessible dashboard URL and approval of the listed
three-person team against the four-to-five-person template remain unresolved.
The GitHub evidence directory is not a claim that an external dashboard exists.
