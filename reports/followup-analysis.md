# Follow-up experiments

These experiments extend the initial midpoint evidence. All GPU measurements remain development results on a power/clock-constrained RTX 4060 Laptop (P8, 210 MHz SM / 405 MHz memory observed), PyTorch 2.9.0+cu128, Triton 3.5.0, WSL2. No L40S/A6000, Nsight memory-counter, full-model quality, or production-backend speedup is claimed.

## CUDA graphs

All 45 full-shape BF16 cases passed correctness. Graph replay includes device dispatch and reuses captured allocations. There are 20 eager warmups, 20 capture warmups, one capture, 20 replay warmups and 100 measured replays per case. T is flattened token count (recorded batch T, context 1). Attention and router computation are excluded.

The grouped/selective event-latency ratio ranges from 0.9627 to 1.0035 across 15 token/distribution cases. This does not establish a consistent fusion benefit. Eager and graph results are separate experiments; their ratios are not fusion speedups.

[Raw graph data](results/olmoe-bf16-graph.json) · [Graph plots](figures/graph)

## Threshold and tile search

The sweep tested 216 correct candidates: thresholds 0/1/2/4, combine columns 16/32/64 and reduction chunks 64/128/256, for T=1/8/32 and uniform/skewed routing. These are isolated down/combine graph timings with precomputed activations and metadata, 10 warmups per phase and 30 measurements. Their cache/allocation conditions differ from a complete block. Best-of-sweep selection is exploratory.

| Routes | T | Selected threshold | N × K | Best ms | Baseline before ms | Baseline drift |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 1 | 0 | 64 × 128 | 0.3886 | 0.3937 | -0.39% |
| uniform | 8 | 1 | 64 × 256 | 19.3224 | 19.4688 | -0.10% |
| uniform | 32 | 4 | 32 × 256 | 29.1384 | 29.3299 | -0.07% |
| skewed | 1 | 0 | 32 × 64 | 0.3871 | 0.4065 | -4.03% |
| skewed | 8 | 1 | 64 × 64 | 7.7373 | 7.7455 | +0.06% |
| skewed | 32 | 0 | 64 × 256 | 10.6880 | 10.7443 | +0.03% |

Threshold zero wins some cases. Small selected differences do not justify adopting a new default; the skewed one-token case also showed about 4% baseline drift. [Raw sweep](results/tuning-down-combine.json)

## Native PyTorch grouped-mm execution

The T=8 uniform BF16 profile contains 3 native grouped-mm calls, 192 individual matrix-multiply operator calls, and device-to-host offset copies. This is evidence of the host-loop fallback on this software/GPU, not execution of a fast single grouped kernel. The trace follows 20 warmups and records one invocation. Profiler durations are not used as benchmark latency. [Profile summary](results/native-profile.json) · [Trace](results/native-trace.json)

## Pretrained first-layer occupancy

A pinned OLMoE checkpoint (6d84c48581ece794365f2b8e9cfb043c68ade9c5) supplied 228.77 MiB of embedding, attention, normalization and router tensors. Later layers and expert MLP weights cannot affect the first router call, so the prefix can capture exact first-layer prefill dependencies without loading the full model. CPU BF16 capture uses upstream eager causal attention and FP32 rotary frequencies. Tiny random-model tests include exact equality with upstream pretrained-loading behavior.

The three authored repository probes are a narrow diagnostic sample, not a public-corpus or generation evaluation. Routes preserve FP32 probabilities before the model's BF16 cast. Replay with random expert weights remains distinct from pretrained block execution.

| Probe | Context tokens | Active experts | Singletons | Maximum occupancy |
| --- | --- | --- | --- | --- |
| technical | 88 | 60 | 4 | 71 |
| narrative | 87 | 64 | 1 | 79 |
| structured | 93 | 62 | 6 | 69 |

[Capture manifest and routes](first-layer-routes) · [Authored probes](../experiments/text-probes.json)

## Interpretation

The measured fusion hypothesis remains unproven. Keep the grouped baseline and existing defaults, report dispatch separately, and rerun on the proposed cluster GPU with stable power/clocks. Broader real-route sampling, a competitive external grouped kernel, memory counters and full-model quality evaluation remain open work.
