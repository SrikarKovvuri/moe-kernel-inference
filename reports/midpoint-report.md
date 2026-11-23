# PhaseMoE midpoint checkpoint

Baseline establishment and characterization

Aedin Pereira, Srikar Kovvuri, David Samtani. UNIs and roles pending.

## Project summary and baseline

We have established a correct, reproducible single-block baseline and begun measuring expert occupancy and execution cost. The research question is whether a different GPU execution design can improve MoE inference under uneven expert workloads. The central optimization is planned work for the second half; no optimization result is claimed.

The checkpoint contains a PyTorch reference, independent FP32 oracle, deterministic synthetic routes, and a fixed grouped-GEMM baseline. All active experts use grouped projections and store down-projection rows before a separate weighted reduction. The inherited gate/up stage shares a launch with activation. This is an inspectable baseline, not a production-performance standard or a finalized research design.

## Experimental setup

Measured hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8.00 GiB, compute capability [8, 9]; host Intel(R) Core(TM) Ultra 9 185H. Python 3.12.3, PyTorch 2.9.0+cu128, Triton 3.5.0, CUDA runtime 12.8; driver/device query: NVIDIA GeForce RTX 4060 Laptop GPU, 621.14, [N/A]. Platform: Linux-5.15.153.1-microsoft-standard-WSL2-x86_64-with-glibc2.39; 1 CPU thread(s).

Shape E=64, K=8, H=2048, I=1024; torch.bfloat16. T=1,8,128 flattened synthetic tokens (recorded batch T, context 1); uniform, balanced and skewed routes. Seeds are 0 for routes, 1 for random weights and 2 for inputs. Correctness/JIT precede 20 warmups and 100 eager samples per case. Both synchronized wall and CUDA-event timings include dispatch, allocation, expert projections, activation and weighted reduction; router selection, attention, normalization and loading are excluded.

All 18 full-shape cases passed FP32-reference checks at atol 0.02 / rtol 0.03; largest relative L2 error 0.483%. Small-shape tests also check an independent oracle, empty batches, tail dimensions and routing/indexing. These checks do not establish pretrained-model quality.

## Preliminary baseline latency

Uniform synthetic routing; medians in ms; setup above, 20 warmups and 100 samples. PyTorch loops over experts and is not grouped GEMM.

| T | PyTorch CUDA | Grouped CUDA | Grouped wall |
| --- | --- | --- | --- |
| 1 | 13.713 | 1.331 | 1.425 |
| 8 | 36.367 | 3.514 | 3.634 |
| 128 | 32.918 | 5.887 | 5.955 |

## Profiling and occupancy observations

Grouped baseline, torch.bfloat16, T=8, uniform routes; same device and model dimensions. Each stage uses 20 warmups and 100 eager samples. Inputs to isolated stages are precomputed; do not add their medians to estimate block latency.

| Stage | CUDA ms | Wall ms |
| --- | --- | --- |
| dispatch | 0.917 | 0.977 |
| gate up activation | 1.760 | 1.836 |
| down projection | 0.969 | 1.024 |
| output reduction | 0.073 | 0.108 |
| full expert block | 3.303 | 3.337 |

The largest measured isolated phase is gate up activation. At T=8, the profiled routes have 26 singleton and 21 empty experts out of 64. Occupancy varies with distribution and token count. This motivates investigating utilization, dispatch and data movement, but does not identify an optimal design or prove a memory-bandwidth bottleneck. A warmed PyTorch profiler trace and occupancy plots are retained; no Nsight traffic counters were collected.

## Remaining technical work

The team will reproduce the baseline on L40S/A6000, establish an optimized external comparison, and collect actual prefill/decode routes across layers and batch sizes. We will independently explore layouts, execution strategies, wider or narrower fusion, and occupancy-aware scheduling. No fusion boundary or singleton threshold is prescribed. Candidates must preserve correctness; full-block measurements including scheduling will determine whether they help. Negative findings remain valid outcomes.

## Limitations and requirements

Preliminary synthetic tests use fixed backend order on laptop/WSL hardware. Snapshots report states P0, P3, P4, SM clocks 210-1890 MHz and memory clocks 405-8001 MHz. Clocks were not locked; these are before/after snapshots, not continuous measurements. CUDA-event and wall medians differ by up to 8.8%; the cause is unresolved. Target-GPU replication and independent repeats are needed. Small-T tests are not actual decode. Historical baseline samples are retained separately, not used for before/after claims.

UNIs, confirmed roles and team-size approval remain pending (three names supplied; template requests four to five). README, JSON and plots document experiments. W&B is optional; the rubric still requires a dashboard link or instructor approval of the repository summary as an alternative.

## Project links and reproducibility

Repository: https://github.com/SrikarKovvuri/moe-kernel-inference

Local evidence: reports/results/baseline-current.json and baseline-profile.json; figures in reports/figures. Historical subset: baseline-initial.json.

References: OLMoE paper https://arxiv.org/abs/2409.02060 and the model configuration linked from the README.

Measured 2026-11-23 UTC. Parent revision 905a50c; working tree modified at capture: True. JSON records source hashes and raw samples. Use those hashes and the checkpoint source files to identify the measured code; the parent revision alone does not identify an uncommitted measurement snapshot.
