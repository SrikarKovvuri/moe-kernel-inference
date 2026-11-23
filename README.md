# PhaseMoE midpoint checkpoint

This checkpoint establishes and characterizes an OLMoE-shaped MoE baseline.
The team's second-half research is to independently design, implement and evaluate
GPU execution strategies for uneven expert workloads. No proposed optimization or
optimization speedup is part of this checkpoint.

## What is implemented

- OLMoE-1B-7B dimensions: 64 experts, top-8 routing, hidden width 2048,
  intermediate width 1024; selected top-k probabilities are not renormalized.
- Independent FP32 token-wise oracle and an expert-loop PyTorch reference.
- A fixed Triton grouped-GEMM baseline with a separate output reduction.
- Deterministic uniform, balanced and skewed synthetic routing; occupancy counts/histograms.
- Baseline correctness tests, eager latency measurement, basic stage profiling and plots.

All active experts use the same grouped projection path. The retained baseline
is extracted from the existing grouped implementation, not a new optimization.
Its gate/up stage computes both projections and SiLU multiplication in one launch;
its down projection materializes every routed row before output reduction.
There is no selectable strategy, threshold policy, alternative dispatcher or tuning tool.
See [baseline design](docs/baseline-design.md) for the exact scope and limitations.

The PyTorch reference loops over experts; it is **not** a grouped-GEMM implementation.
The custom Triton baseline is a reproducible starting point, not an established
production-performance standard or the team's final kernel design.

## Setup and correctness

Python 3.10+ is required. Install the appropriate PyTorch build in an isolated environment.
For CPU development:

```sh
python -m venv .venv
# Activate the environment for your shell.
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,plots]"
python -m pytest -q
python -m phasemoe.benchmark --preset tiny --device cpu --tokens 1 8 \
  --warmup 1 --repetitions 3 --output runs/cpu-smoke.json
```

CPU tests skip GPU cases and do not require Triton. No checkpoint is downloaded.
For the validated Linux/WSL CUDA stack, install CUDA PyTorch first, then the GPU extra:

```sh
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev,plots,gpu]"
python -m pytest -q
bash scripts/run_midpoint.sh runs/midpoint
```

Exact development versions are in [requirements/validation-linux.txt](requirements/validation-linux.txt).
The runner executes tests, 18 BF16 cases (T=1/8/128, three routing distributions,
two baselines), basic stage profiling and figures. It does not install packages or submit jobs.
The [cluster guide](docs/cluster-runbook.md) explains the Slurm template.

## Individual experiments

```sh
python -m phasemoe.benchmark --preset olmoe --device cuda --dtype bfloat16 \
  --tokens 1 8 128 --distributions uniform balanced skewed \
  --backends torch triton_grouped --warmup 20 --repetitions 100 \
  --output runs/baseline.json
python -m phasemoe.profiling --tokens 8 --warmup 20 --repetitions 100 \
  --output runs/profile.json --trace runs/trace.json
python -m phasemoe.plots runs/baseline.json --output runs/figures
```

Full-block timing includes route packing, grouped projections, activation,
allocation and weighted output reduction. It excludes router logits/top-k,
attention, normalization and model loading. Inputs, weights and routes are synthetic;
small token counts are not actual decode and large counts are not model prefill.
Both synchronized wall time and CUDA-event samples are retained. Isolated stage
medians must not be summed into an end-to-end latency claim.

## Midpoint evidence and remaining work

Read the [midpoint report](reports/midpoint-report.md), [PDF](reports/midpoint-report.pdf),
[Word draft](reports/midpoint-report.docx), and [evidence index](reports/README.md).
Preliminary measurements use a power-constrained RTX 4060 Laptop under WSL2.
They characterize this baseline; they do not establish target-GPU performance or
validate any future optimization. Raw samples, correctness tolerances, hardware,
source hashes and dirty-tree status are recorded. Measurements collected during
the restructure retain their original metadata; use the recorded source hashes
and the checkpoint files, not the parent Git revision alone, to identify that code.

The [research plan](docs/experiment-plan.md) covers target-GPU replication,
stronger external baselines, real routing data, and independent kernel exploration.
Fusion scope, execution strategies and scheduling policy remain open decisions.

## Team and links

Aedin Pereira, Srikar Kovvuri, David Samtani. UNIs, confirmed roles, an accessible
experiment-dashboard URL, and approval of the three-person team remain pending.

- [Repository](https://github.com/SrikarKovvuri/moe-kernel-inference)
- [OLMoE paper](https://arxiv.org/abs/2409.02060)
- [OLMoE configuration](https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/6d84c48581ece794365f2b8e9cfb043c68ade9c5/config.json)
- [Triton grouped-GEMM tutorial](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)
