# PhaseMoE

**Expert-Selective Fusion for Efficient Mixture-of-Experts Inference**

Research question: can computing lightly loaded experts' down projections inside
output combination reduce MoE latency enough to offset selective dispatch costs?

This repository contains reproducible single-block experiments. The target is
OLMoE-1B-7B: 64 experts, top-8 routing, hidden width 2048, expert intermediate width
1024, and BF16. Synthetic random weights and routes are explicitly distinguished
from pretrained model execution. No speedup is assumed.

## Scope and evidence

- Build and validate an independent PyTorch reference and an expert-batched baseline.
- Measure occupancy and full expert-block latency with complete run metadata.
- Implement and validate grouped Triton projections and selective down/combine fusion.
- Compare grouped, uniformly fused, and selective execution, including dispatch overhead.
- Maintain a conservative midpoint report separating measurements from planned work.

The initial 60-case full-shape BF16 sweep passed correctness on an RTX 4060 Laptop
GPU through WSL2. Singleton fusion was approximately at parity with our grouped
baseline, with no consistent speedup. **The GPU was power/clock constrained in P8
(210 MHz SM, 405 MHz memory observed); these are development measurements.** They
do not establish L40S/A6000 performance. See the [midpoint draft](reports/midpoint-report.md),
[editable report](reports/midpoint-report.docx), [raw evidence](reports/results), and
[figures](reports/figures).

The [follow-up analysis](reports/followup-analysis.md) separates graph timing,
threshold/tile tuning, native backend profiling, and pretrained first-layer routes.

The expert block includes dispatch, gate/up projections, SiLU multiplication, down
projection, routing weights, and output reduction. Transformer attention, RMSNorm,
router logits/top-k selection, and model loading are outside the initial benchmark.
Routing preparation is included in full-block timings unless explicitly labeled otherwise.

## Development

Use an isolated Python environment. Install the appropriate PyTorch build first;
the default PyPI package may not enable CUDA on your platform.

```sh
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -e ".[dev,plots]"
python -m pytest
```

GPU dependencies are optional; CPU correctness and occupancy analysis remain useful.
The research target is Linux/CUDA on L40S or A6000. Laptop GPU results, if collected,
are development evidence and do not establish target-cluster performance.

## Reproducible benchmark

```sh
python -m phasemoe.benchmark --preset tiny --device cpu --tokens 1 8 32 \
  --distributions uniform balanced skewed --warmup 20 --repetitions 100 \
  --output runs/cpu-smoke.json
```

Each JSON artifact records raw timings, correctness errors/tolerances, hardware,
software, model shape, routing seed/distribution, warmup/repetitions, timing scope,
and Git revision. CPU results validate the pipeline; they do not measure GPU fusion.
Failed runs retain completed cases and an explicit failure status.

For the validated Linux/CUDA stack:

```sh
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev,plots,gpu,model]"
python -m pytest -q
python -m phasemoe.benchmark --preset olmoe --device cuda --dtype bfloat16 \
  --tokens 1 8 32 128 512 --distributions uniform balanced skewed \
  --backends torch triton_grouped triton_hybrid triton_fused \
  --warmup 20 --repetitions 100 --output runs/eager.json
```

The `model` extra is optional. It enables parity tests against the Hugging Face
OLMoE module using random weights and the separate pretrained trace-capture script.
Normal tests and benchmarks do not download any model checkpoint. Exact development
versions are in [requirements/validation-linux.txt](requirements/validation-linux.txt).

Use `--mode cuda_graph` with only Triton backends to measure graph replay. Graph
timing still includes GPU dispatch/occupancy work, but reuses captured allocations
and excludes Python launch work within the block. It is a different timing mode,
not a drop-in replacement for eager latency. Captured inputs and shapes are fixed;
the test suite verifies that changing routing values recomputes the correct output.

Additional controlled alternatives are available:

- `--backends torch_grouped` calls PyTorch's native grouped-mm API (PyTorch 2.9
  in our environment). Some architectures/dtypes use a host-loop fallback; an API
  name alone does not establish execution by a fast grouped GPU kernel. This backend
  is eager-only in the benchmark and requires dimensions aligned to 16-byte strides.
- `--packing triton_scan` replaces stable sort/scatter dispatch with two Triton
  launches for 1–4096 assignments. It scans all assignments per expert, so its
  O(experts × assignments) work is intended for small batches. It raises an error
  outside that range; empty expert blocks still return an empty output.
- `--combine-n` and `--combine-k` expose the reduction tile sizes for independent
  reruns of tuning choices. JSON records the actual Triton configuration. Defaults
  retain the original sort-based dispatcher and 32 × 128 combine tile.

## Profiling and tuning

```sh
python -m phasemoe.profiling --tokens 8 --threshold 1 \
  --output runs/profile.json --trace runs/trace.json
python -m phasemoe.tuning --tokens 1 8 32 --distributions uniform skewed \
  --thresholds 1 2 4 --scope down_combine --output runs/tuning.json
python -m phasemoe.plots runs/eager.json --output runs/figures
```

Isolated stage sums are not end-to-end latency. The tuning tool randomizes candidate
order and measures a baseline before/after each case. A best-of-sweep result needs
an independent full-block rerun. No Nsight memory-traffic measurement is claimed.
See [kernel design](docs/kernel-design.md) for tensor layout, numerical boundaries,
single-writer aggregation, scheduling costs, and the distinction between avoided
traffic and reduced allocation.

Independent full-block controls randomize cases/candidate order, use fresh seeds,
and bracket each case with the original grouped baseline:

```sh
python -m phasemoe.ablation --variants experiments/dispatch-and-fusion.json \
  --seeds 11 22 33 --tokens 1 8 32 --distributions uniform skewed \
  --mode cuda_graph --output runs/ablation.json
python scripts/profile_native_grouped.py --output runs/native-profile.json \
  --trace runs/native-trace.json
```

The ablation JSON retains results after every variant. It distinguishes independent
seed cases from repeated timing samples and records before/after baseline drift.
The native profiler records actual CPU operator and CUDA kernel counts; profile
durations are not substituted for warmed benchmark measurements.

[scripts/run_midpoint.sh](scripts/run_midpoint.sh) reproduces the main suite;
[scripts/midpoint.slurm](scripts/midpoint.slurm) is a template that must be adapted
to the actual cluster partition/account and environment. No cluster job has been submitted.

## Pretrained routing traces

On a machine with sufficient model memory, capture actual router calls explicitly:

```sh
python scripts/capture_olmoe_routes.py --layer 0 --new-tokens 16 --output runs/real-routes
python -m phasemoe.benchmark --preset olmoe --device cuda --dtype bfloat16 \
  --backends triton_grouped triton_hybrid \
  --routing-trace runs/real-routes/layer-00-0000-prefill.json --output runs/trace-replay.json
```

Pin `--revision` to a model commit for published experiments. The capture script
records model revision, layer, phase, normalization policy, and a prompt hash;
it does not save prompt text. Replay uses recorded routes with **synthetic weights
and inputs**, not a checkpoint's hidden states or end-to-end model execution.
No pretrained routes are included in the initial results.

For first-layer prefill occupancy on a smaller machine, only the causal prefix
through its router is needed. An optional script downloads approximately 229 MiB
of selected checkpoint tensors using exact HTTP byte ranges, with a pinned model
commit and per-tensor checksums. It does not load any expert MLP or later layer:

```sh
python scripts/capture_first_layer_routes.py --device cpu --output runs/first-layer-routes
python -m phasemoe.benchmark --preset olmoe --device cuda --dtype bfloat16 \
  --backends triton_grouped triton_hybrid --mode cuda_graph \
  --routing-trace runs/first-layer-routes/probe-00-prefill.json \
  --output runs/first-layer-replay.json
```

The prefix matches the full upstream model's first router logits exactly in small
FP32/BF16 tests. Captured pretrained routes are limited to **layer 0 prefill on three
authored text probes**, not a representative corpus, generation run, later-layer
sample, or end-to-end inference benchmark. The original full-model script remains
the route for actual generation/decode collection on a sufficiently large machine.
See [Hugging Face's safetensors range-reading documentation](https://huggingface.co/docs/safetensors/en/metadata_parsing).

## Team

Aedin Pereira, Srikar Kovvuri, David Samtani. UNIs and confirmed role assignments
must be supplied by the team before submission.

## Sources

- [OLMoE paper](https://arxiv.org/abs/2409.02060)
- [OLMoE model configuration](https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/main/config.json)
- [Locality-aware MoE kernels](https://pytorch.org/blog/accelerating-moe-model/)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)

The model configuration uses **unnormalized selected top-k probabilities** by
default (`norm_topk_prob=false`). Our routing API preserves this choice explicitly.
