# Cluster rerun guide

The development measurements were collected on a power-constrained RTX 4060
Laptop. This guide prepares a comparable L40S/A6000 run; it does not claim that
cluster access or a job submission has been completed.

## Environment

Use one allocated GPU and an isolated Linux Python environment. Choose the CUDA
PyTorch wheel that is compatible with the cluster's driver. The exact validated
development versions are in `requirements/validation-linux.txt`; install them
when the cluster supports that stack. Do not install into a shared base environment.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev,plots,gpu,model]"
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name())'
python -m pytest -q
git status --short
nvidia-smi -q
```

The `model` extra is optional for synthetic benchmarks. It enables random-model
upstream parity tests without downloading a checkpoint. Checkpoint downloads occur
only through the explicit pretrained scripts.

## Main measurements

From an allocated GPU shell, run:

```sh
bash scripts/run_midpoint.sh runs/cluster-midpoint
```

For Slurm, adapt `scripts/midpoint.slurm` to the real account, partition and GPU
constraints. Submit from the repository root with the intended environment active.
The template requests one GPU, 16 GB host memory and one hour; its resource names
are deliberately not specific to an unverified cluster configuration.

Keep the raw JSON and source revision. The script runs tests, the 60-case eager
matrix, 45-case graph matrix, phase profiling and plots. Graph and eager timing
measure different execution modes; compare fusion against the matching grouped
baseline within each mode.

## Independent controls

```sh
python -m phasemoe.tuning --tokens 1 8 32 --distributions uniform skewed \
  --thresholds 0 1 2 4 --output runs/cluster-tuning.json
python -m phasemoe.ablation --variants experiments/dispatch-and-fusion.json \
  --seeds 11 22 33 --mode cuda_graph --output runs/cluster-controls-graph.json
python -m phasemoe.ablation --variants experiments/dispatch-and-fusion.json \
  --seeds 11 22 33 --mode eager --output runs/cluster-controls-eager.json
python scripts/profile_native_grouped.py --output runs/cluster-native-profile.json \
  --trace runs/cluster-native-trace.json
```

The native grouped-mm API may use a host-loop fallback on a given architecture.
Check the actual operator/kernel trace before describing it as a fast grouped
backend. The repository's Triton grouped implementation is a controlled baseline,
not a claim to match the strongest production implementation.

## Real data

```sh
python scripts/prepare_wikitext_probes.py
python scripts/capture_first_layer_routes.py --probes runs/wikitext-probes.json \
  --max-tokens 128 --output runs/cluster-wikitext-routes
python scripts/benchmark_pretrained_block.py --probes runs/wikitext-probes.json \
  --max-tokens 128 --tokens 1 8 0 --output runs/cluster-pretrained-block.json
```

The first-layer prefix requires about 229 MiB of checkpoint tensors; the expert
block adds about 768 MiB. Its tensors are cached outside the repository, with
revision and checksums recorded. The last 1/8 prefill states are not decode data.
For actual generation and later-layer routes, use `capture_olmoe_routes.py` with a
pinned revision on a machine capable of loading the full model.

## Interpreting the run

Require correctness before timing. Inspect device, power/clock telemetry, timing
mode, source hashes, dtype, token counts and before/after control drift. Retain
failed artifacts as failed; do not silently combine partial experiments with
completed ones. A selected tuning minimum needs independent full-block validation.
Repeated timing samples are not independent workload trials.

The development report builders summarize the archived laptop experiments. For a
cluster report, update the narrative and hardware statements from the new evidence;
do not reuse the laptop report's fixed hardware caveats or measured numbers.
