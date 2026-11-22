#!/usr/bin/env bash
set -euo pipefail

# Run from a configured environment. This script does not submit jobs or install packages.
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
OUT="${1:-runs/midpoint}"
mkdir -p "$OUT"

python -m pytest -q
python -m phasemoe.benchmark --preset olmoe --device cuda --dtype bfloat16 \
  --tokens 1 8 32 128 512 --distributions uniform balanced skewed \
  --backends torch triton_grouped triton_hybrid triton_fused \
  --warmup 20 --repetitions 100 --output "$OUT/eager.json"
python -m phasemoe.benchmark --preset olmoe --device cuda --dtype bfloat16 \
  --tokens 1 8 32 128 512 --distributions uniform balanced skewed \
  --backends triton_grouped triton_hybrid triton_fused --mode cuda_graph \
  --warmup 20 --repetitions 100 --output "$OUT/cuda-graph.json"
python -m phasemoe.profiling --tokens 8 --threshold 1 --warmup 20 --repetitions 100 \
  --output "$OUT/profile.json" --trace "$OUT/trace.json"
python -m phasemoe.plots "$OUT/eager.json" --output "$OUT/figures-eager"
python -m phasemoe.plots "$OUT/cuda-graph.json" --output "$OUT/figures-graph"
