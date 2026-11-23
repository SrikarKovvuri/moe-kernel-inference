#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
OUT="${1:-runs/midpoint}"
mkdir -p "$OUT"
python -m pytest -q
python -m phasemoe.benchmark --preset olmoe --device cuda --dtype bfloat16 \
  --tokens 1 8 128 --distributions uniform balanced skewed \
  --backends torch triton_grouped --warmup 20 --repetitions 100 --output "$OUT/baseline.json"
python -m phasemoe.profiling --tokens 8 --warmup 20 --repetitions 100 \
  --output "$OUT/profile.json" --trace "$OUT/trace.json"
python -m phasemoe.plots "$OUT/baseline.json" --output "$OUT/figures"
