# Baseline midpoint evidence

This directory contains baseline measurements and a conservative midpoint report.
It does not contain an implemented research optimization or a performance verdict on one.

- `midpoint-report.md`, `.docx`, `.pdf`: concise report for team review.
- `results/baseline-initial.json`: 30 retained historical PyTorch/grouped-baseline
  cases, with unchanged raw timing samples and correctness metrics. The derivation
  records the original artifact hash, Git commit and removed unused metadata.
  These measurements describe the historical grouped path, not a fresh run of the
  simplified source. The full original artifact remains in Git history.
- `results/baseline-current.json`: fresh 18-case check of the simplified source,
  T=1/8/128, uniform/balanced/skewed, PyTorch/grouped, BF16, 20 warmups, 100 samples.
- `results/baseline-profile.json` and `baseline-trace.json`: fresh grouped-only
  stage measurements and one warmed diagnostic profiler trace at T=8, uniform.
- `results/source-audit.json`: verified measured benchmark/kernel hashes plus
  LF-normalized hashes for Git checkouts; raw measurement metadata is unchanged.
- `results/gpu-power-state.txt`, `cpu-hardware.txt`: original development hardware
  diagnostics. Fresh runs include their own per-case telemetry.
- `figures/`: current baseline latency and synthetic expert occupancy plots.

The report uses the fresh run. Both sets are preliminary laptop/WSL measurements,
not target-cluster, actual decode, pretrained quality or production-performance claims.
Wall and CUDA-event clocks are retained separately. Bands show p10-p90 timing
samples, not confidence intervals from independent workload trials.

Measurements were collected while the source restructure was uncommitted. Metadata
preserves `git_dirty=true`, the parent revision and actual source-file SHA256 hashes.
The source audit verifies the measured benchmark/kernel modules and accounts for
Git's CRLF-to-LF normalization; plot labels were corrected afterward. Do not cite the parent revision alone as the
source for these new results or rewrite the original dirty-tree flag.

## Regenerate

```sh
python -m phasemoe.plots reports/results/baseline-current.json --output reports/figures
python scripts/build_midpoint_report.py --benchmark reports/results/baseline-current.json \
  --profile reports/results/baseline-profile.json --output reports
```

The report builder needs the `report` extra. Render and visually inspect the Word
report before exporting its PDF. Team UNIs, roles, dashboard-link acceptance and
team-size approval remain required before submission.
