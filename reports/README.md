# Midpoint evidence

Read [midpoint-report.md](midpoint-report.md) or the editable
[midpoint-report.docx](midpoint-report.docx). Both are drafts for team review:
UNIs, confirmed roles, the dashboard URL, and team-size approval still need team input.

The initial result is **no consistent singleton-fusion speedup** against this
repository's grouped Triton baseline. It is not a comparison with the strongest
production backend. All 60 full-shape BF16 cases passed the recorded correctness
tolerance, but the laptop GPU was heavily power/clock constrained.

## Evidence files

- `results/olmoe-bf16-initial.json`: 60 cases, 100 measured eager calls and 20 warmups
  per case; 5 token counts x 3 distributions x 4 backends. Raw wall and CUDA-event
  samples, shapes, dtype, source revision, correctness and occupancy are retained.
- `results/profile-eager.json`: isolated stage timings at 8 tokens, uniform routing,
  BF16, threshold 1; 20 warmups and 100 calls per stage. Precomputed stage inputs
  mean these medians must not be summed into an end-to-end latency claim.
- `results/profile-trace.json`: a single warmed PyTorch CPU/CUDA profiler trace.
  Open in Perfetto or a Chrome trace viewer. It is not a Nsight memory-counter run.
- `results/gpu-power-state.txt`: diagnostic snapshot during development, reporting
  P8, 210 MHz SM, 405 MHz memory and software power/thermal caps. It is not a
  continuous per-sample clock trace. Later benchmark artifacts include before/after
  `nvidia-smi` snapshots per case.
- `figures/`: regenerated latency and occupancy plots. Latency bands are p10–p90
  over repeated timing samples, not confidence intervals across independent runs.

All initial measurements use synthetic random expert weights and routes at OLMoE
dimensions. No checkpoint weights, real prompts, or model quality results are
included. Large token counts are shape analogues of prefill, not complete model
prefill timing. Initial backends ran in a fixed order; repeated/randomized controls
are needed before interpreting small differences.

## Regeneration

```sh
python -m phasemoe.plots reports/results/olmoe-bf16-initial.json \
  --output reports/figures \
  --note "Constrained development run: P8, 210 MHz SM / 405 MHz memory observed"
python scripts/build_midpoint_report.py \
  --benchmark reports/results/olmoe-bf16-initial.json \
  --profile reports/results/profile-eager.json --output reports
```

The report builder requires `python-docx`; render the DOCX and inspect every page
after changing it. The benchmark dependencies do not require a document renderer.
