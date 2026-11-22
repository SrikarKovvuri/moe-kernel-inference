# Midpoint evidence

Read the [PDF](midpoint-report.pdf), [Markdown](midpoint-report.md), or editable
[Word report](midpoint-report.docx). These are drafts for team review:
UNIs, confirmed roles, the dashboard URL, and team-size approval still need team input.

The initial result is **no consistent singleton-fusion speedup** against this
repository's grouped Triton baseline. It is not a comparison with the strongest
production backend. All 60 full-shape BF16 cases passed the recorded correctness
tolerance, but the laptop GPU was heavily power/clock constrained.

[Follow-up experiments](followup-analysis.md) add 45 graph cases, a 216-candidate
threshold/tile sweep, native-backend execution profiling, and genuine pretrained
first-layer prefill occupancy. These do not replace or silently extend the initial
eager table. The updated midpoint draft separates its initial table from follow-up
validation; the extended analysis retains the detailed comparison tables.

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
- `results/olmoe-bf16-graph.json`: 45 graph cases, same full dimensions and 100
  measured replays; graph warmup/capture counts are explicit in every result.
- `results/tuning-down-combine.json`: randomized isolated-phase search, 216
  candidates, 10 warmups and 30 samples, with grouped controls before/after each case.
- `results/native-profile.json` and `native-trace.json`: evidence that native
  PyTorch grouped-mm uses its host-loop fallback on this SM89 device/software.
- `first-layer-routes/`: pretrained layer-0 prefill routes from three authored
  text probes; pinned checkpoint revision and per-tensor checksums. This small
  diagnostic sample is not representative of a corpus or later model layers.
- `wikitext-routes/`: three pinned public WikiText-2 test paragraph captures at
  context 128, with source attribution and hashes; raw dataset text is not committed.
- `results/pretrained-block.json`: 27 checkpoint-backed BF16 GPU cases, using
  actual expert weights and prefix hidden states. Every case passes both FP32 and
  upstream BF16 comparisons. See its [source audit](results/pretrained-block-provenance.md)
  for the preserved dirty-tree flag; these remain single-block graph measurements.
- `results/ablation-graph.json`: 126 full-block measurements across 18 independent
  seed/shape/routing cases, separating dispatch improvements from fusion.
- `results/ablation-eager.json`: the same 126 randomized measurements in eager
  execution, with before/after grouped controls and three independent route seeds.

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
python scripts/analyze_followup.py --reports reports
```

The report builder requires `python-docx`; render the DOCX and inspect every page
after changing it. The benchmark dependencies do not require a document renderer.
