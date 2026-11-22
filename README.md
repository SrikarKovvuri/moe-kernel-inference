# PhaseMoE

**Expert-Selective Fusion for Efficient Mixture-of-Experts Inference**

Research question: can computing lightly loaded experts' down projections inside
output combination reduce MoE latency enough to offset selective dispatch costs?

This repository starts with reproducible single-layer experiments. The target is
OLMoE-1B-7B: 64 experts, top-8 routing, hidden width 2048, expert intermediate width
1024, and BF16. Synthetic random weights and routes are explicitly distinguished
from pretrained model execution. No speedup is assumed.

## Scope and evidence

- Build and validate an independent PyTorch reference and an expert-batched baseline.
- Measure occupancy and full expert-block latency with complete run metadata.
- Implement and validate grouped Triton projections and selective down/combine fusion.
- Compare grouped, uniformly fused, and selective execution, including dispatch overhead.
- Maintain a conservative midpoint report separating measurements from planned work.

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
