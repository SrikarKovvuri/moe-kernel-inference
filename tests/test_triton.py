import importlib.util

import pytest
import torch

from phasemoe.config import MoEConfig
from phasemoe.reference import ExpertWeights, error_metrics, moe_oracle, moe_torch
from phasemoe.routing import synthetic_routing

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.triton,
    pytest.mark.skipif(
        not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
        reason="CUDA and Triton are required",
    ),
]


@pytest.mark.parametrize(
    "strategy,threshold",
    [("grouped", 0), ("hybrid", 0), ("hybrid", 1), ("hybrid", 2), ("hybrid", 4), ("fused", 1)],
)
@pytest.mark.parametrize("tokens", [0, 1, 3, 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_matches_oracle_with_tail_dimensions(strategy, threshold, tokens, dtype):
    from phasemoe.triton_backend import moe_triton

    c = MoEConfig(37, 53, 5, 3)
    w = ExpertWeights.random(c, seed=4, device="cuda", dtype=dtype)
    r = synthetic_routing(c, tokens, seed=9).to("cuda")
    x = torch.randn(tokens, c.hidden_size, device="cuda", dtype=dtype)
    out = moe_triton(x, w, r, strategy=strategy, threshold=threshold)
    reference = moe_oracle(x, w, r)
    atol, rtol = (2e-5, 1e-4) if dtype == torch.float32 else (0.02, 0.03)
    metrics = error_metrics(out, reference, atol=atol, rtol=rtol)
    assert metrics["passed"], metrics


@pytest.mark.parametrize("distribution", ["balanced", "uniform", "skewed"])
@pytest.mark.parametrize("tokens", [1, 8, 32, 65])
def test_hybrid_matches_grouped_and_torch(distribution, tokens):
    from phasemoe.triton_backend import moe_triton

    c = MoEConfig(64, 96, 64, 8)
    w = ExpertWeights.random(c, device="cuda", dtype=torch.bfloat16)
    r = synthetic_routing(c, tokens, distribution, seed=5).to("cuda")
    x = torch.randn(tokens, c.hidden_size, device="cuda", dtype=torch.bfloat16)
    baseline = moe_torch(x, w, r)
    for threshold in (0, 1, 2, 4, tokens):
        out = moe_triton(x, w, r, threshold=threshold)
        metrics = error_metrics(out, baseline, atol=0.02, rtol=0.03)
        assert metrics["passed"], metrics


def test_graph_replay_recomputes_routing_and_outputs():
    from phasemoe.triton_backend import moe_triton

    c = MoEConfig.tiny()
    w = ExpertWeights.random(c, device="cuda", dtype=torch.bfloat16)
    r = synthetic_routing(c, 8, "balanced").to("cuda")
    x = torch.randn(8, c.hidden_size, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            moe_triton(x, w, r)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = moe_triton(x, w, r)
    for distribution in ("uniform", "skewed", "balanced"):
        fresh = synthetic_routing(c, 8, distribution, seed=42).to("cuda")
        r.expert_ids.copy_(fresh.expert_ids)
        r.weights.copy_(fresh.weights)
        x.mul_(0.7)
        graph.replay()
        torch.cuda.synchronize()
        expected = moe_torch(x, w, r)
        metrics = error_metrics(actual, expected, atol=0.02, rtol=0.03)
        assert metrics["passed"], metrics
