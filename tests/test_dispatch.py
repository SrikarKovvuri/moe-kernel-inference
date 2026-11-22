import importlib.util

import pytest
import torch

from phasemoe.config import MoEConfig
from phasemoe.routing import pack_routes, synthetic_routing

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.triton,
    pytest.mark.skipif(
        not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
        reason="CUDA and Triton required",
    ),
]


@pytest.mark.parametrize("tokens", [1, 3, 8, 31, 128, 512])
@pytest.mark.parametrize("distribution", ["uniform", "balanced", "skewed"])
@pytest.mark.parametrize("experts,top_k", [(5, 3), (64, 8)])
def test_scan_dispatch_exactly_matches_stable_sort(tokens, distribution, experts, top_k):
    from phasemoe.dispatch import pack_routes_triton

    config = MoEConfig(32, 48, experts, top_k)
    routing = synthetic_routing(config, tokens, distribution, seed=42).to("cuda")
    actual, ends = pack_routes_triton(routing)
    expected = pack_routes(routing)
    for name in ("assignment_ids", "inverse", "counts", "offsets"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=0, rtol=0)
    torch.testing.assert_close(ends, ((expected.counts + 15) // 16).cumsum(0), atol=0, rtol=0)


def test_scan_dispatch_bounds():
    from phasemoe.dispatch import pack_routes_triton

    for tokens in (0, 513):
        routing = synthetic_routing(MoEConfig(), tokens).to("cuda")
        with pytest.raises(ValueError, match="assignments"):
            pack_routes_triton(routing)


@pytest.mark.parametrize("strategy", ["grouped", "hybrid", "fused"])
@pytest.mark.parametrize("tokens", [0, 1, 31, 512])
def test_scan_dispatch_full_block(strategy, tokens):
    from phasemoe.reference import ExpertWeights, error_metrics, moe_torch
    from phasemoe.triton_backend import KernelConfig, moe_triton

    config = MoEConfig(37, 53, 64, 8)
    routing = synthetic_routing(config, tokens, "skewed", seed=42).to("cuda")
    weights = ExpertWeights.random(config, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    actual = moe_triton(
        x, weights, routing, strategy=strategy, config=KernelConfig(packing="triton_scan")
    )
    expected = moe_torch(x, weights, routing)
    check = error_metrics(actual, expected, atol=0.02, rtol=0.03)
    assert check["passed"], check
