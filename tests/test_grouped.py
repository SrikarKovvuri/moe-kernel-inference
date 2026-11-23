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


@pytest.mark.parametrize("tokens", [0, 1, 3, 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_grouped_matches_independent_oracle_with_tail_dimensions(tokens, dtype):
    from phasemoe.grouped import moe_grouped

    c = MoEConfig(37, 53, 5, 3)
    w = ExpertWeights.random(c, seed=4, device="cuda", dtype=dtype)
    r = synthetic_routing(c, tokens, seed=9).to("cuda")
    x = torch.randn(tokens, c.hidden_size, device="cuda", dtype=dtype)
    atol, rtol = (2e-5, 1e-4) if dtype == torch.float32 else (0.02, 0.03)
    metrics = error_metrics(moe_grouped(x, w, r), moe_oracle(x, w, r), atol=atol, rtol=rtol)
    assert metrics["passed"], metrics


@pytest.mark.parametrize("distribution", ["balanced", "uniform", "skewed"])
@pytest.mark.parametrize("tokens", [1, 8, 32, 65])
def test_grouped_matches_pytorch_with_64_experts(distribution, tokens):
    from phasemoe.grouped import moe_grouped

    c = MoEConfig(64, 96, 64, 8)
    w = ExpertWeights.random(c, device="cuda", dtype=torch.bfloat16)
    r = synthetic_routing(c, tokens, distribution, seed=5).to("cuda")
    x = torch.randn(tokens, c.hidden_size, device="cuda", dtype=torch.bfloat16)
    metrics = error_metrics(moe_grouped(x, w, r), moe_torch(x, w, r), atol=0.02, rtol=0.03)
    assert metrics["passed"], metrics


def test_grouped_benchmark_and_profile_end_to_end(tmp_path):
    from phasemoe.benchmark import main as benchmark
    from phasemoe.profiling import main as profile

    result = benchmark(
        [
            "--preset",
            "tiny",
            "--device",
            "cuda",
            "--dtype",
            "bfloat16",
            "--backends",
            "triton_grouped",
            "--tokens",
            "1",
            "8",
            "--warmup",
            "1",
            "--repetitions",
            "2",
            "--output",
            str(tmp_path / "bench.json"),
        ]
    )
    assert result["status"] == "complete"
    assert all(r["correctness"]["passed"] for r in result["results"])
    result = profile(
        [
            "--preset",
            "tiny",
            "--tokens",
            "8",
            "--warmup",
            "1",
            "--repetitions",
            "2",
            "--output",
            str(tmp_path / "profile.json"),
        ]
    )
    assert result["status"] == "complete"
    assert set(result["stages"]) == {
        "dispatch",
        "gate_up_activation",
        "down_projection",
        "output_reduction",
        "full_expert_block",
    }
