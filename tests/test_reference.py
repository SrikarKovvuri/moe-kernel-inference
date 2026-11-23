import pytest
import torch

from phasemoe.config import MoEConfig
from phasemoe.reference import ExpertWeights, error_metrics, moe_oracle, moe_torch, validate_inputs
from phasemoe.routing import Routing, occupancy, pack_routes, synthetic_routing


@pytest.mark.parametrize("distribution", ["uniform", "balanced", "skewed"])
@pytest.mark.parametrize("tokens", [0, 1, 8, 31])
def test_packing_is_a_bijection_and_conserves_assignments(distribution, tokens):
    c = MoEConfig.tiny()
    r = synthetic_routing(c, tokens, distribution, seed=13)
    p = pack_routes(r)
    assert p.offsets[0] == 0
    assert p.offsets[-1] == tokens * c.top_k
    assert torch.equal(p.assignment_ids[p.inverse], torch.arange(tokens * c.top_k))
    assert torch.equal(p.counts, torch.bincount(r.expert_ids.flatten(), minlength=c.num_experts))
    packed_ids = r.expert_ids.flatten()[p.assignment_ids]
    for e in range(c.num_experts):
        assert (packed_ids[p.offsets[e] : p.offsets[e + 1]] == e).all()
    stats = occupancy(r)
    assert sum(stats["histogram"]) == c.num_experts
    assert stats["assignments"] == tokens * c.top_k


@pytest.mark.parametrize("distribution", ["uniform", "balanced", "skewed"])
@pytest.mark.parametrize("tokens", [0, 1, 7])
@pytest.mark.parametrize("normalized", [True, False])
def test_expert_baseline_matches_independent_oracle(distribution, tokens, normalized):
    c = MoEConfig(7, 11, 5, 3, normalized)
    r = synthetic_routing(c, tokens, distribution, seed=11)
    w = ExpertWeights.random(c, seed=7)
    x = torch.randn(tokens, c.hidden_size, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(moe_torch(x, w, r), moe_oracle(x, w, r), atol=1e-6, rtol=1e-5)
    if normalized:
        torch.testing.assert_close(r.weights.sum(-1), torch.ones(tokens))
    else:
        assert (r.weights.sum(-1) < 1).all()


def test_weighted_slots_and_expert_orientation_by_hand():
    x = torch.tensor([[2.0]])
    w = ExpertWeights(
        torch.tensor([[[1.0]], [[2.0]]]),
        torch.tensor([[[3.0]], [[4.0]]]),
        torch.tensor([[[5.0]], [[6.0]]]),
    )
    r = Routing(torch.tensor([[1, 0]]), torch.tensor([[0.2, 0.1]]), 2)
    expected = 0.2 * torch.nn.functional.silu(torch.tensor(4.0)) * 8 * 6
    expected += 0.1 * torch.nn.functional.silu(torch.tensor(2.0)) * 6 * 5
    torch.testing.assert_close(moe_torch(x, w, r).squeeze(), expected)


@pytest.mark.parametrize(
    "ids,weights",
    [
        ([[0, 0]], [[0.2, 0.3]]),
        ([[0, 2]], [[0.2, 0.3]]),
        ([[0, 1]], [[-0.1, 0.3]]),
        ([[0, 1]], [[float("nan"), 0.3]]),
    ],
)
def test_invalid_routes_rejected(ids, weights):
    with pytest.raises(ValueError):
        Routing(torch.tensor(ids), torch.tensor(weights), 2).validate()


def test_inputs_and_empty_comparison():
    c = MoEConfig.tiny()
    with pytest.raises(ValueError):
        validate_inputs(torch.zeros(1, 5), ExpertWeights.random(c), synthetic_routing(c, 1))
    assert error_metrics(torch.empty(0, 5), torch.empty(0, 5), atol=0, rtol=0)["passed"]
    assert not error_metrics(torch.tensor([float("nan")]), torch.ones(1), atol=1, rtol=1)["passed"]


def test_balanced_occupancy_and_singleton_extremes():
    c = MoEConfig()
    r = synthetic_routing(c, 8, "balanced")
    assert occupancy(r)["singleton_experts"] == 64
    assert occupancy(synthetic_routing(c, 1))["singleton_experts"] == 8
    assert occupancy(synthetic_routing(c, 128, "balanced"))["max_occupancy"] == 16


def test_bfloat16_baseline_is_numerically_close():
    c = MoEConfig.tiny()
    x = torch.randn(7, c.hidden_size).bfloat16()
    w = ExpertWeights.random(c, dtype=torch.bfloat16)
    r = synthetic_routing(c, 7)
    metrics = error_metrics(moe_torch(x, w, r), moe_oracle(x, w, r), atol=0.02, rtol=0.03)
    assert metrics["passed"], metrics


@pytest.mark.parametrize("implementation", [moe_torch, moe_oracle])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_precision_is_independent_of_ambient_autocast(implementation, dtype):
    c = MoEConfig.tiny()
    x = torch.randn(7, c.hidden_size, generator=torch.Generator().manual_seed(3)).to(dtype)
    w = ExpertWeights.random(c, seed=7, dtype=dtype)
    r = synthetic_routing(c, 7, seed=11)
    expected = implementation(x, w, r)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = implementation(x, w, r)
        assert torch.is_autocast_enabled("cpu")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == (torch.float32 if implementation is moe_oracle else dtype)
