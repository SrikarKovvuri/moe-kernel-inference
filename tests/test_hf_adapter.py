import pytest
import torch

pytest.importorskip("transformers", reason="optional model integration dependencies")
from transformers import OlmoeConfig  # noqa: E402
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock  # noqa: E402

from phasemoe.hf_adapter import config_from_hf, routing_from_hf, weights_from_hf  # noqa: E402
from phasemoe.reference import error_metrics, moe_torch  # noqa: E402


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_against_upstream_olmoe_module_without_checkpoint(normalized, dtype):
    config = OlmoeConfig(
        hidden_size=32,
        intermediate_size=48,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=normalized,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        module = OlmoeSparseMoeBlock(config).to(dtype=dtype).eval()
        x = torch.randn(2, 3, config.hidden_size).to(dtype=dtype)
    with torch.inference_mode():
        expected, _ = module(x)
        flattened = x.reshape(-1, config.hidden_size)
        routing = routing_from_hf(module, flattened)
        actual = moe_torch(flattened, weights_from_hf(module), routing).reshape_as(expected)
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    else:
        metrics = error_metrics(actual, expected, atol=0.002, rtol=0.03)
        assert metrics["passed"] and metrics["relative_l2_error"] < 0.015, metrics
    assert config_from_hf(module).top_k == 2
    assert routing.weights.dtype == dtype
