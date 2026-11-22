import pytest
import torch

pytest.importorskip("transformers")

from transformers import OlmoeConfig, OlmoeForCausalLM

from phasemoe.pretrained_prefix import FirstLayerRouter
from phasemoe.traces import routing_from_logits


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 7, 31])
def test_prefix_matches_first_router_of_full_upstream_model(dtype, tokens):
    config = OlmoeConfig(
        vocab_size=101,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    full = OlmoeForCausalLM(config).to(dtype=dtype).eval()
    prefix = FirstLayerRouter(config).to(dtype=dtype).eval()
    source = full.state_dict()
    prefix.load_state_dict(
        {name: source[prefix.checkpoint_key(name)] for name in prefix.state_dict()}
    )
    inputs = torch.randint(0, 101, (1, tokens))
    with torch.inference_mode():
        expected = full(inputs, output_router_logits=True, use_cache=False).router_logits[0]
        actual = prefix(inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected_routes = routing_from_logits(expected, top_k=2)
    actual_routes = routing_from_logits(actual, top_k=2)
    torch.testing.assert_close(actual_routes.expert_ids, expected_routes.expert_ids, atol=0, rtol=0)
    torch.testing.assert_close(actual_routes.weights, expected_routes.weights, atol=0, rtol=0)


def test_prefix_matches_pretrained_loading_precision(tmp_path):
    config = OlmoeConfig(
        vocab_size=101,
        hidden_size=128,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
    )
    torch.manual_seed(7)
    OlmoeForCausalLM(config).save_pretrained(tmp_path)
    full = OlmoeForCausalLM.from_pretrained(
        tmp_path, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    prefix = FirstLayerRouter(config).to_inference()
    source = full.state_dict()
    prefix.load_state_dict(
        {name: source[prefix.checkpoint_key(name)] for name in prefix.state_dict()}
    )
    assert prefix.rotary_emb.inv_freq.dtype == torch.float32
    torch.testing.assert_close(
        prefix.rotary_emb.inv_freq, full.model.rotary_emb.inv_freq, atol=0, rtol=0
    )
    inputs = torch.randint(0, 101, (1, 47))
    with torch.inference_mode():
        expected = full(inputs, output_router_logits=True, use_cache=False).router_logits[0]
    torch.testing.assert_close(prefix(inputs), expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "status,content_range,payload",
    [
        (200, "", b"abcd"),
        (206, "bytes 5-8/100", b"abcd"),
        (206, "bytes 4-7/100", b"abc"),
        (206, "bytes 4-7/100", b"abcde"),
    ],
)
def test_range_loader_rejects_ignored_wrong_or_truncated_responses(
    monkeypatch, status, content_range, payload
):
    import requests

    from phasemoe.pretrained_prefix import read_range

    class Response:
        status_code = status
        headers = {"Content-Range": content_range}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, _size):
            yield payload

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError):
        read_range("https://example.invalid/weights", 4, 7)
