import json

import pytest
import torch

from phasemoe.config import MoEConfig
from phasemoe.routing import synthetic_routing
from phasemoe.traces import load_trace, routing_from_logits, save_trace


@pytest.mark.parametrize("tokens", [0, 1, 17])
def test_trace_roundtrip_and_provenance(tmp_path, tokens):
    r = synthetic_routing(MoEConfig.tiny(), tokens)
    path = tmp_path / "routes.json"
    provenance = dict(
        source="synthetic",
        model="none",
        model_revision="none",
        layer=0,
        phase="decode",
        norm_topk_prob=False,
    )
    save_trace(path, r, provenance=provenance)
    actual, observed = load_trace(path)
    assert observed == provenance
    torch.testing.assert_close(actual.expert_ids, r.expert_ids)
    torch.testing.assert_close(actual.weights, r.weights)


def test_router_probability_semantics():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    raw = routing_from_logits(logits, top_k=2)
    normalized = routing_from_logits(logits, top_k=2, norm_topk_prob=True)
    assert raw.expert_ids.tolist() == [[3, 2]]
    torch.testing.assert_close(raw.weights, logits.softmax(-1)[:, [3, 2]])
    assert raw.weights.sum() < 1
    torch.testing.assert_close(normalized.weights.sum(), torch.tensor(1.0))


def test_trace_refuses_missing_provenance_and_fractional_ids(tmp_path):
    r = synthetic_routing(MoEConfig.tiny(), 1)
    path = tmp_path / "trace.json"
    with pytest.raises(ValueError, match="provenance"):
        save_trace(path, r, provenance={})
    save_trace(
        path,
        r,
        provenance=dict(
            source="synthetic",
            model="none",
            model_revision="none",
            layer=0,
            phase="decode",
            norm_topk_prob=False,
        ),
    )
    data = json.loads(path.read_text())
    data["expert_ids"][0][0] = 0.5
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="integers"):
        load_trace(path)
