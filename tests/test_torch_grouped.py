import pytest
import torch

from phasemoe.config import MoEConfig
from phasemoe.reference import ExpertWeights, error_metrics, moe_oracle
from phasemoe.routing import synthetic_routing
from phasemoe.torch_grouped import moe_torch_grouped

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "_grouped_mm") or hasattr(torch.nn.functional, "grouped_mm")),
    reason="native grouped-mm unavailable",
)


@pytest.mark.parametrize("tokens", [0, 1, 7, 32])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_native_grouped_matches_oracle(tokens, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    c = MoEConfig.tiny()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    w = ExpertWeights.random(c, device=device, dtype=dtype)
    r = synthetic_routing(c, tokens).to(device)
    x = torch.randn(tokens, c.hidden_size, device=device, dtype=dtype)
    actual = moe_torch_grouped(x, w, r)
    expected = moe_oracle(x, w, r)
    atol, rtol = (0.02, 0.03) if device == "cuda" else (1e-6, 1e-5)
    check = error_metrics(actual, expected, atol=atol, rtol=rtol)
    assert check["passed"], check
