"""Adapt an existing Hugging Face OlmoeSparseMoeBlock without downloading weights."""

import torch

from .config import MoEConfig
from .reference import ExpertWeights
from .traces import routing_from_logits


def config_from_hf(module):
    first = module.experts[0]
    return MoEConfig(
        hidden_size=first.gate_proj.in_features,
        intermediate_size=first.gate_proj.out_features,
        num_experts=module.num_experts,
        top_k=module.top_k,
        norm_topk_prob=module.norm_topk_prob,
    )


@torch.inference_mode()
def weights_from_hf(module):
    """Copy Linear [out,in] weights into contiguous PhaseMoE [expert,in,out] stacks."""

    def stack(name):
        return torch.stack(
            [getattr(expert, name).weight.detach().T.contiguous() for expert in module.experts]
        )

    return ExpertWeights(stack("gate_proj"), stack("up_proj"), stack("down_proj"))


@torch.inference_mode()
def routing_from_hf(module, x):
    config = config_from_hf(module)
    if x.ndim != 2 or x.shape[1] != config.hidden_size:
        raise ValueError("x must have shape [tokens, hidden_size]")
    routing = routing_from_logits(
        module.gate(x), top_k=config.top_k, norm_topk_prob=config.norm_topk_prob
    )
    # Match the upstream module's cast after FP32 softmax/top-k.
    routing.weights = routing.weights.to(x.dtype)
    return routing
