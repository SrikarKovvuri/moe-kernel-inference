"""Capture actual pretrained OLMoE routes on a sufficiently large CUDA GPU.

Optional dependencies: transformers and accelerate. This script downloads the model
only when explicitly run. It does not record prompt text or activation/weight tensors.
"""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import torch

from phasemoe.traces import routing_from_logits, save_trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument(
        "--revision", default="main", help="prefer a pinned model commit for publication"
    )
    p.add_argument("--layer", type=int, default=0)
    p.add_argument(
        "--prompt", default="Mixture of experts models route each token to a few experts."
    )
    p.add_argument("--new-tokens", type=int, default=16)
    p.add_argument("--output", type=Path, default=Path("runs/real-routes"))
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("this capture script requires a CUDA GPU with enough memory for the full model")
    if args.layer < 0 or args.new_tokens < 1:
        p.error("layer must be nonnegative and new-tokens positive")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    ).eval()
    if args.layer >= len(model.model.layers):
        p.error("layer exceeds model depth")
    module = model.model.layers[args.layer].mlp
    calls = 0
    resolved_revision = getattr(model.config, "_commit_hash", None) or args.revision

    def record_router(_module, _inputs, output):
        nonlocal calls
        logits = output.detach().reshape(-1, model.config.num_experts)
        routing = routing_from_logits(
            logits,
            top_k=model.config.num_experts_per_tok,
            norm_topk_prob=model.config.norm_topk_prob,
        )
        phase = "prefill" if calls == 0 else "decode"
        save_trace(
            args.output / f"layer-{args.layer:02d}-{calls:04d}-{phase}.json",
            routing,
            provenance={
                "source": "pretrained_model",
                "model": args.model,
                "model_revision": resolved_revision,
                "layer": args.layer,
                "phase": phase,
                "call_index": calls,
                "norm_topk_prob": model.config.norm_topk_prob,
                "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "weight_dtype": "bfloat16",
                "routing_weight_dtype": "float32 before model cast",
            },
        )
        calls += 1

    handle = module.gate.register_forward_hook(record_router)
    try:
        inputs = tokenizer(args.prompt, return_tensors="pt").to(
            model.get_input_embeddings().weight.device
        )
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=args.new_tokens, do_sample=False)
    finally:
        handle.remove()
    print(f"Saved {calls} observed router calls to {args.output}")


if __name__ == "__main__":
    main()
