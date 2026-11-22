"""Capture pretrained first-layer prefill routes without downloading expert weights.

Uses authored, public repository probes by default. This is not a public-corpus
evaluation, generation run, or a representative sample of later-layer routing.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from phasemoe.benchmark import environment, write_json
from phasemoe.pretrained_prefix import FirstLayerRouter, fetch_prefix_weights
from phasemoe.routing import occupancy
from phasemoe.traces import routing_from_logits, save_trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--revision", default="6d84c48581ece794365f2b8e9cfb043c68ade9c5")
    p.add_argument("--probes", type=Path, default=Path("experiments/text-probes.json"))
    p.add_argument("--cache", type=Path, default=Path.home() / ".cache/phasemoe/prefix")
    p.add_argument("--output", type=Path, default=Path("runs/first-layer-routes"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--download-only", action="store_true")
    args = p.parse_args()
    if args.max_tokens < 1:
        p.error("max-tokens must be positive")
    from huggingface_hub import model_info
    from transformers import AutoConfig, AutoTokenizer

    torch.set_num_threads(1)
    revision = model_info(args.model, revision=args.revision).sha
    config = AutoConfig.from_pretrained(args.model, revision=revision, trust_remote_code=False)
    if config.model_type != "olmoe":
        p.error("this prefix only supports OLMoE")
    prefix = FirstLayerRouter(config).to_inference()
    names = [prefix.checkpoint_key(name) for name in prefix.state_dict()]
    tensors, manifest = fetch_prefix_weights(args.model, revision, names, args.cache)
    if args.download_only:
        print(f"Cached {manifest['tensor_bytes'] / 2**20:.2f} MiB of pretrained prefix tensors")
        return
    prefix.load_state_dict(
        {name: tensors[prefix.checkpoint_key(name)] for name in prefix.state_dict()}
    )
    del tensors
    prefix.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=revision, trust_remote_code=False
    )
    probes = json.loads(args.probes.read_text())
    report = dict(
        schema_version=1,
        kind="pretrained_first_layer_routes",
        status="running",
        environment=environment(args.device),
        manifest=manifest,
        traces=[],
        scope="first-layer router prefix only; no expert MLPs or generation",
    )
    for index, probe in enumerate(probes):
        tokens = tokenizer(
            probe["text"], return_tensors="pt", truncation=True, max_length=args.max_tokens
        ).input_ids.to(args.device)
        logits = prefix(tokens)
        routing = routing_from_logits(
            logits, top_k=config.num_experts_per_tok, norm_topk_prob=config.norm_topk_prob
        )
        path = args.output / f"probe-{index:02d}-prefill.json"
        provenance = dict(
            source="pretrained_first_layer_prefix",
            model=args.model,
            model_revision=revision,
            layer=0,
            phase="prefill",
            prompt_name=probe["name"],
            prompt_sha256=hashlib.sha256(probe["text"].encode()).hexdigest(),
            token_ids_sha256=hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
            prompt_source="authored repository text probe; not a representative corpus",
            context_tokens=tokens.numel(),
            batch_size=1,
            max_tokens=args.max_tokens,
            weight_dtype="bfloat16",
            routing_weight_dtype="float32 before model cast",
            attention_backend="upstream eager causal attention",
            device=args.device,
            norm_topk_prob=config.norm_topk_prob,
            captured_at_utc=datetime.now(timezone.utc).isoformat(),
            scope="embeddings, first attention/residual/norm and router; no expert MLPs",
            tensor_manifest_sha256=hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest(),
        )
        save_trace(path, routing, provenance=provenance)
        report["traces"].append(
            dict(path=str(path), provenance=provenance, occupancy=occupancy(routing))
        )
        print(f"{probe['name']}: {routing.num_tokens} tokens; {occupancy(routing)}", flush=True)
        write_json(args.output / "capture-manifest.json", report)
    report["status"] = "complete"
    write_json(args.output / "capture-manifest.json", report)


if __name__ == "__main__":
    main()
