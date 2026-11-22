"""First-layer expert-block validation with pretrained weights and actual prefix states.

No full-model generation or quality claim. Attention/router execution is outside
timing. Small token subsets are selected from prefill, not called decode workloads.
"""

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch

from phasemoe.benchmark import environment, gpu_telemetry, measure, write_json
from phasemoe.hf_adapter import config_from_hf, routing_from_hf, weights_from_hf
from phasemoe.pretrained_prefix import FirstLayerRouter, fetch_prefix_weights
from phasemoe.reference import error_metrics, moe_torch, validate_inputs
from phasemoe.routing import occupancy
from phasemoe.torch_grouped import moe_torch_grouped
from phasemoe.triton_backend import KernelConfig, moe_triton


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--revision", default="6d84c48581ece794365f2b8e9cfb043c68ade9c5")
    p.add_argument("--probes", type=Path, default=Path("experiments/text-probes.json"))
    p.add_argument("--cache", type=Path, default=Path.home() / ".cache/phasemoe/prefix")
    p.add_argument("--output", type=Path, default=Path("runs/pretrained-block.json"))
    p.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=[1, 8, 0],
        help="prefix-state suffix sizes; 0 means the whole prompt",
    )
    p.add_argument("--mode", choices=["eager", "cuda_graph"], default="cuda_graph")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--backends",
        nargs="+",
        choices=["torch", "torch_grouped", "triton_grouped", "triton_hybrid", "triton_fused"],
        default=["triton_grouped", "triton_hybrid", "triton_fused"],
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repetitions", type=int, default=100)
    p.add_argument("--download-only", action="store_true")
    args = p.parse_args()
    if min(args.tokens) < 0 or args.warmup < 0 or args.repetitions < 1:
        p.error("tokens/warmup must be nonnegative; repetitions positive")
    if args.mode == "cuda_graph" and any(not b.startswith("triton_") for b in args.backends):
        p.error("graph mode requires Triton backends")
    device = torch.device(args.device)
    if (
        device.type != "cuda"
        and (args.mode == "cuda_graph" or any(b.startswith("triton_") for b in args.backends))
        and not args.download_only
    ):
        p.error("CPU validation requires eager timing and PyTorch backends")
    from huggingface_hub import model_info
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

    torch.set_num_threads(1)
    revision = model_info(args.model, revision=args.revision).sha
    config = AutoConfig.from_pretrained(args.model, revision=revision, trust_remote_code=False)
    prefix = FirstLayerRouter(config).to_inference()
    tensors, prefix_manifest = fetch_prefix_weights(
        args.model, revision, [prefix.checkpoint_key(n) for n in prefix.state_dict()], args.cache
    )
    prefix.load_state_dict({n: tensors[prefix.checkpoint_key(n)] for n in prefix.state_dict()})
    del tensors
    with torch.device("meta"):
        module = OlmoeSparseMoeBlock(config)
    names = ["model.layers.0.mlp." + n for n in module.state_dict()]
    tensors, expert_manifest = fetch_prefix_weights(
        args.model, revision, names, args.cache, cache_name="layer-00-experts"
    )
    if args.download_only:
        print(f"Cached {expert_manifest['tensor_bytes'] / 2**20:.2f} MiB of expert tensors")
        return
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("block benchmark requires CUDA")
    module.load_state_dict(
        {n: tensors["model.layers.0.mlp." + n] for n in module.state_dict()}, assign=True
    )
    del tensors
    module = module.to(device).eval()
    prefix = prefix.to(device)
    weights = weights_from_hf(module)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=revision, trust_remote_code=False
    )
    report = dict(
        schema_version=1,
        kind="pretrained_first_expert_block",
        status="running",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        environment=environment(device),
        model=args.model,
        model_revision=revision,
        layer=0,
        dtype="torch.bfloat16",
        scope="expert_block_including_dispatch",
        excluded=["attention", "router_logits_topk", "normalization"],
        config=asdict(config_from_hf(module)),
        prefix_manifest=prefix_manifest,
        expert_manifest=expert_manifest,
        probes_sha256=hashlib.sha256(args.probes.read_bytes()).hexdigest(),
        results=[],
        limitations=(
            "Three authored probes, layer 0, prefill or prefill-state subsets; "
            "no generation/quality test."
        ),
    )
    write_json(args.output, report)
    rng = random.Random(2026)
    try:
        with torch.inference_mode():
            for probe in json.loads(args.probes.read_text()):
                ids = tokenizer(probe["text"], return_tensors="pt").input_ids.to(device)
                all_hidden = prefix.hidden_states(ids)
                for requested in args.tokens:
                    if requested > all_hidden.shape[0]:
                        raise ValueError("requested subset exceeds the prompt length")
                    x = (
                        all_hidden[-requested:].contiguous()
                        if requested
                        else all_hidden.contiguous()
                    )
                    routing = routing_from_hf(module, x)
                    validate_inputs(x, weights, routing)
                    hf_output = module(x.unsqueeze(0))[0].squeeze(0)
                    reference = moe_torch(x.float(), weights.to(dtype=torch.float32), routing)
                    backends = list(args.backends)
                    rng.shuffle(backends)
                    for backend in backends:
                        if backend == "torch":
                            fn = partial(moe_torch, x, weights, routing)
                        elif backend == "torch_grouped":
                            fn = partial(moe_torch_grouped, x, weights, routing)
                        else:
                            fn = partial(
                                moe_triton,
                                x,
                                weights,
                                routing,
                                strategy=backend.removeprefix("triton_"),
                                threshold=1,
                            )
                        actual = fn()
                        check = error_metrics(actual, reference, atol=0.02, rtol=0.03)
                        hf_check = error_metrics(actual, hf_output, atol=0.02, rtol=0.03)
                        row = dict(
                            prompt=probe["name"],
                            prompt_sha256=hashlib.sha256(probe["text"].encode()).hexdigest(),
                            prompt_source=probe.get("source", "authored repository text probe"),
                            context_tokens=ids.numel(),
                            tokens=x.shape[0],
                            token_positions=[all_hidden.shape[0] - x.shape[0], all_hidden.shape[0]],
                            phase="prefill" if not requested else "prefill_state_subset",
                            backend=backend,
                            kernel_config=asdict(KernelConfig())
                            if backend.startswith("triton_")
                            else None,
                            threshold=1,
                            occupancy=occupancy(routing),
                            correctness=check,
                            upstream_bf16_comparison=hf_check,
                            gpu_telemetry_before=gpu_telemetry() if device.type == "cuda" else None,
                        )
                        report["results"].append(row)
                        write_json(args.output, report)
                        if not check["passed"] or not hf_check["passed"]:
                            raise RuntimeError(
                                f"{backend} pretrained correctness failed: {check}; HF: {hf_check}"
                            )
                        row["timing"] = measure(
                            fn,
                            device=device,
                            warmup=args.warmup,
                            repetitions=args.repetitions,
                            mode=args.mode,
                        )
                        row["gpu_telemetry_after"] = (
                            gpu_telemetry() if device.type == "cuda" else None
                        )
                        write_json(args.output, report)
                        print(
                            f"{probe['name']} T={x.shape[0]} {backend}: "
                            f"{row['timing']['wall']['median_ms']:.4f} ms wall",
                            flush=True,
                        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(args.output, report)
        raise
    report["status"] = "complete"
    write_json(args.output, report)


if __name__ == "__main__":
    main()
