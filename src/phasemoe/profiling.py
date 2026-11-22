"""Stage timing and optional PyTorch CUDA trace for an OLMoE expert block."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch

from .benchmark import environment, measure, write_json
from .config import MoEConfig
from .reference import ExpertWeights, validate_inputs
from .routing import occupancy, synthetic_routing
from .triton_backend import (
    KernelConfig,
    combine,
    moe_triton,
    prepare,
    project_down,
    project_gate_up,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["tiny", "olmoe"], default="olmoe")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--distribution", choices=["uniform", "balanced", "skewed"], default="uniform")
    p.add_argument("--threshold", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repetitions", type=int, default=100)
    p.add_argument("--mode", choices=["eager", "cuda_graph"], default="eager")
    p.add_argument("--output", type=Path, default=Path("runs/profile.json"))
    p.add_argument("--trace", type=Path, help="optional Chrome trace output (.json)")
    args = p.parse_args(argv)
    if args.tokens < 1 or args.threshold < 0 or args.warmup < 0 or args.repetitions < 1:
        p.error("tokens/repetitions must be positive; threshold/warmup nonnegative")
    if not torch.cuda.is_available():
        p.error("profiling requires a CUDA GPU")
    torch.set_num_threads(1)
    c = MoEConfig() if args.preset == "olmoe" else MoEConfig.tiny()
    k = KernelConfig()
    r = synthetic_routing(c, args.tokens, args.distribution, args.seed).to("cuda")
    w = ExpertWeights.random(c, seed=args.seed + 1, device="cuda", dtype=torch.bfloat16)
    x = (
        torch.randn(
            args.tokens, c.hidden_size, generator=torch.Generator().manual_seed(args.seed + 2)
        )
        .cuda()
        .bfloat16()
    )
    validate_inputs(x, w, r)
    packed, ends = prepare(r, k)
    hidden = project_gate_up(x, w, r, packed, ends, k)
    projected = project_down(hidden, w, r, packed, ends, k, args.threshold)
    stages = {
        "dispatch": partial(prepare, r, k),
        "gate_up_activation": partial(project_gate_up, x, w, r, packed, ends, k),
        "heavy_down": partial(project_down, hidden, w, r, packed, ends, k, args.threshold),
        "selective_combine": partial(combine, hidden, w, projected, r, packed, k, args.threshold),
        "full_expert_block": partial(moe_triton, x, w, r, threshold=args.threshold),
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment("cuda"),
        "config": asdict(c),
        "kernel_config": asdict(k),
        "tokens": args.tokens,
        "batch_size": args.tokens,
        "context_size": 1,
        "dtype": "torch.bfloat16",
        "backend": "triton_hybrid",
        "threshold": args.threshold,
        "distribution": args.distribution,
        "seed": args.seed,
        "occupancy": occupancy(r, args.threshold),
        "note": ("Isolated stages use precomputed inputs; their sum is not end-to-end latency."),
        "stages": {},
    }
    with torch.inference_mode():
        for name, fn in stages.items():
            report["stages"][name] = measure(
                fn, device="cuda", warmup=args.warmup, repetitions=args.repetitions, mode=args.mode
            )
            print(name, report["stages"][name]["wall"]["median_ms"], flush=True)
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof:
                with torch.profiler.record_function("phasemoe.full_expert_block"):
                    moe_triton(x, w, r, threshold=args.threshold)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.trace))
            report["trace_path"] = str(args.trace)
            report["profiler_table"] = prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=30
            )
    write_json(args.output, report)
    return report


if __name__ == "__main__":
    main()
