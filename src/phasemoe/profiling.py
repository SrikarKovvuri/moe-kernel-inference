"""Basic grouped-baseline stage timing and optional PyTorch profiler trace."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch

from .benchmark import environment, gpu_telemetry, measure, write_json
from .config import MoEConfig
from .grouped import CONFIG, moe_grouped, prepare, project_down, project_gate_up, reduce_output
from .reference import ExpertWeights, error_metrics, moe_torch, validate_inputs
from .routing import occupancy, synthetic_routing


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["tiny", "olmoe"], default="olmoe")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--distribution", choices=["uniform", "balanced", "skewed"], default="uniform")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repetitions", type=int, default=100)
    p.add_argument("--output", type=Path, default=Path("runs/baseline-profile.json"))
    p.add_argument("--trace", type=Path, help="optional Chrome trace output")
    args = p.parse_args(argv)
    if args.tokens < 1 or args.warmup < 0 or args.repetitions < 1:
        p.error("tokens/repetitions must be positive; warmup nonnegative")
    if not torch.cuda.is_available():
        p.error("profiling requires CUDA")
    torch.set_num_threads(1)
    c = MoEConfig() if args.preset == "olmoe" else MoEConfig.tiny()
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
    with torch.inference_mode():
        check = error_metrics(
            moe_grouped(x, w, r),
            moe_torch(x.float(), w.to(dtype=torch.float32), r),
            atol=0.02,
            rtol=0.03,
        )
        if not check["passed"]:
            raise RuntimeError(f"baseline correctness failed: {check}")
        packed, ends = prepare(r)
        hidden = project_gate_up(x, w, r, packed, ends)
        projected = project_down(hidden, w, r, packed, ends)
        stages = {
            "dispatch": partial(prepare, r),
            "gate_up_activation": partial(project_gate_up, x, w, r, packed, ends),
            "down_projection": partial(project_down, hidden, w, r, packed, ends),
            "output_reduction": partial(reduce_output, projected, r, packed),
            "full_expert_block": partial(moe_grouped, x, w, r),
        }
        report = {
            "schema_version": 1,
            "kind": "baseline_profile",
            "status": "running",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": environment("cuda"),
            "config": asdict(c),
            "kernel_config": asdict(CONFIG),
            "tokens": args.tokens,
            "batch_size": args.tokens,
            "context_size": 1,
            "dtype": "torch.bfloat16",
            "backend": "triton_grouped",
            "distribution": args.distribution,
            "seed": args.seed,
            "occupancy": occupancy(r),
            "correctness": check,
            "note": (
                "Synthetic inputs/routes/weights. Isolated stages have precomputed inputs; "
                "do not sum their medians into block latency."
            ),
            "excluded": ["router", "attention", "normalization", "model_loading"],
            "stages": {},
        }
        write_json(args.output, report)
        try:
            for name, fn in stages.items():
                report["stages"][name] = {
                    "gpu_telemetry_before": gpu_telemetry(),
                    **measure(fn, device="cuda", warmup=args.warmup, repetitions=args.repetitions),
                    "gpu_telemetry_after": gpu_telemetry(),
                }
                write_json(args.output, report)
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
                    with torch.profiler.record_function("phasemoe.grouped_baseline"):
                        moe_grouped(x, w, r)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.trace))
                report["trace_path"] = str(args.trace)
                report["profiler_table"] = prof.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=20
                )
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_json(args.output, report)
            raise
        report["status"] = "complete"
        write_json(args.output, report)
    return report


if __name__ == "__main__":
    main()
