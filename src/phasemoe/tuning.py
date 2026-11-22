"""Controlled threshold/tile sweep; improvements require a separate full-block rerun."""

import argparse
import random
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch

from .benchmark import environment, gpu_telemetry, measure, write_json
from .config import MoEConfig
from .reference import ExpertWeights, error_metrics, moe_torch, validate_inputs
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
    p.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument(
        "--distributions",
        nargs="+",
        choices=["uniform", "balanced", "skewed"],
        default=["uniform", "skewed"],
    )
    p.add_argument("--thresholds", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--combine-n", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--combine-k", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument(
        "--scope",
        choices=["down_combine", "expert_block_including_dispatch"],
        default="down_combine",
    )
    p.add_argument("--mode", choices=["eager", "cuda_graph"], default="cuda_graph")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repetitions", type=int, default=30)
    p.add_argument("--output", type=Path, default=Path("runs/tuning.json"))
    args = p.parse_args(argv)
    if min(args.tokens) < 1 or min(args.thresholds) < 0:
        p.error("tokens must be positive; thresholds nonnegative")
    if args.warmup < 0 or args.repetitions < 1:
        p.error("warmup must be nonnegative; repetitions positive")
    if not torch.cuda.is_available():
        p.error("tuning requires CUDA")
    configs = [
        replace(KernelConfig(), combine_n=n, combine_k=k)
        for n in args.combine_n
        for k in args.combine_k
    ]
    torch.set_num_threads(1)
    c = MoEConfig() if args.preset == "olmoe" else MoEConfig.tiny()
    weights = ExpertWeights.random(c, seed=args.seed + 1, device="cuda", dtype=torch.bfloat16)
    report = {
        "schema_version": 1,
        "kind": "threshold_tile_sweep",
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment("cuda"),
        "config": asdict(c),
        "dtype": "torch.bfloat16",
        "scope": args.scope,
        "seed": args.seed,
        "selection_warning": "Best-of-sweep is exploratory, not an unbiased speedup claim.",
        "arguments": vars(args) | {"output": str(args.output)},
        "cases": [],
    }
    write_json(args.output, report)
    try:
        with torch.inference_mode():
            for distribution in args.distributions:
                for tokens in args.tokens:
                    routing = synthetic_routing(c, tokens, distribution, args.seed).to("cuda")
                    x = torch.randn(
                        tokens,
                        c.hidden_size,
                        generator=torch.Generator().manual_seed(args.seed + 2),
                    )
                    x = x.cuda().bfloat16()
                    validate_inputs(x, weights, routing)
                    expected = moe_torch(x.float(), weights.to(dtype=torch.float32), routing)
                    default = KernelConfig()
                    packed, ends = prepare(routing, default)
                    hidden = project_gate_up(x, weights, routing, packed, ends, default)

                    def make_fn(kernel_config, threshold):
                        if args.scope == "expert_block_including_dispatch":
                            return partial(
                                moe_triton,
                                x,
                                weights,
                                routing,
                                threshold=threshold,
                                config=kernel_config,
                            )

                        def down_combine():
                            projected = project_down(
                                hidden, weights, routing, packed, ends, kernel_config, threshold
                            )
                            return combine(
                                hidden,
                                weights,
                                projected,
                                routing,
                                packed,
                                kernel_config,
                                threshold,
                            )

                        return down_combine

                    baseline = make_fn(default, 0)
                    baseline_check = error_metrics(baseline(), expected, atol=0.02, rtol=0.03)
                    if not baseline_check["passed"]:
                        raise RuntimeError(f"baseline correctness failed: {baseline_check}")
                    measure_args = dict(
                        device="cuda",
                        warmup=args.warmup,
                        repetitions=args.repetitions,
                        mode=args.mode,
                    )
                    case = {
                        "baseline_correctness": baseline_check,
                        "baseline_kernel_config": asdict(default),
                        "tokens": tokens,
                        "batch_size": tokens,
                        "context_size": 1,
                        "distribution": distribution,
                        "occupancy": occupancy(routing),
                        "telemetry_before": gpu_telemetry(),
                        "baseline_before": measure(baseline, **measure_args),
                        "candidates": [],
                    }
                    candidates = [
                        (cfg, threshold) for cfg in configs for threshold in args.thresholds
                    ]
                    random.Random(args.seed + tokens).shuffle(candidates)
                    for cfg, threshold in candidates:
                        fn = make_fn(cfg, threshold)
                        check = error_metrics(fn(), expected, atol=0.02, rtol=0.03)
                        if not check["passed"]:
                            raise RuntimeError(f"candidate correctness failed: {cfg}, {threshold}")
                        timing = measure(fn, **measure_args)
                        case["candidates"].append(
                            {
                                "kernel_config": asdict(cfg),
                                "threshold": threshold,
                                "correctness": check,
                                "timing": timing,
                            }
                        )
                    case["baseline_after"] = measure(baseline, **measure_args)
                    case["telemetry_after"] = gpu_telemetry()
                    before = case["baseline_before"]["cuda_event"]["median_ms"]
                    after = case["baseline_after"]["cuda_event"]["median_ms"]
                    case["baseline_drift_fraction"] = (after - before) / before if before else None
                    report["cases"].append(case)
                    write_json(args.output, report)
                    best = min(
                        case["candidates"], key=lambda row: row["timing"]["cuda_event"]["median_ms"]
                    )
                    print(
                        f"{distribution} T={tokens}: best threshold={best['threshold']} "
                        f"N={best['kernel_config']['combine_n']} "
                        f"K={best['kernel_config']['combine_k']} "
                        f"{best['timing']['cuda_event']['median_ms']:.4f} ms; "
                        f"baseline {before:.4f}/{after:.4f} ms",
                        flush=True,
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
