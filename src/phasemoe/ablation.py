"""Randomized full-block controls with independent seeds and baseline drift checks."""

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from .benchmark import environment, run_case, write_json
from .config import MoEConfig
from .triton_backend import KernelConfig


def load_variants(path):
    variants = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(variants, list) or not variants:
        raise ValueError("variants must be a nonempty JSON list")
    names = set()
    result = []
    for item in variants:
        if set(item) - {"name", "strategy", "threshold", "kernel_config"}:
            raise ValueError("unknown variant field")
        name = item["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("variant names must be nonempty and unique")
        names.add(name)
        strategy = item.get("strategy", "hybrid")
        threshold = item.get("threshold", 1)
        if strategy not in ("grouped", "hybrid", "fused"):
            raise ValueError("invalid variant strategy")
        if type(threshold) is not int or threshold < 0:
            raise ValueError("threshold must be a nonnegative integer")
        config = KernelConfig(**item.get("kernel_config", {}))
        result.append(dict(name=name, strategy=strategy, threshold=threshold, kernel_config=config))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", type=Path, required=True)
    p.add_argument("--preset", choices=["tiny", "olmoe"], default="olmoe")
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument(
        "--distributions",
        nargs="+",
        choices=["uniform", "balanced", "skewed"],
        default=["uniform", "skewed"],
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    p.add_argument("--order-seed", type=int, default=2026)
    p.add_argument("--mode", choices=["eager", "cuda_graph"], default="cuda_graph")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repetitions", type=int, default=100)
    p.add_argument("--output", type=Path, default=Path("runs/ablation.json"))
    args = p.parse_args(argv)
    if min(args.tokens) < 1 or args.warmup < 0 or args.repetitions < 1:
        p.error("tokens/repetitions must be positive; warmup nonnegative")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("seeds must be distinct")
    if not torch.cuda.is_available():
        p.error("ablation requires CUDA")
    variants = load_variants(args.variants)
    torch.set_num_threads(1)
    config = MoEConfig() if args.preset == "olmoe" else MoEConfig.tiny()
    report = {
        "schema_version": 1,
        "kind": "randomized_full_block_ablation",
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment("cuda"),
        "arguments": vars(args) | {"variants": str(args.variants), "output": str(args.output)},
        "variants": [v | {"kernel_config": asdict(v["kernel_config"])} for v in variants],
        "protocol": (
            "Each seed uses identical generated inputs/routes/weights for all variants. "
            "Cases and candidate order are randomized. Default grouped controls bracket "
            "each case. Raw invocations are repeated measurements, not independent trials."
        ),
        "cases": [],
    }
    rng = random.Random(args.order_seed)
    cases = [(t, d, s) for t in args.tokens for d in args.distributions for s in args.seeds]
    rng.shuffle(cases)
    write_json(args.output, report)
    try:
        for tokens, distribution, seed in cases:
            case = dict(
                tokens=tokens, distribution=distribution, seed=seed, status="running", runs=[]
            )
            report["cases"].append(case)
            ordered = list(variants)
            rng.shuffle(ordered)
            control = dict(strategy="grouped", threshold=0, kernel_config=KernelConfig())
            ordered = [
                control | {"name": "control_before"},
                *ordered,
                control | {"name": "control_after"},
            ]
            for variant in ordered:
                result = run_case(
                    config,
                    tokens=tokens,
                    distribution=distribution,
                    seed=seed,
                    backend="triton_" + variant["strategy"],
                    threshold=variant["threshold"],
                    kernel_config=variant["kernel_config"],
                    dtype=torch.bfloat16,
                    device="cuda",
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    mode=args.mode,
                )
                case["runs"].append(dict(name=variant["name"], result=result))
                write_json(args.output, report)
                print(
                    f"{distribution} T={tokens} seed={seed} {variant['name']}: "
                    f"{result['timing']['cuda_event']['median_ms']:.4f} ms",
                    flush=True,
                )
            before = case["runs"][0]["result"]["timing"]["cuda_event"]["median_ms"]
            after = case["runs"][-1]["result"]["timing"]["cuda_event"]["median_ms"]
            case.update(status="complete", baseline_drift_fraction=(after - before) / before)
            write_json(args.output, report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(args.output, report)
        raise
    report["status"] = "complete"
    write_json(args.output, report)
    return report


if __name__ == "__main__":
    main()
