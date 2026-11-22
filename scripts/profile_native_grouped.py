"""Inspect actual native grouped-mm execution, including possible host-loop fallback."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from phasemoe.benchmark import environment, gpu_telemetry, write_json
from phasemoe.config import MoEConfig
from phasemoe.reference import ExpertWeights, error_metrics, moe_torch, validate_inputs
from phasemoe.routing import occupancy, synthetic_routing
from phasemoe.torch_grouped import moe_torch_grouped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("runs/native-profile.json"))
    p.add_argument("--trace", type=Path, default=Path("runs/native-trace.json"))
    args = p.parse_args()
    if args.tokens < 1 or not torch.cuda.is_available():
        p.error("positive tokens and CUDA required")
    torch.set_num_threads(1)
    config = MoEConfig()
    weights = ExpertWeights.random(config, seed=args.seed + 1, dtype=torch.bfloat16, device="cuda")
    routing = synthetic_routing(config, args.tokens, seed=args.seed).to("cuda")
    x = (
        torch.randn(
            args.tokens, config.hidden_size, generator=torch.Generator().manual_seed(args.seed + 2)
        )
        .cuda()
        .bfloat16()
    )
    validate_inputs(x, weights, routing)
    with torch.inference_mode():
        expected = moe_torch(x.float(), weights.to(dtype=torch.float32), routing)
        actual = moe_torch_grouped(x, weights, routing)
        check = error_metrics(actual, expected, atol=0.02, rtol=0.03)
        if not check["passed"]:
            raise RuntimeError(f"correctness failed: {check}")
        for _ in range(20):
            moe_torch_grouped(x, weights, routing)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            moe_torch_grouped(x, weights, routing)
            torch.cuda.synchronize()
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.trace))
    cpu = {item.key: item.count for item in prof.key_averages() if item.key.startswith("aten::")}
    gpu = Counter(
        item.name for item in prof.events() if item.device_type == torch.autograd.DeviceType.CUDA
    )
    report = dict(
        schema_version=1,
        kind="native_grouped_execution_profile",
        status="complete",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        environment=environment("cuda"),
        config=asdict(config),
        tokens=args.tokens,
        batch_size=args.tokens,
        context_size=1,
        dtype="torch.bfloat16",
        seed=args.seed,
        distribution="uniform",
        warmup=20,
        repetitions=1,
        scope="expert_block_including_dispatch",
        correctness=check,
        occupancy=occupancy(routing),
        cpu_operator_counts=cpu,
        gpu_kernel_counts=dict(gpu),
        trace_path=str(args.trace),
        gpu_telemetry=gpu_telemetry(),
        interpretation=(
            "Profiler execution evidence only; profiler timings are not benchmark latency."
        ),
    )
    write_json(args.output, report)
    print(cpu)
    print(dict(gpu))


if __name__ == "__main__":
    main()
