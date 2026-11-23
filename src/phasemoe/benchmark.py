"""Benchmark protocol: raw measurements, provenance, and explicit timing scope."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch

from .config import MoEConfig
from .reference import ExpertWeights, error_metrics, moe_oracle, moe_torch, validate_inputs
from .routing import occupancy, synthetic_routing


def _command(args):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment(device):
    device = torch.device(device)
    status = _command(["git", "status", "--porcelain"])
    source_dir = Path(__file__).resolve().parent
    source_hashes = {
        f"phasemoe/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_dir.glob("*.py"))
    }
    entrypoint = Path(sys.argv[0])
    if entrypoint.is_file():
        source_hashes[f"entrypoint/{entrypoint.name}"] = hashlib.sha256(
            entrypoint.read_bytes()
        ).hexdigest()
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu_model,
        )
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(status),
        "git_status": status,
        "source_sha256": source_hashes,
        "matmul_fp32_precision": (
            torch.backends.cuda.matmul.fp32_precision
            if hasattr(torch.backends.cuda.matmul, "fp32_precision")
            else ("tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee")
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    for package in (
        "triton",
        "triton-windows",
        "numpy",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(
            gpu_name=props.name,
            gpu_memory_bytes=props.total_memory,
            compute_capability=list(torch.cuda.get_device_capability(device)),
            multiprocessors=props.multi_processor_count,
            nvidia_smi=_command(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,power.limit",
                    "--format=csv,noheader",
                ]
            ),
        )
    return result


def gpu_telemetry():
    """Best-effort snapshots, outside timing; not a continuous power/clock trace."""
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": "name,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu",
        "values": _command(
            [
                "nvidia-smi",
                "--query-gpu=name,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
    }


def summarize(samples):
    if not samples or any(not math.isfinite(v) or v < 0 for v in samples):
        raise ValueError("timing samples must be nonempty, finite, and nonnegative")
    ordered = sorted(samples)

    def quantile(q):
        pos = q * (len(ordered) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p10_ms": quantile(0.1),
        "p90_ms": quantile(0.9),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


@torch.inference_mode()
def measure(fn, *, device, warmup=20, repetitions=100):
    """Eager calls with paired CUDA-event and synchronized wall-clock samples.

    Allocation and Python launch work are included. Events may include idle gaps
    between launches; these are not isolated kernel durations.
    """
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be nonnegative and repetitions positive")
    device = torch.device(device)
    for _ in range(warmup):
        fn()
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(device))
        end.record(torch.cuda.current_stream(device))
        end.synchronize()
    host, gpu = [], []
    for _ in range(repetitions):
        begin = time.perf_counter_ns()
        if use_cuda:
            start.record(torch.cuda.current_stream(device))
        fn()
        if use_cuda:
            end.record(torch.cuda.current_stream(device))
            end.synchronize()
        host.append((time.perf_counter_ns() - begin) / 1e6)
        if use_cuda:
            gpu.append(start.elapsed_time(end))
    result = {
        "wall": summarize(host),
        "warmup": warmup,
        "repetitions": repetitions,
        "mode": "eager",
        "allocation_policy": "per invocation",
        "wall_method": "perf_counter_ns; CUDA synchronized if applicable",
    }
    if gpu:
        result["cuda_event"] = summarize(gpu)
    return result


def write_json(path, value):
    """Replace atomically so interrupted runs retain valid completed rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_case(config, *, tokens, distribution, dtype, device, seed, backend, warmup, repetitions):
    routing = synthetic_routing(config, tokens, distribution, seed).to(device)
    weights = ExpertWeights.random(config, seed=seed + 1, dtype=dtype, device=device)
    x = torch.randn(
        tokens, config.hidden_size, generator=torch.Generator().manual_seed(seed + 2)
    ).to(device=device, dtype=dtype)
    validate_inputs(x, weights, routing)
    kernel_config = None
    if backend == "torch":
        fn = partial(moe_torch, x, weights, routing)
    elif backend == "triton_grouped":
        from .grouped import CONFIG, moe_grouped

        kernel_config = asdict(CONFIG)
        fn = partial(moe_grouped, x, weights, routing)
    else:
        raise ValueError(f"unknown baseline: {backend}")
    actual = fn()
    reference = moe_torch(x.float(), weights.to(dtype=torch.float32), routing)
    atol, rtol = (2e-5, 1e-4) if dtype == torch.float32 else (0.02, 0.03)
    correctness = error_metrics(actual, reference, atol=atol, rtol=rtol)
    correctness["reference"] = "FP32 expert-wise PyTorch"
    if config.hidden_size <= 64 and tokens <= 32:
        correctness["independent_oracle"] = error_metrics(
            actual, moe_oracle(x, weights, routing), atol=atol, rtol=rtol
        )
        correctness["passed"] &= correctness["independent_oracle"]["passed"]
    if not correctness["passed"]:
        raise RuntimeError(f"correctness failed for {backend}: {correctness}")
    del actual, reference
    cuda = torch.device(device).type == "cuda"
    telemetry_before = gpu_telemetry() if cuda else None
    timing = measure(fn, device=device, warmup=warmup, repetitions=repetitions)
    result = {
        "config": asdict(config),
        "tokens": tokens,
        "batch_size": tokens,
        "context_size": 1,
        "workload_interpretation": "synthetic flattened tokens, not model prefill or decode",
        "distribution": distribution,
        "seed": seed,
        "dtype": str(dtype),
        "backend": backend,
        "kernel_config": kernel_config,
        "weight_source": "synthetic random",
        "routing_source": "synthetic",
        "scope": "expert_block_including_dispatch",
        "excluded": [
            "router_linear",
            "softmax_topk",
            "attention",
            "normalization",
            "model_loading",
        ],
        "occupancy": occupancy(routing),
        "correctness": correctness,
        "timing": timing,
        "weight_bytes": config.weight_bytes(torch.empty((), dtype=dtype).element_size()),
    }
    if cuda:
        result.update(gpu_telemetry_before=telemetry_before, gpu_telemetry_after=gpu_telemetry())
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["tiny", "olmoe"], default="tiny")
    p.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument(
        "--distributions", nargs="+", choices=["uniform", "balanced", "skewed"], default=["uniform"]
    )
    p.add_argument("--backends", nargs="+", choices=["torch", "triton_grouped"], default=["torch"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repetitions", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("runs/baseline.json"))
    args = p.parse_args(argv)
    if min(args.tokens) < 0 or args.warmup < 0 or args.repetitions < 1 or args.threads < 1:
        p.error("tokens/warmup must be nonnegative; repetitions/threads positive")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        p.error("baseline timing supports CPU or CUDA")
    if "triton_grouped" in args.backends and device.type != "cuda":
        p.error("triton_grouped requires CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable; use --device cpu --backends torch for logic validation")
    torch.set_num_threads(args.threads)
    config = MoEConfig.tiny() if args.preset == "tiny" else MoEConfig()
    report = {
        "schema_version": 1,
        "kind": "baseline_benchmark",
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment(device),
        "arguments": vars(args) | {"output": str(args.output)},
        "results": [],
    }
    write_json(args.output, report)
    try:
        for distribution in args.distributions:
            for tokens in args.tokens:
                for backend in args.backends:
                    result = run_case(
                        config,
                        tokens=tokens,
                        distribution=distribution,
                        dtype=getattr(torch, args.dtype),
                        device=device,
                        seed=args.seed,
                        backend=backend,
                        warmup=args.warmup,
                        repetitions=args.repetitions,
                    )
                    report["results"].append(result)
                    write_json(args.output, report)
                    print(
                        f"{backend} T={tokens} {distribution}: "
                        f"{result['timing']['wall']['median_ms']:.4f} ms wall; correctness passed",
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
