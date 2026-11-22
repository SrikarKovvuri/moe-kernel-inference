"""Benchmark protocol: raw measurements, provenance, and explicit timing scope."""

import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import MoEConfig
from .reference import ExpertWeights, error_metrics, moe_oracle, moe_torch, validate_inputs
from .routing import occupancy, synthetic_routing


def _command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment(device):
    device = torch.device(device)
    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_command(["git", "status", "--porcelain"])),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    for package in ("triton", "triton-windows", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(gpu_name=props.name, gpu_memory_bytes=props.total_memory,
                      compute_capability=list(torch.cuda.get_device_capability(device)),
                      multiprocessors=props.multi_processor_count,
                      nvidia_smi=_command(["nvidia-smi", "--query-gpu=name,driver_version,power.limit",
                                           "--format=csv,noheader"]))
    return result


def summarize(samples):
    if not samples or any(not math.isfinite(v) or v < 0 for v in samples):
        raise ValueError("timing samples must be nonempty, finite, and nonnegative")
    ordered = sorted(samples)

    def quantile(q):
        pos = q * (len(ordered) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    return {"median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples),
            "p10_ms": quantile(0.1), "p90_ms": quantile(0.9),
            "min_ms": min(samples), "max_ms": max(samples), "samples_ms": samples}


@torch.inference_mode()
def measure(fn, *, device, warmup=20, repetitions=100):
    """Paired eager CUDA-event and synchronized host timings per invocation.

    One call per sample includes Python dispatch and allocations. Events can include
    stream idle gaps between host launches; they are not isolated kernel timings.
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
        # Initialize event resources outside the sample loop.
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
    result = {"wall": summarize(host), "warmup": warmup, "repetitions": repetitions,
              "mode": "eager", "wall_method": "perf_counter_ns; CUDA synchronized if applicable"}
    if gpu:
        result["cuda_event"] = summarize(gpu)
    return result


def write_json(path, value):
    """Atomic replacement prevents an interrupted experiment leaving partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_case(config, *, tokens, distribution, dtype, device, seed, backend, threshold,
             warmup, repetitions):
    routing = synthetic_routing(config, tokens, distribution, seed).to(device)
    weights = ExpertWeights.random(config, seed=seed + 1, dtype=dtype, device=device)
    x = torch.randn(tokens, config.hidden_size,
                    generator=torch.Generator().manual_seed(seed + 2)).to(device=device, dtype=dtype)
    validate_inputs(x, weights, routing)
    if backend == "torch":
        fn = lambda: moe_torch(x, weights, routing)
    else:
        from .triton_backend import moe_triton
        fn = lambda: moe_triton(x, weights, routing, threshold=threshold,
                               strategy=backend.removeprefix("triton_"))
    actual = fn()
    # FP32 expert baseline scales to full dimensions; independent token oracle is
    # additionally exercised on the tiny preset and in the unit tests.
    reference = moe_torch(x.float(), weights.to(dtype=torch.float32), routing)
    atol, rtol = (2e-5, 1e-4) if dtype == torch.float32 else (0.02, 0.03)
    correctness = error_metrics(actual, reference, atol=atol, rtol=rtol)
    correctness["reference"] = "FP32 expert-wise PyTorch"
    if config.hidden_size <= 64 and tokens <= 32:
        oracle = moe_oracle(x, weights, routing)
        correctness["independent_oracle"] = error_metrics(actual, oracle, atol=atol, rtol=rtol)
        correctness["passed"] &= correctness["independent_oracle"]["passed"]
    if not correctness["passed"]:
        raise RuntimeError(f"correctness failed for {backend}: {correctness}")
    del actual, reference
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings = measure(fn, device=device, warmup=warmup, repetitions=repetitions)
    result = {
        "config": asdict(config), "tokens": tokens,
        "batch_size": tokens, "context_size": 1, "workload_interpretation": "flattened tokens",
        "distribution": distribution, "seed": seed, "dtype": str(dtype),
        "backend": backend, "threshold": threshold, "weight_source": "synthetic random",
        "routing_source": "synthetic", "scope": "expert_block_including_dispatch",
        "excluded": ["router_linear", "softmax_topk", "attention", "normalization", "model_loading"],
        "occupancy": occupancy(routing, threshold), "correctness": correctness, "timing": timings,
        "weight_bytes": config.weight_bytes(torch.empty((), dtype=dtype).element_size()),
    }
    if torch.device(device).type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["tiny", "olmoe"], default="tiny")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--distributions", nargs="+", choices=["uniform", "balanced", "skewed"],
                        default=["uniform"])
    parser.add_argument("--backends", nargs="+",
                        choices=["torch", "triton_grouped", "triton_hybrid", "triton_fused"],
                        default=["torch"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark.json"))
    args = parser.parse_args(argv)
    if min(args.tokens) < 0 or args.threshold < 0 or args.warmup < 0 or args.repetitions < 1:
        parser.error("tokens/threshold/warmup must be nonnegative; repetitions must be positive")
    if args.threads < 1:
        parser.error("threads must be positive")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use --device cpu for logic validation")
    torch.set_num_threads(args.threads)
    config = MoEConfig.tiny() if args.preset == "tiny" else MoEConfig()
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "environment": environment(args.device), "arguments": vars(args) | {"output": str(args.output)},
              "results": [], "status": "running"}
    write_json(args.output, report)
    try:
        for distribution in args.distributions:
            for tokens in args.tokens:
                for backend in args.backends:
                    result = run_case(config, tokens=tokens, distribution=distribution,
                                      dtype=getattr(torch, args.dtype), device=args.device,
                                      seed=args.seed, backend=backend, threshold=args.threshold,
                                      warmup=args.warmup, repetitions=args.repetitions)
                    report["results"].append(result)
                    write_json(args.output, report)
                    print(f"{backend} tokens={tokens} {distribution}: "
                          f"{result['timing']['wall']['median_ms']:.4f} ms wall; correctness passed",
                          flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(args.output, report)
        raise
    report["status"] = "complete"
    write_json(args.output, report)
    return report


if __name__ == "__main__":
    main()
