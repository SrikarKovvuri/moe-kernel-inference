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
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_command(["git", "status", "--porcelain"])),
        "matmul_fp32_precision": (
            torch.backends.cuda.matmul.fp32_precision
            if hasattr(torch.backends.cuda.matmul, "fp32_precision")
            else ("tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee")
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    for package in ("triton", "triton-windows", "numpy"):
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
def measure(fn, *, device, warmup=20, repetitions=100, mode="eager"):
    """Paired eager CUDA-event and synchronized host timings per invocation.

    One call per sample includes Python dispatch and allocations. Events can include
    stream idle gaps between host launches; they are not isolated kernel timings.
    """
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be nonnegative and repetitions positive")
    device = torch.device(device)
    if mode not in ("eager", "cuda_graph"):
        raise ValueError("timing mode must be eager or cuda_graph")
    if mode == "cuda_graph" and device.type != "cuda":
        raise ValueError("CUDA graph timing requires a CUDA device")
    for _ in range(warmup):
        fn()
    use_cuda = device.type == "cuda"
    captured_output = None
    timed_fn = fn  # Retain the original callable and its input tensors through replay.
    if mode == "cuda_graph":
        # Compile/initialize on a side stream before capture, as required by PyTorch.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(max(3, warmup)):
                fn()
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_output = fn()
        timed_fn = graph.replay
        for _ in range(warmup):
            timed_fn()
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
        timed_fn()
        if use_cuda:
            end.record(torch.cuda.current_stream(device))
            end.synchronize()
        host.append((time.perf_counter_ns() - begin) / 1e6)
        if use_cuda:
            gpu.append(start.elapsed_time(end))
    result = {
        "wall": summarize(host),
        "warmup": warmup,
        "eager_warmup_calls": warmup,
        "capture_warmup_calls": max(3, warmup) if mode == "cuda_graph" else 0,
        "graph_warmup_replays": warmup if mode == "cuda_graph" else 0,
        "capture_calls": 1 if mode == "cuda_graph" else 0,
        "repetitions": repetitions,
        "mode": mode,
        "wall_method": "perf_counter_ns; CUDA synchronized if applicable",
        "allocation_policy": "graph pool reused" if mode == "cuda_graph" else "per invocation",
    }
    # Keep captured output storage alive until all graph replays have completed.
    del captured_output
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


def run_case(
    config,
    *,
    tokens,
    distribution,
    dtype,
    device,
    seed,
    backend,
    threshold,
    warmup,
    repetitions,
    mode="eager",
    trace_path=None,
):
    provenance = None
    if trace_path is None:
        routing = synthetic_routing(config, tokens, distribution, seed).to(device)
    else:
        from .traces import load_trace

        routing, provenance = load_trace(trace_path)
        if (routing.num_experts, routing.top_k, routing.num_tokens) != (
            config.num_experts,
            config.top_k,
            tokens,
        ):
            raise ValueError("trace dimensions must match the preset and requested tokens")
        routing = routing.to(device)
    weights = ExpertWeights.random(config, seed=seed + 1, dtype=dtype, device=device)
    x = torch.randn(
        tokens, config.hidden_size, generator=torch.Generator().manual_seed(seed + 2)
    ).to(device=device, dtype=dtype)
    validate_inputs(x, weights, routing)
    if backend == "torch":
        fn = partial(moe_torch, x, weights, routing)
    else:
        from .triton_backend import moe_triton

        fn = partial(
            moe_triton,
            x,
            weights,
            routing,
            threshold=threshold,
            strategy=backend.removeprefix("triton_"),
        )
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
    telemetry_before = gpu_telemetry() if torch.device(device).type == "cuda" else None
    timings = measure(fn, device=device, warmup=warmup, repetitions=repetitions, mode=mode)
    result = {
        "config": asdict(config),
        "tokens": tokens,
        "batch_size": tokens,
        "context_size": 1,
        "workload_interpretation": "flattened tokens",
        "distribution": distribution,
        "seed": seed,
        "dtype": str(dtype),
        "backend": backend,
        "threshold": threshold,
        "effective_threshold": 0
        if backend == "triton_grouped"
        else (tokens if backend == "triton_fused" else threshold),
        "weight_source": "synthetic random",
        "routing_source": provenance["source"] if provenance else "synthetic",
        "routing_provenance": provenance,
        "scope": "expert_block_including_dispatch",
        "excluded": [
            "router_linear",
            "softmax_topk",
            "attention",
            "normalization",
            "model_loading",
        ],
        "occupancy": occupancy(routing, threshold),
        "correctness": correctness,
        "timing": timings,
        "weight_bytes": config.weight_bytes(torch.empty((), dtype=dtype).element_size()),
    }
    if torch.device(device).type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["gpu_telemetry_before"] = telemetry_before
        result["gpu_telemetry_after"] = gpu_telemetry()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["tiny", "olmoe"], default="tiny")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument(
        "--distributions", nargs="+", choices=["uniform", "balanced", "skewed"], default=["uniform"]
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["torch", "triton_grouped", "triton_hybrid", "triton_fused"],
        default=["torch"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument(
        "--routing-trace",
        type=Path,
        help="replay saved routes; overrides --tokens and --distributions",
    )
    parser.add_argument("--mode", choices=["eager", "cuda_graph"], default="eager")
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
    if args.mode == "cuda_graph" and (
        torch.device(args.device).type != "cuda" or "torch" in args.backends
    ):
        parser.error("CUDA graphs require CUDA and Triton backends (torch uses dynamic nonzero)")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use --device cpu for logic validation")
    torch.set_num_threads(args.threads)
    config = MoEConfig.tiny() if args.preset == "tiny" else MoEConfig()
    if args.routing_trace:
        from .traces import load_trace

        saved_routes, _ = load_trace(args.routing_trace)
        if (saved_routes.num_experts, saved_routes.top_k) != (config.num_experts, config.top_k):
            parser.error("trace expert count/top_k do not match the selected preset")
        args.tokens = [saved_routes.num_tokens]
        args.distributions = ["trace"]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment(args.device),
        "arguments": vars(args)
        | {
            "output": str(args.output),
            "routing_trace": str(args.routing_trace) if args.routing_trace else None,
        },
        "results": [],
        "status": "running",
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
                        device=args.device,
                        seed=args.seed,
                        backend=backend,
                        threshold=args.threshold,
                        warmup=args.warmup,
                        repetitions=args.repetitions,
                        mode=args.mode,
                        trace_path=args.routing_trace,
                    )
                    report["results"].append(result)
                    write_json(args.output, report)
                    print(
                        f"{backend} tokens={tokens} {distribution}: "
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
