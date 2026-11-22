"""Generate publication-style figures from recorded benchmark artifacts."""

import argparse
import json
from pathlib import Path


def load_report(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("status") != "complete":
        raise ValueError("plots require a completed schema-version-1 benchmark")
    if not data.get("results"):
        raise ValueError("benchmark has no results")
    if any(not r["correctness"]["passed"] for r in data["results"]):
        raise ValueError("refusing to plot a failed correctness case")
    # A chart must not silently combine incomparable shapes, dtypes or timing scopes.
    signatures = {
        (
            json.dumps(r["config"], sort_keys=True),
            r["dtype"],
            r["scope"],
            r["timing"]["mode"],
            r["seed"],
            r["threshold"],
            r["timing"]["warmup"],
            r["timing"]["repetitions"],
        )
        for r in data["results"]
    }
    if len(signatures) != 1:
        raise ValueError("split mixed shapes/dtypes/scopes/timing modes into separate reports")
    return data


def make_plots(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = report["results"]
    first = rows[0]
    device = report["environment"].get("gpu_name", report["environment"]["processor"] or "CPU")
    date = report["created_at_utc"][:10]
    mode = first["timing"]["mode"]
    route_source = first.get("routing_source", "synthetic")
    caption = (
        f"{device} | {first['dtype']} | {mode} | "
        f"warmup={first['timing']['warmup']}, repetitions={first['timing']['repetitions']}\n"
        f"Synthetic weights, {route_source} routes; expert block incl. dispatch; source {date}, "
        f"commit {str(report['environment']['git_commit'])[:7]}"
    )
    distributions = list(dict.fromkeys(r["distribution"] for r in rows))
    colors = {
        "torch": "#777777",
        "triton_grouped": "#304E78",
        "triton_hybrid": "#B15B35",
        "triton_fused": "#567D64",
    }
    labels = {
        "torch": "PyTorch expert loop",
        "triton_grouped": "Triton grouped",
        "triton_hybrid": f"Selective (threshold {first['threshold']})",
        "triton_fused": "Uniformly fused",
    }
    for metric in ("wall", "cuda_event"):
        if metric not in first["timing"]:
            continue
        fig, axes = plt.subplots(
            1, len(distributions), figsize=(5 * len(distributions), 4.9), squeeze=False
        )
        for ax, distribution in zip(axes[0], distributions):
            selected = [r for r in rows if r["distribution"] == distribution]
            for backend in dict.fromkeys(r["backend"] for r in selected):
                series = sorted(
                    (r for r in selected if r["backend"] == backend), key=lambda r: r["tokens"]
                )
                xs = [r["tokens"] for r in series]
                ys = [r["timing"][metric]["median_ms"] for r in series]
                low = [r["timing"][metric]["p10_ms"] for r in series]
                high = [r["timing"][metric]["p90_ms"] for r in series]
                ax.plot(xs, ys, "o-", label=labels.get(backend, backend), color=colors.get(backend))
                ax.fill_between(xs, low, high, alpha=0.12, color=colors.get(backend))
            ax.set(xlabel="Tokens (count)", ylabel="Median latency (ms)", title=distribution)
            ax.set_xscale("symlog", linthresh=1)
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        title = "Synchronized wall latency" if metric == "wall" else "CUDA-event stream latency"
        fig.suptitle(f"{title} by token count (bands: p10–p90)", fontsize=13)
        fig.text(0.5, 0.02, caption, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.13, 1, 0.94))
        fig.savefig(output / f"latency-{metric}.png", dpi=180)
        fig.savefig(output / f"latency-{metric}.pdf")
        plt.close(fig)

    # Occupancy is a property of routes, independent of backend. Select one row per case.
    unique = {}
    for row in rows:
        unique.setdefault((row["distribution"], row["tokens"]), row)
    fig, axes = plt.subplots(
        1, len(distributions), figsize=(5 * len(distributions), 4.9), squeeze=False
    )
    for ax, distribution in zip(axes[0], distributions):
        series = sorted(
            (r for (d, _), r in unique.items() if d == distribution), key=lambda r: r["tokens"]
        )
        positions = np.arange(len(series))
        zero = np.array([r["occupancy"]["empty_experts"] for r in series])
        one = np.array([r["occupancy"]["singleton_experts"] for r in series])
        many = first["config"]["num_experts"] - zero - one
        ax.bar(positions, zero, label="0 tokens", color="#D3D7DE")
        ax.bar(positions, one, bottom=zero, label="1 token", color="#B15B35")
        ax.bar(positions, many, bottom=zero + one, label="2+ tokens", color="#304E78")
        ax.set_xticks(positions, [str(r["tokens"]) for r in series])
        ax.set(xlabel="Tokens in batch (count)", ylabel="Experts (count)", title=distribution)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Expert occupancy ({route_source} routes)", fontsize=13)
    fig.text(
        0.5,
        0.02,
        f"E={first['config']['num_experts']}, top-k={first['config']['top_k']}, "
        f"seed={first['seed']} | route source: {route_source}\n"
        f"Source: recorded benchmark {date}",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(output / "occupancy.png", dpi=180)
    fig.savefig(output / "occupancy.pdf")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, default=Path("runs/figures"))
    args = p.parse_args(argv)
    make_plots(load_report(args.input), args.output)


if __name__ == "__main__":
    main()
