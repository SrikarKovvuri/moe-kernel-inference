"""Summarize completed development experiments without mixing timing scopes."""

import argparse
import json
import statistics
from pathlib import Path


def read(path):
    data = json.loads(Path(path).read_text())
    if data.get("status") != "complete":
        raise ValueError(f"incomplete experiment: {path}")
    return data


def ms(row):
    return row["timing"]["cuda_event"]["median_ms"]


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(v) for v in row) + " |" for row in rows),
        ]
    )


def ablation_summary(data):
    comparisons = [
        ("Fusion, default dispatch", "control_before", "selective_default"),
        ("Dispatch only, grouped", "control_before", "grouped_scan"),
        ("Fusion with scan dispatch", "grouped_scan", "selective_scan"),
        ("Fusion with wide combine tile", "grouped_wide", "selective_wide"),
    ]
    summaries = []
    for label, baseline, candidate in comparisons:
        for tokens in sorted({c["tokens"] for c in data["cases"]}):
            for distribution in sorted({c["distribution"] for c in data["cases"]}):
                ratios = []
                for case in data["cases"]:
                    if (case["tokens"], case["distribution"]) != (tokens, distribution):
                        continue
                    runs = {r["name"]: r["result"] for r in case["runs"]}
                    base = ms(runs[baseline])
                    if baseline == "control_before":
                        base = (base + ms(runs["control_after"])) / 2
                    ratios.append(base / ms(runs[candidate]))
                summaries.append(
                    dict(
                        comparison=label,
                        tokens=tokens,
                        distribution=distribution,
                        median=statistics.median(ratios),
                        low=min(ratios),
                        high=max(ratios),
                        seed_count=len(ratios),
                    )
                )
    return summaries


def plot_ablation(rows, output, mode):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(r["comparison"] for r in rows))
    distributions = list(dict.fromkeys(r["distribution"] for r in rows))
    fig, axes = plt.subplots(1, len(distributions), figsize=(12, 5.5), squeeze=False)
    for ax, distribution in zip(axes[0], distributions):
        for index, label in enumerate(labels):
            selected = [
                r for r in rows if r["distribution"] == distribution and r["comparison"] == label
            ]
            positions = [i + (index - 1.5) * 0.12 for i in range(len(selected))]
            values = [r["median"] for r in selected]
            errors = [
                [r["median"] - r["low"] for r in selected],
                [r["high"] - r["median"] for r in selected],
            ]
            ax.errorbar(positions, values, yerr=errors, fmt="o", capsize=3, label=label)
        ax.set_xticks(range(len(selected)), [str(r["tokens"]) for r in selected])
        ax.axhline(1, color="#777777", linestyle="--", linewidth=1)
        ax.set(title=distribution, xlabel="Tokens", ylabel="Matched baseline / candidate latency")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Full-block controls ({mode}); above 1 favors candidate")
    fig.text(
        0.5,
        0.02,
        "RTX 4060 Laptop, observed P8 cap; OLMoE dimensions, BF16; 20 warmups / 100 samples\n"
        "Points: median ratio across 3 seeds; bars: seed min–max, not confidence intervals.\n"
        "Dispatch gains are separated from selective-fusion gains.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, default=Path("reports"))
    args = p.parse_args()
    root = args.reports
    graph = read(root / "results/olmoe-bf16-graph.json")
    tuning = read(root / "results/tuning-down-combine.json")
    native = read(root / "results/native-profile.json")
    capture = read(root / "first-layer-routes/capture-manifest.json")
    grouped = {
        (r["tokens"], r["distribution"]): r
        for r in graph["results"]
        if r["backend"] == "triton_grouped"
    }
    ratios = [
        ms(grouped[r["tokens"], r["distribution"]]) / ms(r)
        for r in graph["results"]
        if r["backend"] == "triton_hybrid"
    ]
    text = [
        "# Follow-up experiments",
        "These experiments extend the initial midpoint evidence. All GPU measurements remain "
        "development results on a power/clock-constrained RTX 4060 Laptop (P8, 210 MHz SM / "
        "405 MHz memory observed), PyTorch 2.9.0+cu128, Triton 3.5.0, WSL2. No L40S/A6000, "
        "Nsight memory-counter, full-model quality, or production-backend speedup is claimed.",
        "## CUDA graphs",
        f"All {len(graph['results'])} full-shape BF16 cases passed correctness. "
        "Graph replay includes "
        "device dispatch and reuses captured allocations. There are 20 eager warmups, 20 capture "
        "warmups, one capture, 20 replay warmups and 100 measured replays per case. T is flattened "
        "token count (recorded batch T, context 1). Attention and router computation are excluded.",
        "The grouped/selective event-latency ratio ranges from "
        f"{min(ratios):.4f} to {max(ratios):.4f} "
        "across 15 token/distribution cases. This does not establish a consistent fusion benefit. "
        "Eager and graph results are separate experiments; their ratios are not fusion speedups.",
        "[Raw graph data](results/olmoe-bf16-graph.json) · [Graph plots](figures/graph)",
        "## Threshold and tile search",
        f"The sweep tested {sum(len(c['candidates']) for c in tuning['cases'])} "
        "correct candidates: "
        "thresholds 0/1/2/4, combine columns 16/32/64 and reduction chunks 64/128/256, for "
        "T=1/8/32 and uniform/skewed routing. These are isolated down/combine graph timings with "
        "precomputed activations and metadata, 10 warmups per phase and 30 measurements. Their "
        "cache/allocation conditions differ from a complete block. "
        "Best-of-sweep selection is exploratory.",
    ]
    rows = []
    for c in tuning["cases"]:
        best = min(c["candidates"], key=ms)
        rows.append(
            [
                c["distribution"],
                c["tokens"],
                best["threshold"],
                f"{best['kernel_config']['combine_n']} × {best['kernel_config']['combine_k']}",
                f"{ms(best):.4f}",
                f"{c['baseline_before']['cuda_event']['median_ms']:.4f}",
                f"{100 * c['baseline_drift_fraction']:+.2f}%",
            ]
        )
    text += [
        table(
            [
                "Routes",
                "T",
                "Selected threshold",
                "N × K",
                "Best ms",
                "Baseline before ms",
                "Baseline drift",
            ],
            rows,
        ),
        "Threshold zero wins some cases. Small selected differences do not justify adopting a "
        "new default; the skewed one-token case also showed about 4% baseline drift. "
        "[Raw sweep](results/tuning-down-combine.json)",
    ]
    for mode in ("graph", "eager"):
        path = root / f"results/ablation-{mode}.json"
        if not path.exists():
            continue
        data = read(path)
        summaries = ablation_summary(data)
        drift = max(abs(c["baseline_drift_fraction"]) for c in data["cases"])
        text += [
            f"## Independent full-block controls: {mode}",
            f"{len(data['cases'])} cases use fresh seeds 11/22/33, T=1/8/32, uniform/skewed "
            "routes, randomized case and candidate order, and default grouped controls before "
            "and after each case. Each variant has 20 warmups and 100 samples. "
            f"Maximum absolute baseline drift was {100 * drift:.2f}%. Ratios use matching seeds; "
            "the default control is the mean of its before/after medians. Bars below are ranges "
            "over three seed cases, not confidence intervals. Repeated invocations are not "
            "treated as independent trials.",
            table(
                ["Comparison", "Routes", "T", "Median ratio", "Seed range"],
                [
                    [
                        r["comparison"],
                        r["distribution"],
                        r["tokens"],
                        f"{r['median']:.4f}",
                        f"{r['low']:.4f}–{r['high']:.4f}",
                    ]
                    for r in summaries
                ],
            ),
            f"![Matched full-block comparisons](figures/ablation-{mode}.png)",
            f"[Raw controls](results/ablation-{mode}.json)",
        ]
        plot_ablation(summaries, root / f"figures/ablation-{mode}", mode)
    cpu = native["cpu_operator_counts"]
    text += [
        "## Native PyTorch grouped-mm execution",
        f"The T=8 uniform BF16 profile contains {cpu['aten::_grouped_mm']} native grouped-mm "
        f"calls, {cpu['aten::mm']} individual matrix-multiply operator calls, and device-to-host "
        "offset copies. This is evidence of the host-loop fallback on this software/GPU, "
        "not execution of a fast single grouped kernel. The trace follows 20 warmups and "
        "records one invocation. Profiler durations are not used as benchmark latency. "
        "[Profile summary](results/native-profile.json) · [Trace](results/native-trace.json)",
        "## Pretrained first-layer occupancy",
        f"A pinned OLMoE checkpoint ({capture['manifest']['revision']}) supplied "
        f"{capture['manifest']['tensor_bytes'] / 2**20:.2f} MiB of embedding, attention, "
        "normalization and router tensors. Later layers and expert MLP weights cannot affect "
        "the first router call, so the prefix can capture exact first-layer prefill dependencies "
        "without loading the full model. CPU BF16 capture uses upstream eager causal attention "
        "and FP32 rotary frequencies. Tiny random-model tests include exact equality with "
        "upstream pretrained-loading behavior.",
        "The three authored repository probes are a narrow diagnostic sample, not a public-corpus "
        "or generation evaluation. Routes preserve FP32 probabilities before the model's BF16 "
        "cast. Replay with random expert weights remains distinct from pretrained block execution.",
        table(
            ["Probe", "Context tokens", "Active experts", "Singletons", "Maximum occupancy"],
            [
                [
                    t["provenance"]["prompt_name"],
                    t["provenance"]["context_tokens"],
                    t["occupancy"]["active_experts"],
                    t["occupancy"]["singleton_experts"],
                    t["occupancy"]["max_occupancy"],
                ]
                for t in capture["traces"]
            ],
        ),
        "[Capture manifest and routes](first-layer-routes) · "
        "[Authored probes](../experiments/text-probes.json)",
    ]
    public_path = root / "wikitext-routes/capture-manifest.json"
    if public_path.exists():
        public = read(public_path)
        text += [
            "## Public-text routing probes",
            "Three deterministic long paragraphs from a pinned, tokenized WikiText-2 test "
            "split were truncated to 128 tokens. Capture uses the same pretrained layer-0 "
            "prefix and CPU BF16 policy. This is a narrow diagnostic sample, not corpus-wide "
            "performance or generation evaluation. Source URL, commit, original line, file "
            "hash, prompt hash and token hash are retained without redistributing the text.",
            table(
                ["Source line", "Tokens", "Active experts", "Singletons", "Maximum occupancy"],
                [
                    [
                        t["provenance"]["source_metadata"]["source_line"],
                        t["provenance"]["context_tokens"],
                        t["occupancy"]["active_experts"],
                        t["occupancy"]["singleton_experts"],
                        t["occupancy"]["max_occupancy"],
                    ]
                    for t in public["traces"]
                ],
            ),
            "[Public-text capture evidence](wikitext-routes) · "
            "[WikiText attribution and license](https://huggingface.co/datasets/Salesforce/wikitext)",
        ]
    pretrained_path = root / "results/pretrained-block.json"
    if pretrained_path.exists():
        pretrained = read(pretrained_path)
        rows = pretrained["results"]
        maximum = max(r["correctness"]["relative_l2_error"] for r in rows)
        hf_maximum = max(r["upstream_bf16_comparison"]["relative_l2_error"] for r in rows)
        text += [
            "## Pretrained expert-block validation",
            f"All {len(rows)} BF16 cases passed both the FP32 reference and upstream BF16 "
            "module comparison at atol 0.02 / rtol 0.03. Unlike route replay, these cases "
            "use actual checkpoint expert weights and actual hidden states from the first "
            "attention/norm prefix. The maximum relative L2 errors were "
            f"{maximum:.5f} versus FP32 and {hf_maximum:.5f} versus upstream BF16.",
            "Graph timings use 20 warmups per phase and 100 samples, default sort dispatch, "
            "threshold 1 and a 32 × 128 combine tile. They include expert dispatch but exclude "
            "attention, normalization and router computation. T=1/8 rows are suffix subsets "
            "of prefill hidden states, not decode. Full-prompt cases contain 87–93 tokens. "
            "The three authored prompts do not establish model quality or later-layer behavior.",
            table(
                ["Prompt", "T", "Phase", "Backend", "CUDA-event median ms"],
                [
                    [r["prompt"], r["tokens"], r["phase"], r["backend"], f"{ms(r):.4f}"]
                    for r in rows
                ],
            ),
            "[Pretrained block evidence and tensor checksums](results/pretrained-block.json)",
            "The pretrained artifact preserves a dirty-working-tree flag from concurrent "
            "report archiving. A [source audit](results/pretrained-block-provenance.md) verifies "
            "that the measured benchmark and kernel files match the recorded revision.",
        ]
    text += [
        "## Interpretation",
        "The measured fusion hypothesis remains unproven. Keep the grouped baseline and existing "
        "defaults, report dispatch separately, and rerun on the proposed cluster GPU with stable "
        "power/clocks. Broader real-route sampling, a competitive external grouped kernel, "
        "memory counters and full-model quality evaluation remain open work.",
    ]
    (root / "followup-analysis.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
