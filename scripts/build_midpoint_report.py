"""Build an evidence-backed report from a completed benchmark and profile artifact.

Run with a Python environment containing python-docx. For Codex document work, use
the bundled workspace runtime, then render with the Documents skill renderer.
"""

import argparse
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

REPO = "https://github.com/SrikarKovvuri/moe-kernel-inference"


def add_prose(document, text):
    paragraph = document.add_paragraph()
    cursor = 0
    for match in re.finditer(r"https?://\S+", text):
        paragraph.add_run(text[cursor : match.start()])
        url = match.group()
        link = OxmlElement("w:hyperlink")
        link.set(
            qn("r:id"), paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
        )
        run = OxmlElement("w:r")
        properties = OxmlElement("w:rPr")
        style = OxmlElement("w:rStyle")
        style.set(qn("w:val"), "Hyperlink")
        properties.append(style)
        run.append(properties)
        content = OxmlElement("w:t")
        content.text = url
        run.append(content)
        link.append(run)
        paragraph._p.append(link)
        cursor = match.end()
    paragraph.add_run(text[cursor:])
    return paragraph


def add_table(document, headers, rows, widths):
    table = document.add_table(rows=1, cols=len(headers))
    table.autofit = False
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    for cell, text, width in zip(table.rows[0].cells, headers, widths):
        cell.text = text
        cell.width = Inches(width)
    repeat = OxmlElement("w:tblHeader")
    table.rows[0]._tr.get_or_add_trPr().append(repeat)
    for values in rows:
        for cell, text, width in zip(table.add_row().cells, values, widths):
            cell.text = str(text)
            cell.width = Inches(width)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement("w:" + side)
        for key, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
            border.set(qn("w:" + key), value)
        borders.append(border)
    table._tbl.tblPr.append(borders)
    for index, row in enumerate(table.rows):
        for col, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            props = cell._tc.get_or_add_tcPr()
            fill = OxmlElement("w:shd")
            fill.set(
                qn("w:fill"), "304E78" if index == 0 else ("F3F5F7" if index % 2 else "FFFFFF")
            )
            props.append(fill)
            margins = OxmlElement("w:tcMar")
            for side in ("top", "left", "bottom", "right"):
                margin = OxmlElement("w:" + side)
                margin.set(qn("w:w"), "90")
                margin.set(qn("w:type"), "dxa")
                margins.append(margin)
            props.append(margins)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(1)
                paragraph.paragraph_format.space_before = Pt(1)
                paragraph.paragraph_format.line_spacing = 1.05
                paragraph.paragraph_format.keep_with_next = index < len(table.rows) - 1
                if col > 0:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.font.bold = index == 0
                    run.font.color.rgb = RGBColor.from_string("FFFFFF" if index == 0 else "000000")
    document.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def build(benchmark, profile, output):
    data = json.loads(Path(benchmark).read_text(encoding="utf-8"))
    prof = json.loads(Path(profile).read_text(encoding="utf-8"))
    if data["status"] != "complete" or any(not r["correctness"]["passed"] for r in data["results"]):
        raise ValueError("report requires completed, correct benchmark evidence")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = data["results"]
    env = data["environment"]
    uniform = {(r["tokens"], r["backend"]): r for r in rows if r["distribution"] == "uniform"}
    grouped = {
        (r["distribution"], r["tokens"]): r for r in rows if r["backend"] == "triton_grouped"
    }
    ratios = [
        grouped[r["distribution"], r["tokens"]]["timing"]["cuda_event"]["median_ms"]
        / r["timing"]["cuda_event"]["median_ms"]
        for r in rows
        if r["backend"] == "triton_hybrid"
    ]
    maximum_error = max(r["correctness"]["relative_l2_error"] for r in rows)
    stamp = data["created_at_utc"][:10]
    blocks = [
        (
            "Project summary",
            [
                "PhaseMoE investigates whether expert occupancy can guide down-projection execution "
                "in a Mixture-of-Experts layer. Light experts use a fused down-projection and output "
                "combination kernel; heavy experts retain grouped GEMM. The hypothesis is that avoided "
                "intermediate writes and launches can outweigh the extra scheduling and reduction costs.",
                "At this midpoint, we have a working PyTorch reference, grouped Triton baseline, "
                "selective and uniformly fused implementations, correctness tests, and reproducible "
                "single-block benchmarks. The initial measurements do not establish a consistent "
                "selective-fusion speedup. They are development evidence from a power-constrained "
                "laptop GPU, not results on the proposed L40S or A6000 cluster hardware.",
                "The workload uses OLMoE-1B-7B dimensions: 64 experts, top-8 routing, hidden width "
                "2048 and intermediate width 1024. We use synthetic random weights and controlled "
                "uniform, balanced and skewed routing. No pretrained checkpoint or public-text routing "
                "trace was used for the reported measurements.",
            ],
        ),
        (
            "Baseline and experimental setup",
            [
                f"Measured hardware: {env['gpu_name']}, {env['gpu_memory_bytes'] / 2**30:.2f} GiB "
                f"reported device memory, compute capability 8.9 and 24 SMs. Software: Python "
                f"{env['python']}, PyTorch {env['torch']}, Triton {env['triton']}, CUDA runtime "
                f"{env['cuda_runtime']}, NVIDIA driver 621.14, Linux under WSL2. "
                "Host CPU: Intel Core Ultra 9 185H. PyTorch uses one CPU thread.",
                "The GPU was observed in P8 during sustained load, at 210 MHz SM and 405 MHz memory "
                "clocks, with a 15 W current power limit and software power/thermal caps reported. "
                "This is a material limitation on performance interpretation; a diagnostic snapshot "
                "is archived with the results. Clocks were not locked and Nsight memory counters were not collected.",
                "Every case uses BF16 inputs, weights and intermediate storage, with FP32 dot/reduction "
                "accumulation. Routing probabilities use FP32 softmax and are not renormalized after "
                "top-k selection. Seeds are 0 for routing, 1 for weights and 2 for inputs. The expert "
                "weight tensors occupy 768 MiB. Token counts are 1, 8, 32, 128 and 512; the recorded "
                "batch/context interpretation is T by 1 flattened tokens. Large batches are prefill-like "
                "shape tests, not end-to-end prompt-prefill measurements.",
                "For each backend and case, correctness and JIT compilation precede 20 warmup calls "
                "and 100 measured eager invocations. CUDA events measure stream elapsed time; synchronized "
                "perf_counter timing measures wall latency. Each sample includes dispatch/occupancy "
                "metadata, gate/up projections, SiLU multiplication, down projection and output reduction, "
                "including allocation and launch overhead. Router linear/top-k selection, attention, "
                "normalization and model loading are excluded. Raw samples and p10/p90 values are retained.",
                "The PyTorch baseline launches separate GEMMs per active expert and is not called a "
                "grouped-GEMM backend. The controlled comparison uses our grouped Triton implementation "
                "with identical gate/up work and changes only down/combine execution. This implementation "
                "has not been compared with a production vLLM or MegaBlocks backend.",
            ],
        ),
        (
            "Profiling and baseline analysis",
            [
                f"All {len(rows)} full-shape BF16 cases passed comparison with an FP32 PyTorch reference "
                f"using atol 0.02 and rtol 0.03. The largest relative L2 error was {maximum_error:.5f} "
                f"({maximum_error * 100:.3f}%). Independent token-wise tests additionally cover small shapes. "
                "Small random-weight cases were checked against the upstream Hugging Face OLMoE module "
                "in FP32 and BF16; this is integration validation, not pretrained-model evaluation.",
                f"Across all 15 routing/token configurations, the grouped-to-selective CUDA-event latency "
                f"ratio ranged from {min(ratios):.3f} to {max(ratios):.3f} at occupancy threshold 1. "
                "Values above 1 favor selective fusion. This narrow range around parity, plus the "
                "constrained hardware and fixed backend order, does not support a robust speedup claim. "
                "Uniform fusion becomes substantially slower for larger batches.",
                "For the uniform 8-token workload, 26 experts are singletons. Profiling shows that "
                "gate/up/activation is the largest phase. The measurements below are isolated-stage "
                "timings with precomputed inputs; they are not additive estimates of end-to-end latency. "
                "A separate PyTorch profiler trace confirms the projection-heavy execution pattern.",
                "The traffic opportunity is small relative to weights in this case. Analytically, "
                "avoiding one BF16 projected row write and read saves 8 KiB per singleton assignment "
                "at hidden width 2048, while that expert's down weights occupy 4 MiB. For 26 singleton "
                "assignments, the avoided intermediate traffic is 208 KiB, versus 104 MiB of singleton "
                "down weights. These are tensor-size estimates, not measured DRAM traffic or cache misses.",
            ],
        ),
        (
            "First optimization and implementation status",
            [
                "The selective kernel is implemented and GPU-tested. It gives each token/output tile "
                "one writer, computes light-expert down projections in chunks, and combines them with "
                "precomputed heavy-expert rows using the routing weights. Single-writer reduction "
                "avoids output atomics. Tests cover empty batches, mixed occupancy, thresholds 0/1/2/4, "
                "unaligned dimensions, and FP32/FP16/BF16.",
                "The prototype avoids projected-row traffic for light experts but still reserves the "
                "full projected-output buffer in hybrid mode. It does not demonstrate an allocation "
                "reduction. The direct reduction can lose weight reuse and Tensor Core efficiency. "
                "CUDA-graph timing, phase profiling, and a randomized threshold/tile search are available "
                "as separate experiments; their results must not be mixed with the eager latency table.",
            ],
        ),
        (
            "Future work and scope adjustments",
            [
                "First, rerun the exact benchmark on an L40S or A6000 with stable power and clock "
                "conditions, recording repeated trials and both eager and graph timing. Compare with "
                "a stronger external grouped-GEMM backend before claiming a general performance benefit.",
                "Second, use isolated down/combine profiling to tune occupancy thresholds 1, 2 and 4 "
                "and combine tile sizes, then validate chosen settings on separate full-block runs and "
                "new seeds. Include the extra metadata/scheduling cost. Keep threshold 0 as a control.",
                "Third, capture real prefill and decode routing traces from pretrained OLMoE on public "
                "text using the supplied capture script, replay the observed occupancies, and investigate "
                "whether their workload regimes differ from synthetic routing. Full-model latency, "
                "H100 evaluation and broad threshold generalization remain stretch goals.",
                "The scope remains a single expert block with an explicit numerical policy. We have "
                "not established complete transformer-layer or end-to-end language-model speedups, "
                "quality preservation, measured memory-traffic reduction, or target-cluster performance.",
            ],
        ),
        (
            "Blockers and limitations",
            [
                "The initial Windows environment was CPU-only and had insufficient free C-drive "
                "space for CUDA PyTorch. We resolved local GPU execution by creating an isolated "
                "Linux/WSL environment backed by D-drive storage. The remaining power/clock constraint "
                "is documented rather than treated as normal GPU performance.",
                "Cluster execution and pretrained routing collection remain unverified. No external "
                "experiment-tracking dashboard is connected. Team UNIs, confirmed role assignments "
                "and the dashboard URL must be completed before submission. The supplied template "
                "mentions four to five students, while the proposal lists three; confirm the approved "
                "team-size arrangement with the course.",
            ],
        ),
        (
            "Project links and references",
            [
                f"Repository: {REPO}",
                f"Versioned experiment archive: {REPO}/tree/main/reports/results",
                "Experiment tracking dashboard: pending team-provided accessible URL. The archive "
                "contains JSON evidence and figures; it is not a claim that a W&B dashboard exists.",
                f"Initial benchmark source revision: {data['environment']['git_commit']}. "
                f"Profile source revision: {prof['environment']['git_commit']}. "
                f"Measurement date: {stamp}.",
                "OLMoE model configuration: https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/main/config.json",
                "OLMoE paper: https://arxiv.org/abs/2409.02060",
                "Locality-aware MoE kernels: https://pytorch.org/blog/accelerating-moe-model/",
                "Triton grouped GEMM tutorial: https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html",
            ],
        ),
    ]
    capture_path = output / "first-layer-routes/capture-manifest.json"
    public_path = output / "wikitext-routes/capture-manifest.json"
    route_rows = []
    if capture_path.exists() and public_path.exists():
        for source, path in (("Authored", capture_path), ("WikiText-2", public_path)):
            capture = json.loads(path.read_text())
            if capture["status"] != "complete":
                raise ValueError("routing evidence must be complete")
            for index, trace in enumerate(capture["traces"], 1):
                route_rows.append(
                    [
                        f"{source} {index}",
                        str(trace["provenance"]["context_tokens"]),
                        str(trace["occupancy"]["singleton_experts"]),
                        str(trace["occupancy"]["max_occupancy"]),
                    ]
                )
        revised = []
        for heading, paragraphs in blocks:
            if heading == "Project summary":
                paragraphs[-1] = (
                    "The initial table uses OLMoE-1B-7B dimensions: 64 experts, top-8 routing, "
                    "hidden width 2048 and intermediate width 1024, with random weights and "
                    "synthetic routes. Follow-up experiments separately identify pretrained "
                    "first-layer routing and checkpoint-backed numerical validation."
                )
            if heading == "Future work and scope adjustments":
                followup = [
                    "Follow-up experiments include 45 full-shape CUDA-graph cases and 216 isolated "
                    "down/combine candidates (thresholds 0/1/2/4; combine N=16/32/64, K=64/128/256). "
                    "The sweep uses 10 warmups per phase and 30 samples; full-block graph runs use "
                    "20 warmups per phase and 100 replays. All use the same constrained GPU and BF16 "
                    "dimensions. Isolated timings and graph replay are not mixed with the eager table.",
                    "Independent full-block controls use seeds 11/22/33, T=1/8/32 and uniform/skewed "
                    "routes, randomized order, and grouped controls before/after each case. The 126 "
                    "graph measurements show fusion near parity. A separate two-kernel routing packer "
                    "reduces one-token graph latency by about 1.7%; this is a dispatch result, not "
                    "a fusion gain. Eager controls and raw samples are recorded separately.",
                    "A small exact prefix captures pretrained layer-0 prefill routes using 228.77 MiB "
                    "of embedding/attention/norm/router tensors. Six probes comprise three authored "
                    "texts (87–93 tokens) and three pinned WikiText-2 test paragraphs (128 tokens). "
                    "They have 1–6 singleton experts. This is a narrow first-layer diagnostic sample, "
                    "not a generation run or corpus-wide evaluation.",
                ]
                pretrained_path = output / "results/pretrained-block.json"
                if pretrained_path.exists():
                    pretrained = json.loads(pretrained_path.read_text())
                    if pretrained["status"] != "complete":
                        raise ValueError("pretrained evidence must be complete")
                    p_rows = pretrained["results"]
                    error = max(r["correctness"]["relative_l2_error"] for r in p_rows)
                    followup.append(
                        f"All {len(p_rows)} checkpoint-backed BF16 GPU cases pass both FP32 and upstream "
                        f"BF16 comparisons (atol 0.02, rtol 0.03); maximum relative L2 error versus "
                        f"FP32 is {error:.5f}. They use actual first-layer expert weights and prefix "
                        "hidden states, 20 warmups per graph phase and 100 replays. T=1/8 are prefill "
                        "state subsets; full prompts have 87–93 tokens. Attention/router work is "
                        "excluded. This is not end-to-end model latency or a quality evaluation."
                    )
                revised.append(("Follow-up validation", followup))
                paragraphs = [
                    "Rerun on an L40S or A6000 with stable power/clocks and a competitive external "
                    "grouped-GEMM backend. The native PyTorch grouped-mm API was profiled here and "
                    "uses a host-loop fallback (3 grouped calls expand to 192 matrix-multiply calls).",
                    "Extend real-route collection to more prompts, later layers and actual decode. "
                    "Collect Nsight memory counters and test model quality before any broader claim. "
                    "The current scope remains one expert block; full-model and H100 evaluation are "
                    "stretch goals. Existing defaults remain unchanged because fusion gains are unproven.",
                ]
            if heading == "Blockers and limitations":
                paragraphs[1] = paragraphs[1].replace(
                    "Cluster execution and pretrained routing collection remain unverified.",
                    "Cluster execution, later-layer/decode tracing and model quality remain unverified.",
                )
            if heading == "Project links and references":
                paragraphs.insert(
                    2,
                    f"Follow-up methods and evidence: {REPO}/blob/main/reports/followup-analysis.md",
                )
            revised.append((heading, paragraphs))
        blocks = revised
    latency_rows = []
    for tokens in sorted({key[0] for key in uniform}):
        latency_rows.append(
            [str(tokens)]
            + [
                f"{uniform[tokens, backend]['timing']['cuda_event']['median_ms']:.3f}"
                for backend in ("torch", "triton_grouped", "triton_hybrid", "triton_fused")
            ]
        )
    profile_rows = [
        [
            name.replace("_", " "),
            f"{value['cuda_event']['median_ms']:.3f}",
            f"{value['wall']['median_ms']:.3f}",
        ]
        for name, value in prof["stages"].items()
    ]

    document = Document()
    sec = document.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin = sec.bottom_margin = Inches(0.7)
    sec.left_margin = sec.right_margin = Inches(0.75)
    normal = document.styles["Normal"]
    normal.font.name, normal.font.size = "Calibri", Pt(11)
    normal.paragraph_format.line_spacing = 1.08
    normal.paragraph_format.space_after = Pt(7)
    for name in ("Title", "Subtitle", "Heading 1", "Heading 2"):
        document.styles[name].font.color.rgb = RGBColor(0, 0, 0)
    for style in document.styles:
        for border in style.element.xpath(".//w:pBdr"):
            border.getparent().remove(border)
    document.styles["Title"].font.size = Pt(24)
    document.styles["Heading 1"].font.size = Pt(15)
    document.add_paragraph("PhaseMoE Midpoint Report", "Title")
    document.add_paragraph(
        "Expert Selective Fusion for Efficient Mixture of Experts Inference", "Subtitle"
    )
    document.add_paragraph("Aedin Pereira · Srikar Kovvuri · David Samtani")
    document.add_paragraph(
        "Draft for team review | November 22 2026\nUNIs and confirmed roles pending team input"
    )
    md = [
        "# PhaseMoE midpoint report",
        "",
        "**Expert-Selective Fusion for Efficient Mixture-of-Experts Inference**",
        "",
        "Aedin Pereira, Srikar Kovvuri, David Samtani.",
        "",
        "Draft for team review — November 22, 2026. UNIs and confirmed roles pending team input.",
        "",
    ]
    for heading, paragraphs in blocks:
        document.add_heading(heading, level=1)
        md.extend([f"## {heading}", ""])
        for paragraph in paragraphs:
            add_prose(document, paragraph)
            md.extend([paragraph, ""])
        if heading == "Profiling and baseline analysis":
            document.add_heading("Uniform routing latency snapshot", level=2)
            caption = (
                "Median CUDA-event latency in ms; OLMoE dimensions; BF16; RTX 4060 Laptop "
                "under observed P8 caps; eager; 20 warmups and 100 samples. Threshold 1."
            )
            document.add_paragraph(caption).paragraph_format.keep_with_next = True
            headers = ["Tokens", "PyTorch", "Grouped", "Selective", "Uniform fused"]
            add_table(document, headers, latency_rows, [0.6, 1.1, 1.1, 1.1, 1.3])
            md.extend(
                [
                    "### Uniform routing latency snapshot",
                    "",
                    caption,
                    "",
                    "| " + " | ".join(headers) + " |",
                    "|" + " --- |" * len(headers),
                ]
            )
            md.extend("| " + " | ".join(row) + " |" for row in latency_rows)
            md.extend(
                [
                    "",
                    "### Isolated phases for uniform routing with eight tokens",
                    "",
                    "Same device/dtype/shape; threshold 1; 20 warmups, 100 samples per stage; eager.",
                    "",
                    "| Stage | CUDA event ms | Wall ms |",
                    "| --- | --- | --- |",
                ]
            )
            document.add_heading("Isolated phases for eight tokens", level=2)
            document.add_paragraph(
                "Same device, dtype and shape; uniform routing, threshold 1; "
                "20 warmups and 100 samples per isolated stage, eager."
            ).paragraph_format.keep_with_next = True
            add_table(
                document, ["Stage", "CUDA event ms", "Wall ms"], profile_rows, [3.0, 1.3, 1.3]
            )
            md.extend("| " + " | ".join(row) + " |" for row in profile_rows)
            md.append("")
    figure = output / "figures" / "uniform" / "occupancy.png"
    if route_rows:
        document.add_heading("Observed pretrained first-layer occupancy", level=2)
        caption = (
            "Intel Core Ultra 9 185H, CPU BF16, upstream eager causal attention, batch 1; "
            "context length below. "
            "Each row is one prefill capture, not a latency measurement. No warmup/repetitions "
            "apply to these deterministic occupancy counts. Checkpoint and source hashes are archived."
        )
        document.add_paragraph(caption).paragraph_format.keep_with_next = True
        headers = ["Probe", "Context tokens", "Singleton experts", "Maximum occupancy"]
        add_table(document, headers, route_rows, [1.4, 1.2, 1.4, 1.5])
        md.extend(
            [
                "## Observed pretrained first-layer occupancy",
                "",
                caption,
                "",
                "| " + " | ".join(headers) + " |",
                "|" + " --- |" * len(headers),
            ]
        )
        md.extend("| " + " | ".join(row) + " |" for row in route_rows)
        md.extend(["", "Additional latency and occupancy plots are in [figures](figures).", ""])
    elif figure.exists():
        document.add_heading("Observed synthetic occupancy", level=2)
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run().add_picture(str(figure), width=Inches(5.0))
        md.extend(
            [
                "## Observed synthetic occupancy",
                "",
                "![Uniform synthetic expert occupancy](figures/uniform/occupancy.png)",
                "",
                "Additional latency and occupancy figures are in [figures](figures).",
                "",
            ]
        )
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("PhaseMoE midpoint draft  |  ").font.size = Pt(9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    document.core_properties.title = "PhaseMoE Midpoint Report"
    document.core_properties.author = "PhaseMoE team"
    document.save(output / "midpoint-report.docx")
    (output / "midpoint-report.md").write_text("\n".join(md), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("reports"))
    args = p.parse_args()
    build(args.benchmark, args.profile, args.output)


if __name__ == "__main__":
    main()
