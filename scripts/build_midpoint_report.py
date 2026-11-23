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
    if data.get("status") != "complete" or prof.get("status") != "complete":
        raise ValueError("report requires complete baseline benchmark and profile")
    rows, env = data["results"], data["environment"]
    if not rows or any(
        r["backend"] not in ("torch", "triton_grouped") or not r["correctness"]["passed"]
        for r in rows
    ):
        raise ValueError("report accepts only correct baseline cases")
    if prof["backend"] != "triton_grouped" or not prof["correctness"]["passed"]:
        raise ValueError("report requires a correct grouped-only profile")
    first, shape = rows[0], rows[0]["config"]
    if any(
        r["config"] != shape
        or r["dtype"] != first["dtype"]
        or r["timing"]["mode"] != "eager"
        or r["timing"]["warmup"] != first["timing"]["warmup"]
        or r["timing"]["repetitions"] != first["timing"]["repetitions"]
        for r in rows
    ):
        raise ValueError("split incompatible baseline setups before report generation")
    selected = {(r["tokens"], r["backend"]): r for r in rows if r["distribution"] == "uniform"}
    tokens = sorted({r["tokens"] for r in rows})
    latency_rows = [
        [
            str(t),
            f"{selected[t, 'torch']['timing']['cuda_event']['median_ms']:.3f}",
            f"{selected[t, 'triton_grouped']['timing']['cuda_event']['median_ms']:.3f}",
            f"{selected[t, 'triton_grouped']['timing']['wall']['median_ms']:.3f}",
        ]
        for t in tokens
    ]
    profile_rows = [
        [
            name.replace("_", " "),
            f"{v['cuda_event']['median_ms']:.3f}",
            f"{v['wall']['median_ms']:.3f}",
        ]
        for name, v in prof["stages"].items()
    ]
    largest = max(
        (k for k in prof["stages"] if k != "full_expert_block"),
        key=lambda k: prof["stages"][k]["cuda_event"]["median_ms"],
    )
    maximum_error = max(r["correctness"]["relative_l2_error"] for r in rows)
    clock_gap = max(
        abs(r["timing"]["cuda_event"]["median_ms"] / r["timing"]["wall"]["median_ms"] - 1)
        for r in rows
    )
    telemetry = []
    for row in rows:
        for key in ("gpu_telemetry_before", "gpu_telemetry_after"):
            value = row.get(key, {}).get("values")
            if value:
                parts = [s.strip() for s in value.split(",")]
                if len(parts) == 7:
                    telemetry.append(parts)
    clock_note = "Per-case device snapshots are retained; clocks were not locked."
    if telemetry:
        sm = [float(v[4].split()[0]) for v in telemetry]
        mem = [float(v[5].split()[0]) for v in telemetry]
        clock_note = (
            f"Snapshots report states {', '.join(sorted({v[1] for v in telemetry}))}, "
            f"SM clocks {min(sm):.0f}-{max(sm):.0f} MHz and memory clocks "
            f"{min(mem):.0f}-{max(mem):.0f} MHz. Clocks were not locked; these are "
            "before/after snapshots, not continuous measurements."
        )
    warm, reps = first["timing"]["warmup"], first["timing"]["repetitions"]
    device = env.get("gpu_name", env["device"])
    setup = (
        f"Measured hardware: {device}, {env.get('gpu_memory_bytes', 0) / 2**30:.2f} GiB, "
        f"compute capability {env.get('compute_capability')}; host {env.get('cpu_model', 'unrecorded')}. "
        f"Python {env['python']}, PyTorch {env['torch']}, Triton {env.get('triton')}, CUDA runtime "
        f"{env.get('cuda_runtime')}; driver/device query: {env.get('nvidia_smi')}. "
        f"Platform: {env['platform']}; {env['cpu_threads']} CPU thread(s)."
    )
    protocol = (
        f"Shape E={shape['num_experts']}, K={shape['top_k']}, H={shape['hidden_size']}, "
        f"I={shape['intermediate_size']}; {first['dtype']}. T={','.join(map(str, tokens))} "
        "flattened synthetic tokens (recorded batch T, context 1); uniform, balanced and skewed routes. "
        f"Seeds are {first['seed']} for routes, {first['seed'] + 1} for random weights and "
        f"{first['seed'] + 2} for inputs. Correctness/JIT precede {warm} warmups and {reps} "
        "eager samples per case. Both synchronized wall and CUDA-event timings include dispatch, "
        "allocation, expert projections, activation and weighted reduction; router selection, "
        "attention, normalization and loading are excluded."
    )
    summary = (
        "We have established a correct, reproducible single-block baseline and begun measuring "
        "expert occupancy and execution cost. The research question is whether a different GPU "
        "execution design can improve MoE inference under uneven expert workloads. The central "
        "optimization is planned work for the second half; no optimization result is claimed."
    )
    baseline = (
        "The checkpoint contains a PyTorch reference, independent FP32 oracle, deterministic "
        "synthetic routes, and a fixed grouped-GEMM baseline. All active experts use grouped "
        "projections and store down-projection rows before a separate weighted reduction. "
        "The inherited gate/up stage shares a launch with activation. This is an inspectable "
        "baseline, not a production-performance standard or a finalized research design."
    )
    validation = (
        f"All {len(rows)} full-shape cases passed FP32-reference checks at atol "
        f"{first['correctness']['atol']} / rtol {first['correctness']['rtol']}; largest relative L2 "
        f"error {100 * maximum_error:.3f}%. Small-shape tests also check an independent oracle, "
        "empty batches, tail dimensions and routing/indexing. These checks do not establish "
        "pretrained-model quality."
    )
    profile_note = (
        f"Grouped baseline, {prof['dtype']}, T={prof['tokens']}, {prof['distribution']} routes; "
        f"same device and model dimensions. Each stage uses "
        f"{prof['stages']['dispatch']['warmup']} warmups and "
        f"{prof['stages']['dispatch']['repetitions']} eager samples. "
        "Inputs to isolated stages are precomputed; do not add their medians to estimate block latency."
    )
    observations = (
        f"The largest measured isolated phase is {largest.replace('_', ' ')}. "
        f"At T={prof['tokens']}, the profiled routes have {prof['occupancy']['singleton_experts']} "
        f"singleton and {prof['occupancy']['empty_experts']} empty experts out of "
        f"{shape['num_experts']}. Occupancy varies with distribution and token count. "
        "This motivates investigating utilization, dispatch and data movement, but does not "
        "identify an optimal design or prove a memory-bandwidth bottleneck. A warmed PyTorch "
        "profiler trace and occupancy plots are retained; no Nsight traffic counters were collected."
    )
    future = (
        "The team will reproduce the baseline on L40S/A6000, establish an optimized external "
        "comparison, and collect actual prefill/decode routes across layers and batch sizes. "
        "We will independently explore layouts, execution strategies, wider or narrower fusion, "
        "and occupancy-aware scheduling. No fusion boundary or singleton threshold is prescribed. "
        "Candidates must preserve correctness; full-block measurements including scheduling will "
        "determine whether they help. Negative findings remain valid outcomes."
    )
    limits = (
        f"Preliminary synthetic tests use fixed backend order on laptop/WSL hardware. {clock_note} "
        f"CUDA-event and wall medians differ by up to {100 * clock_gap:.1f}%; the cause is unresolved. "
        "Target-GPU replication and independent repeats are needed. Small-T tests are not actual "
        "decode. Historical baseline samples are retained separately, not used for before/after claims."
    )
    admin = (
        "UNIs, confirmed roles and team-size approval remain pending (three names supplied; "
        "template requests four to five). README, JSON and plots document experiments. W&B is "
        "optional; the rubric still requires a dashboard link or instructor approval of the "
        "repository summary as an alternative."
    )
    provenance = (
        f"Measured {data['created_at_utc'][:10]} UTC. Parent revision {env.get('git_commit', '')[:7]}; "
        f"working tree modified at capture: {env.get('git_dirty')}. JSON records source hashes and raw "
        "samples. Use those hashes and the checkpoint source files to identify the measured code; "
        "the parent revision alone does not identify an uncommitted measurement snapshot."
    )

    document = Document()
    for border in document.styles.element.xpath(".//w:pBdr"):
        border.getparent().remove(border)
    sec = document.sections[0]
    sec.top_margin = Inches(0.65)
    sec.bottom_margin = Inches(0.85)
    sec.left_margin = sec.right_margin = Inches(0.75)
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    normal = document.styles["Normal"]
    normal.font.name, normal.font.size = "Calibri", Pt(10)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.04
    for name, size in [("Title", 21), ("Heading 1", 13), ("Heading 2", 11)]:
        style = document.styles[name]
        style.font.name, style.font.size = "Calibri", Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(9)
        style.paragraph_format.space_after = Pt(4)
    document.add_paragraph("PhaseMoE midpoint checkpoint", "Title")
    document.add_paragraph("Baseline establishment and characterization")
    document.add_paragraph("Aedin Pereira, Srikar Kovvuri, David Samtani | UNIs and roles pending")
    md = [
        "# PhaseMoE midpoint checkpoint",
        "",
        "Baseline establishment and characterization",
        "",
        "Aedin Pereira, Srikar Kovvuri, David Samtani. UNIs and roles pending.",
        "",
    ]

    def section(title, paragraphs):
        document.add_heading(title, level=1)
        md.extend([f"## {title}", ""])
        for text in paragraphs:
            add_prose(document, text)
            md.extend([text, ""])

    def tab(headers, values, widths):
        add_table(document, headers, values, widths)
        md.extend(["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)])
        md.extend("| " + " | ".join(row) + " |" for row in values)
        md.append("")

    section("Project summary and baseline", [summary, baseline])
    section("Experimental setup", [setup, protocol, validation])
    section(
        "Preliminary baseline latency",
        [
            f"Uniform synthetic routing; medians in ms; setup above, {warm} warmups and {reps} samples. PyTorch loops over experts and is not grouped GEMM."
        ],
    )
    tab(["T", "PyTorch CUDA", "Grouped CUDA", "Grouped wall"], latency_rows, [0.5, 1.8, 1.8, 1.8])
    document.add_page_break()
    section("Profiling and occupancy observations", [profile_note])
    tab(["Stage", "CUDA ms", "Wall ms"], profile_rows, [3.9, 1.0, 1.0])
    add_prose(document, observations)
    md.extend([observations, ""])
    section("Remaining technical work", [future])
    section("Limitations and requirements", [limits, admin])
    section(
        "Project links and reproducibility",
        [
            f"Repository: {REPO}",
            "Local evidence: reports/results/baseline-current.json and baseline-profile.json; figures in reports/figures. Historical subset: baseline-initial.json.",
            "References: OLMoE paper https://arxiv.org/abs/2409.02060 and the model configuration linked from the README.",
            provenance,
        ],
    )
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("PhaseMoE midpoint draft | ").font.size = Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    document.core_properties.title = "PhaseMoE midpoint checkpoint"
    document.core_properties.author = "PhaseMoE team"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
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
