"""Generate a PDF with one plot per result's extracted_points."""

import argparse
import json
import re
import difflib
import textwrap
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

# Use a non-interactive backend for headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import cycle


def iter_json_objects(path: Path) -> Iterable[Dict]:
    """Yield concatenated JSON objects from a file (pretty-printed JSONL style)."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)

    while idx < length:
        # Skip whitespace between objects.
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, end = decoder.raw_decode(text, idx)
        yield obj
        idx = end


def extract_series_points(result: Dict) -> Tuple[str, List[Dict]]:
    """Return case_id and series list from a result object."""
    case_id = result.get("case_id", "unknown_case")
    extracted = result.get("extracted_points") or {}
    series = extracted.get("series") or []
    return case_id, series


def maybe_apply_axis(
    ax: plt.Axes, series: List[Dict], data_range: Dict[str, Tuple[Optional[float], Optional[float]]]
) -> None:
    """Apply axis limits/scales if present in the series metadata, expanded to data range."""
    for axis_key in ("x", "y"):
        axis_info = None
        for s in series:
            axis_info = (s.get("axis") or {}).get(axis_key)
            if axis_info:
                break
        if not axis_info:
            # If no explicit axis info, fall back to data-driven limits.
            min_val, max_val = data_range.get(axis_key, (None, None))
            if min_val is not None and max_val is not None:
                getattr(ax, f"set_{axis_key}lim")(min_val, max_val)
            continue

        scale = axis_info.get("scale")
        if scale == "log":
            getattr(ax, f"set_{axis_key}scale")("log")

        min_val = axis_info.get("min")
        max_val = axis_info.get("max")
        data_min, data_max = data_range.get(axis_key, (None, None))
        if min_val is not None and max_val is not None:
            # Expand limits if actual data goes beyond declared axis range.
            if data_min is not None:
                min_val = min(min_val, data_min)
            if data_max is not None:
                max_val = max(max_val, data_max)
            getattr(ax, f"set_{axis_key}lim")(min_val, max_val)


def collect_data_range(series_list: List[Dict]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Compute min/max for x and y across all series."""
    mins = {"x": None, "y": None}
    maxs = {"x": None, "y": None}
    for s in series_list:
        for p in s.get("points") or []:
            for axis in ("x", "y"):
                val = p.get(axis)
                if val is None:
                    continue
                mins[axis] = val if mins[axis] is None else min(mins[axis], val)
                maxs[axis] = val if maxs[axis] is None else max(maxs[axis], val)
    return {axis: (mins[axis], maxs[axis]) for axis in ("x", "y")}


def normalize_label(label: str) -> str:
    """Normalize labels for fuzzy matching."""
    s = (label or "").lower()
    # Standardize common symbols.
    s = (
        s.replace("≤", "<=")
        .replace("≥", ">=")
        .replace("⩽", "<=")
        .replace("⩾", ">=")
        .replace("–", "-")
        .replace("−", "-")
    )
    tokens = re.findall(r"(?:<=|>=|<|>|=)|[a-z0-9]+", s)
    return " ".join(tokens)


def similarity(a: str, b: str) -> float:
    """Blend sequence ratio with token overlap to score label closeness."""
    ra = normalize_label(a)
    rb = normalize_label(b)
    if not ra or not rb:
        return 0.0
    seq_score = difflib.SequenceMatcher(None, ra, rb).ratio()
    ta = set(ra.split())
    tb = set(rb.split())
    jaccard = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
    return 0.7 * seq_score + 0.3 * jaccard


def match_series_labels(
    extracted: List[Dict], gt: List[Dict], metric_pairs: Optional[List[Dict]] = None
) -> Dict[int, int]:
    """
    Return mapping gt_index -> extracted_index for best label matches.
    Prefer explicit mappings provided in metrics["pairs"] (model_label -> real_label),
    then fill remaining with heuristic similarity-based matching.
    """
    matches: Dict[int, int] = {}
    used_gt = set()
    used_extracted = set()

    # Index labels for quick lookup.
    extracted_labels = {normalize_label(s.get("label", f"series_{i}")): i for i, s in enumerate(extracted)}
    gt_labels = {normalize_label(s.get("label", f"gt_{i}")): i for i, s in enumerate(gt)}

    def find_closest(label: str, pool: Dict[str, int]) -> Optional[int]:
        target = normalize_label(label)
        if target in pool:
            return pool[target]
        best_idx = None
        best_score = 0.0
        for name, idx in pool.items():
            score = similarity(name, target)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx if best_score >= 0.45 else None

    # 1) Use explicit metric pairs if available.
    if metric_pairs:
        for pair in metric_pairs:
            m_label = pair.get("model_label")
            g_label = pair.get("real_label")
            if not m_label or not g_label:
                continue
            mi = find_closest(m_label, extracted_labels)
            gi = find_closest(g_label, gt_labels)
            if mi is not None and gi is not None and mi not in used_extracted and gi not in used_gt:
                matches[gi] = mi
                used_gt.add(gi)
                used_extracted.add(mi)

    # 2) Heuristic fill for remaining.
    pair_scores: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        if gi in used_gt:
            continue
        g_label = g.get("label", f"gt_{gi}")
        for ei, e in enumerate(extracted):
            if ei in used_extracted:
                continue
            e_label = e.get("label", f"series_{ei}")
            pair_scores.append((similarity(e_label, g_label), gi, ei))

    # Sort by descending similarity to pick strongest pairs first.
    pair_scores.sort(reverse=True, key=lambda t: t[0])

    for score, gi, ei in pair_scores:
        if score < 0.35:  # lower threshold to catch near-variants while avoiding noise
            break
        if gi in used_gt or ei in used_extracted:
            continue
        matches[gi] = ei
        used_gt.add(gi)
        used_extracted.add(ei)

    return matches


def compute_integral_l1_normalized(
    metrics: Optional[Dict], gt_series: List[Dict], model_series: List[Dict]
) -> Optional[float]:
    """
    Compute mean integral_l1 normalized by (y_range * x_overlap_length) for each pair,
    analogous to nMAE but for integrated error.
    """
    if not metrics or not gt_series or not model_series:
        return None
    pairs = metrics.get("pairs") or []
    if not pairs:
        return None

    def series_range(s: Dict, axis: str) -> Optional[Tuple[float, float]]:
        vals = [p.get(axis) for p in s.get("points") or [] if p.get(axis) is not None]
        if not vals:
            return None
        return min(vals), max(vals)

    def best_match(label: str, series_list: List[Dict]) -> Optional[Dict]:
        target = normalize_label(label or "")
        best = None
        best_score = 0.0
        for s in series_list:
            score = similarity(normalize_label(s.get("label", "")), target)
            if score > best_score:
                best_score = score
                best = s
        return best if best_score >= 0.35 else None

    norm_vals: List[float] = []
    for pair in pairs:
        val = pair.get("integral_l1")
        if not isinstance(val, (int, float)) or math.isnan(val):
            continue

        gt_s = best_match(pair.get("real_label"), gt_series)
        model_s = best_match(pair.get("model_label"), model_series)
        if not gt_s or not model_s:
            continue

        y_range = series_range(gt_s, "y")
        gt_x = series_range(gt_s, "x")
        model_x = series_range(model_s, "x")
        if not y_range or not gt_x or not model_x:
            continue

        y_span = y_range[1] - y_range[0]
        x_overlap = max(0.0, min(gt_x[1], model_x[1]) - max(gt_x[0], model_x[0]))
        if y_span <= 0 or x_overlap <= 0:
            continue

        norm_vals.append(val / (y_span * x_overlap))

    if not norm_vals:
        return None

    return sum(norm_vals) / len(norm_vals)


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def compute_summary_stats(cases: List[Dict]) -> Dict:
    n = len(cases)
    success = sum(1 for c in cases if c.get("success"))

    def collect(key: str) -> List[float]:
        vals = []
        for c in cases:
            v = c.get(key)
            if isinstance(v, (int, float)) and not math.isnan(v):
                vals.append(float(v))
        return vals

    nmae_vals = collect("overall_nmae")
    l1_norm_vals = collect("overall_integral_l1_normalized")
    duration_vals = collect("extractor_seconds")
    gemini_counts = collect("extracted_series_count")
    serie_counts = collect("GT_series_count")

    def stat_block(vals: List[float]) -> Dict[str, Optional[float]]:
        if not vals:
            return {"mean": None, "median": None, "count": 0}
        return {"mean": sum(vals) / len(vals), "median": median(vals), "count": len(vals)}

    mismatch_series = sum(
        1
        for c in cases
        if isinstance(c.get("extracted_series_count"), (int, float))
        and isinstance(c.get("extraction_series_count"), (int, float))
        and c["extracted_series_count"] != c["extraction_series_count"]
    )

    return {
        "case_count": n,
        "success_count": success,
        "nmae": stat_block(nmae_vals),
        "l1_norm": stat_block(l1_norm_vals),
        "duration": stat_block(duration_vals),
        "extracted_series_count": stat_block(gemini_counts),
        "series_count": stat_block(serie_counts),
        "series_mismatch_count": mismatch_series,
    }


def title_from_input_path(input_path: Path) -> str:
    """Derive a friendly title from the input file path."""
    parent = input_path.parent.name
    prefix = "benchmark_results_"
    name = parent[len(prefix) :] if parent.startswith(prefix) else parent
    parts = name.split("_")
    if len(parts) >= 2:
        model = " ".join(parts[:-1]).title()
        dataset = parts[-1].upper()
        return f"{model} - {dataset} dataset - Benchmark summary"
    model = name.replace("_", " ").title()
    return f"{model} - Benchmark summary"


SUMMARY_PROMPT = textwrap.dedent(
    """
    You are an expert system that extracts NUMERICAL data from 2D scientific plots.
    You must return ALL visible series from the given image.

    Rules:
    - Do not guess or invent data.
    - Work in three internal steps (do NOT show them):
      1) Read axes:
         - Determine x_min, x_max (detect log10 if any).
         - Determine y_min, y_max (detect log10 if any).
         - List visible tick labels for x and y (as numbers).
      2) Identify all visible data series:
         - For discrete markers: read their positions and map to axis coordinates.
         - For continuous curves: sample >= 20 points (include local extrema, inflections, and uniform samples).
      3) Validate:
         - Ensure x in [x_min, x_max], y in [y_min, y_max].
         - For linear axes, reject perfectly uniform sequences unless they exactly match tick labels.
         - If uncertain about a point, omit it. Never invent.

    Output format (STRICT JSON, no extra text):
    {
      "series": [
        {
          "label": "<legend name if visible else 'series_N'>",
          "points": [
            {"x": <float>, "y": <float>, "confidence": <float 0..1>},
            ...
          ],
          "axis": {
            "x": {"min": <float>, "max": <float>, "scale": "linear"|"log10"},
            "y": {"min": <float>, "max": <float>, "scale": "linear"|"log10"}
          }
        },
        ...
      ]
    }

    Additional rules:
    - Include ALL visible series.
    - Keep points sorted by increasing x.
    - Use at least 20 points for continuous curves if readable; otherwise return all discrete markers.
    - Never include explanations, placeholders, or text outside JSON.
    """
)


def make_summary_page(stats: Dict) -> plt.Figure:
    """Create a summary page describing the benchmark setup."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")

    header = stats.get("title", "Benchmark Summary")
    body_lines = [
        f"Model: gemini-3-pro-preview", #"Model: gpt-5.2"
        f"Cases: {stats.get('case_count', 0)}",
        f"Successes: {stats.get('success_count', 0)}",
        "",
        "nMAE (mean / median, n="
        f"{stats['nmae'].get('count')})  "
        f"{stats['nmae'].get('mean'):.4f} / {stats['nmae'].get('median'):.4f}"
        if stats.get("nmae") and stats["nmae"].get("mean") is not None
        else "nMAE: —",
        "L1 normalized (mean / median, n="
        f"{stats['l1_norm'].get('count')})  "
        f"{stats['l1_norm'].get('mean'):.4f} / {stats['l1_norm'].get('median'):.4f}"
        if stats.get("l1_norm") and stats["l1_norm"].get("mean") is not None
        else "L1 normalized: —",
        "Duration seconds (mean / median, n="
        f"{stats['duration'].get('count')})  "
        f"{stats['duration'].get('mean'):.1f} / {stats['duration'].get('median'):.1f}"
        if stats.get("duration") and stats["duration"].get("mean") is not None
        else "Duration: —",
        f"Series count mismatches (extracted vs GT): {stats.get('series_mismatch_count', 0)}",
        "",
    ]
    body = "\n".join(body_lines)

    ax.text(
        0.02,
        0.98,
        f"{header}",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.86,
        body,
        ha="left",
        va="top",
        fontsize=11,
        wrap=True,
    )
    fig.tight_layout()
    return fig


def make_prompt_page(title: str) -> plt.Figure:
    """Create a page showing the prompt used."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        f"{title} - Prompt",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )
    wrapped_prompt = textwrap.fill(SUMMARY_PROMPT, width=92)
    ax.text(
        0.02,
        0.90,
        wrapped_prompt,
        ha="left",
        va="top",
        fontsize=9.5,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )
    fig.tight_layout()
    return fig


def load_ground_truth_series(csv_path: Path) -> List[Dict]:
    """Load ground truth series from a two-column-per-series CSV (X,Y)."""
    import csv

    if not csv_path.exists():
        return []

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return []

    headers = rows[0]
    data_rows = rows[1:]
    series_list: List[Dict] = []

    # Expect pairs of columns: [label_x, label_y, ...]
    for i in range(0, len(headers), 2):
        label = headers[i].strip()
        if not label:
            continue
        x_idx, y_idx = i, i + 1 if i + 1 < len(headers) else None
        points = []
        for row in data_rows:
            if x_idx >= len(row) or (y_idx is not None and y_idx >= len(row)):
                continue
            x_raw = row[x_idx].strip() if x_idx < len(row) else ""
            y_raw = row[y_idx].strip() if y_idx is not None and y_idx < len(row) else ""
            try:
                x_val = float(x_raw)
                y_val = float(y_raw)
            except (TypeError, ValueError):
                continue
            points.append({"x": x_val, "y": y_val})
        if points:
            series_list.append({"label": label, "points": points})
    return series_list

def canonical_to_categorical_plot(series_obj):
    """
    Convert canonical categorical series into a single plottable series
    with category labels on x-axis.
    """

    categories = []
    values = []

    for s in series_obj["series"]:
        label = s["label"]
        pts = s["points"]

        # Take the main value (first point)
        main_pt = pts[0]

        categories.append(label)
        values.append(main_pt["y"])

    return {
        "categories": categories,
        "values": values
    }

def is_categorical(series: List[Dict]) -> bool:
    return (
        series
        and all(len(s.get("points", [])) <= 4 for s in series)
        and len(series) >= 2
    )

def plot_result_series(
    case_id: str,
    series: List[Dict],
    gt_series: List[Dict],
    image_path: Optional[Path],
    metrics: Optional[Dict],
) -> plt.Figure:
    """Create a matplotlib Figure for a single result entry, overlaying ground truth, image, and metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1.4, 0.9]})
    ax_img, ax_plot, ax_table = axes

    combined_series = list(series) + list(gt_series)
    categorical_mode = is_categorical(series)
    if not combined_series:
        ax_plot.text(0.5, 0.5, "No data to plot", ha="center", va="center", fontsize=12)
        ax_plot.set_axis_off()
        ax_img.axis("off")
        ax_table.axis("off")
        return fig
    
    if categorical_mode:
        # --- categorical plot: one point per category ---
        labels = []
        ys = []

        for s in series:
            pts = s.get("points", [])
            if not pts:
                continue
            labels.append(s["label"])
            ys.append(pts[0]["y"])

        xs = list(range(len(labels)))
        ax_plot.scatter(xs, ys, s=50, zorder=3)

        ax_plot.set_xticks(xs)
        ax_plot.set_xticklabels(labels, rotation=30, ha="right")

    else:
        # --- normal continuous plotting (UNCHANGED) ---
        for idx, s in enumerate(series):
            points = s.get("points") or []
            if not points:
                continue
            xs = [p["x"] for p in points]
            ys = [p["y"] for p in points]
            label = s.get("label", f"series_{idx+1}")
            ax_plot.plot(xs, ys, marker="o", linewidth=1.5, label=label)

    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
    matches = match_series_labels(series, gt_series, metrics.get("pairs") if metrics else None)
    extracted_colors: Dict[int, str] = {}
    gt_colors: Dict[int, str] = {}

    def next_color() -> str:
        return next(color_cycle)

    # Assign shared colors to matched pairs.
    for gi, ei in matches.items():
        c = next_color()
        extracted_colors[ei] = c
        gt_colors[gi] = c

    # Assign remaining extracted series.
    for ei in range(len(series)):
        if ei not in extracted_colors:
            extracted_colors[ei] = next_color()

    # Assign remaining GT series.
    for gi in range(len(gt_series)):
        if gi not in gt_colors:
            gt_colors[gi] = next_color()

    for idx, s in enumerate(series):
        points = s.get("points") or []
        if not points:
            continue

        xs = [p.get("x") for p in points if p.get("x") is not None and p.get("y") is not None]
        ys = [p.get("y") for p in points if p.get("x") is not None and p.get("y") is not None]
        confs = [p.get("confidence") for p in points]

        label = s.get("label") or f"series_{idx + 1}"
        color = extracted_colors.get(idx, next_color())
        (line,) = ax_plot.plot(
            xs, ys, marker="o", linewidth=1.5, label=f"{label} (AI extraction)", color=color
        )

        # Visualize confidence via marker transparency/size when available.
        if any(c is not None for c in confs):
            alphas = [max(0.2, min(1.0, c if c is not None else 0.8)) for c in confs]
            sizes = [20 + 60 * a for a in alphas]
            ax_plot.scatter(xs, ys, color=line.get_color(), s=sizes, alpha=0.35, edgecolor="none")

    if not categorical_mode:
        for idx, s in enumerate(gt_series):
            points = s.get("points") or []
            if not points:
                continue
            xs = [p.get("x") for p in points if p.get("x") is not None and p.get("y") is not None]
            ys = [p.get("y") for p in points if p.get("x") is not None and p.get("y") is not None]
            label = s.get("label") or f"gt_{idx + 1}"
            color = gt_colors.get(idx, next_color())
            ax_plot.plot(
                xs,
                ys,
                linestyle="--",
                linewidth=1.5,
                marker="x",
                label=f"{label} (manual extraction)",
                color=color,
            )

    data_range = collect_data_range(combined_series)
    maybe_apply_axis(ax_plot, combined_series, data_range)
    ax_plot.set_title(case_id)
    ax_plot.set_xlabel("x")
    ax_plot.set_ylabel("y")
    ax_plot.grid(True, linestyle="--", alpha=0.3)
    ax_plot.legend()
    # Show source image on the left.
    if image_path and image_path.exists():
        try:
            img = plt.imread(str(image_path))
            ax_img.imshow(img)
            ax_img.set_title("Source image")
        except Exception as exc:  # noqa: BLE001
            ax_img.text(0.5, 0.5, f"Failed to load image:\\n{exc}", ha="center", va="center", fontsize=10)
            ax_img.set_title("Source image (missing)")
    else:
        ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=12)
        ax_img.set_title("Source image (missing)")
    ax_img.axis("off")

    # Show metrics table on the right.
    ax_table.axis("off")
    metrics_data = metrics or {}
    integral_l1_norm = compute_integral_l1_normalized(metrics_data, gt_series, series)
    if integral_l1_norm is not None:
        metrics_data = dict(metrics_data)
        metrics_data["overall_integral_l1_normalized"] = integral_l1_norm
    rows = [
        ("duration (s)", metrics_data.get("extractor_seconds")),
        ("overall_nmae", metrics_data.get("overall_nmae")),
        ("overall_integral_l1", metrics_data.get("overall_integral_l1")),
        ("overall_integral_l1_normalized", metrics_data.get("overall_integral_l1_normalized")),
        ("overall_x_misaligned", metrics_data.get("overall_x_misaligned")),
        ("extracted_series_count", metrics_data.get("model_series_count")),
        ("GT_series_count", metrics_data.get("gt_series_count")),
    ]

    table_cells = []
    for key, val in rows:
        if key == "duration (s)" and isinstance(val, (int, float)):
            table_cells.append((key, f"{int(round(val))}"))
        elif isinstance(val, float):
            table_cells.append((key, f"{val:.4g}"))
        else:
            table_cells.append((key, str(val) if val is not None else "—"))

    cell_text = [[k, v] for k, v in table_cells]
    table = ax_table.table(
        cellText=cell_text,
        colLabels=["Metric", "Value"],
        loc="center",
        colWidths=[0.6, 0.4],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    ax_table.set_title("Metrics", pad=10)

    fig.tight_layout()
    return fig

    
def plot_extracted_points(directory: Path) -> None:
    INPUT_FILE_PATH = directory / "benchmark_results.jsonl"
    OUTPUT_FILE_PATH = directory / "extracted_points_plots_simple.pdf"
    parser = argparse.ArgumentParser(
        description="Generate a PDF with plots for every extracted_points entry in a benchmark results file."
    )
    parser.add_argument(
        "--input",
        default=INPUT_FILE_PATH,
        help="Path to the benchmark_results.jsonl file (supports concatenated JSON objects).",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE_PATH,
        help="Where to write the PDF.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    results = list(iter_json_objects(input_path))
    if not results:
        raise ValueError(f"No JSON objects found in {input_path}")

    # Precompute cases to drive both summary and per-case pages.
    cases: List[Dict] = []
    for obj in results:
        result_obj = obj.get("result", obj)
        case_id, series = extract_series_points(result_obj)
        gt_path_str = result_obj.get("gt_path")
        gt_series: List[Dict] = []
        if gt_path_str:
            try:
                gt_series = load_ground_truth_series(Path(gt_path_str))
            except Exception as exc:  # noqa: BLE001
                gt_series = []
                print(f"[warn] Failed to load GT for {case_id}: {exc}")

        metrics = result_obj.get("metrics") or {}
        if result_obj.get("extractor_seconds") is not None:
            metrics = dict(metrics)
            metrics["extractor_seconds"] = result_obj.get("extractor_seconds")

        integral_l1_norm = compute_integral_l1_normalized(metrics, gt_series, series)
        if integral_l1_norm is not None:
            metrics = dict(metrics)
            metrics["overall_integral_l1_normalized"] = integral_l1_norm

        cases.append(
            {
                "case_id": case_id,
                "series": series,
                "gt_series": gt_series,
                "image_path": Path(result_obj["image_path"]) if result_obj.get("image_path") else None,
                "metrics": metrics,
                "success": result_obj.get("success", False),
                "overall_nmae": metrics.get("overall_nmae"),
                "overall_integral_l1_normalized": metrics.get("overall_integral_l1_normalized"),
                "extractor_seconds": metrics.get("extractor_seconds"),
                "extracted_series_count": metrics.get("model_series_count"),
                "GT_series_count": metrics.get("gt_series_count"),
            }
        )

    stats = compute_summary_stats(cases)
    title = title_from_input_path(input_path)
    stats["title"] = title

    with PdfPages(output_path) as pdf:
        summary_fig = make_summary_page(stats)
        pdf.savefig(summary_fig)
        plt.close(summary_fig)

        prompt_fig = make_prompt_page(title)
        pdf.savefig(prompt_fig)
        plt.close(prompt_fig)

        def sort_key(case: Dict) -> float:
            val = case.get("overall_nmae")
            if isinstance(val, (int, float)) and not math.isnan(val):
                return float(val)
            return float("inf")

        for case in sorted(cases, key=sort_key):
            fig = plot_result_series(
                case["case_id"],
                case["series"],
                case["gt_series"],
                case["image_path"],
                case["metrics"],
            )
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {output_path} with {len(results)} page(s).")


if __name__ == "__main__":
    dir = "data/benchmark_results_gpt_simple/"
    plot_extracted_points(directory=Path(dir))
