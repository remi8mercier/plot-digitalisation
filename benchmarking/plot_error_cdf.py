"""
Plot the error CDF for each method.
"""


import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
METHOD_ROOTS = {
    "D&C": "data/benchmark_results_pipeline/",
    "D&C with KM snapping": "data/benchmark_results_pipeline_with_km/",
    "gemini": "data/benchmark_results_gemini_only/",
    "gpt_5": "data/benchmark_results_gpt_5/",
    "gpt_5_2": "data/benchmark_results_gpt_simple/",
    "gemini with KM snapping": "data/benchmark_results_gemini_km_snapping/",

}

METRIC_COL = "overall_nmae"   # or "l1_scaled_by_gt_yrange"


# ------------------------------------------------------------
# Load ONE comparison_metrics.json
# ------------------------------------------------------------
def load_case_metrics(metrics_path: Path) -> Dict | None:
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as exc:
        print(f"[warn] Failed to read {metrics_path}: {exc}")
        return None

    return {
        "case_id": metrics_path.parent.name,
        "overall_nmae": metrics.get("overall_nmae"),
        "l1_scaled_by_gt_yrange": metrics.get("l1_scaled_by_gt_yrange"),
    }


# ------------------------------------------------------------
# Collect ALL datapoints for one method
# ------------------------------------------------------------
def collect_method_df(root: Path) -> pd.DataFrame:
    rows: List[Dict] = []

    for metrics_path in root.rglob("comparison_metrics.json"):
        row = load_case_metrics(metrics_path)
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError(f"No comparison_metrics.json found under {root}")

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Plot empirical CDF
# ------------------------------------------------------------
def plot_cdf(ax, values: np.ndarray, label: str):
    values = np.sort(values)
    y = np.arange(1, len(values) + 1) / len(values)
    ax.plot(values, y, label=label)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(
    output_dir: str | None = None,
    show: bool = True,
    dpi: int = 300,
):
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Precompute full CDFs (once)
    # ------------------------------------------------------------
    method_curves = {}

    for method, root_path in METHOD_ROOTS.items():
        root = Path(root_path)
        if not root.exists():
            print(f"[warn] Missing root folder: {root}")
            continue

        df = collect_method_df(root)
        values = np.sort(df[METRIC_COL].dropna().values)
        if len(values) == 0:
            continue

        y = np.arange(1, len(values) + 1) / len(values)
        method_curves[method] = (values, y)

    if not method_curves:
        raise RuntimeError("No valid datapoints found.")

    # ============================================================
    # Figure 1: Zoom on [0, 0.3]
    # ============================================================
    fig1, ax1 = plt.subplots(figsize=(6, 4))

    for method, (x, y) in method_curves.items():
        ax1.plot(x, y, label=method)

    ax1.set_xlabel("NMAE")
    ax1.set_ylabel("Fraction of figures (CDF)")
    ax1.set_xlim(0.0, 0.3)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    plt.tight_layout()

    if output_dir is not None:
        fig1.savefig(
            output_dir / "error_cdf_nmae_zoom_0_0.3.png",
            dpi=dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig1)

    # ============================================================
    # Figure 2: Zoom on [0.3, 1.0]
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    for method, (x, y) in method_curves.items():
        ax2.plot(x, y, label=method)

    ax2.set_xlabel("NMAE")
    ax2.set_ylabel("Fraction of figures (CDF)")
    ax2.set_xlim(0.3, 1.0)

    # Optional: tighten y-range automatically (true zoom)
    y_vals = []
    for x, y in method_curves.values():
        mask = (x >= 0.3) & (x <= 1.0)
        if np.any(mask):
            y_vals.append(y[mask].min())
            y_vals.append(y[mask].max())

    if y_vals:
        ymin, ymax = min(y_vals), max(y_vals)
        pad = 0.02 * (ymax - ymin)
        ax2.set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))

    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if output_dir is not None:
        fig2.savefig(
            output_dir / "error_cdf_nmae_zoom_0.3_1.0.png",
            dpi=dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig2)



if __name__ == "__main__":
    main(
        output_dir="examples_thesis",
        show=False,
    )

    