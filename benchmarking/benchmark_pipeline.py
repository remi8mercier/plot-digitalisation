# benchmark_pipeline.py
import os
import json
import glob
import math
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, List, Optional, Tuple

import pandas as pd

# --- Your judge module (single source of truth for metrics) ---
from judge_generated_points import judge_model_output  # <- all scoring goes through this


# ---------------------------------
# Data structures to store results
# ---------------------------------
@dataclass
class CaseSpec:
    image_path: str
    gt_path: str
    case_id: str


@dataclass
class CaseResult:
    case_id: str
    image_path: str
    gt_path: str
    success: bool
    metrics: Dict[str, Any]
    error: Optional[str] = None
    # Optional artifacts you might want to log
    extractor_seconds: Optional[float] = None
    judge_seconds: Optional[float] = None
    model_series_count: Optional[int] = None
    gt_series_count: Optional[int] = None
    extracted_points: Any=None


# ---------------------------
# Utilities
# ---------------------------
def _read_csv_ground_truth(csv_path: str) -> Dict[str, Any]:
    """
    Universal GT parser supporting:
      A) Paired-wide, 2-row header with multiple rows -> multiple series
      B) Paired-wide, 2-row header with ONE data row -> single combined series
      C) Paired-wide single-row header (Naik etc.)
      D) Long form (x,y,label)
      E) Wide form (x + many y columns)
    """

    import pandas as pd
    import numpy as np

    # Read raw (no headers) to detect 2-row paired-wide patterns
    df_raw = pd.read_csv(csv_path, header=None)

    # ------------------------------------------------------------
    # Detect paired-wide 2-row header format
    # ------------------------------------------------------------
    first = df_raw.iloc[0].tolist()
    second = df_raw.iloc[1].tolist()

    is_XY = lambda v: isinstance(v, str) and v.strip().upper() in ("X", "Y")

    if all(is_XY(v) for v in second) and len(second) % 2 == 0:
        # How many data rows?
        data = df_raw.iloc[2:]
        n_data_rows = len(data)

        # ----------------------------------------------------
        # CASE B — Single data row → multiple single-point series (categorical)
        # ----------------------------------------------------
        if n_data_rows == 1:
            row = data.iloc[0]
            num_series = len(second) // 2

            series_out = []

            for s in range(num_series):
                col_x = 2 * s
                col_y = col_x + 1

                label = str(first[col_x]).strip()
                x = pd.to_numeric(row[col_x], errors="coerce")
                y = pd.to_numeric(row[col_y], errors="coerce")

                if pd.isna(x) or pd.isna(y):
                    continue

                series_out.append({
                    "label": label,
                    "points": [{"x": float(x), "y": float(y)}],
                })

            if series_out:
                return {"series": series_out}

        # ----------------------------------------------------
        # CASE A — Normal paired-wide -> multiple curves
        # ----------------------------------------------------
        series_out = []
        num_series = len(second) // 2

        for s in range(num_series):
            col_x = 2 * s
            col_y = col_x + 1
            label = str(first[col_x]).strip()

            sub = df_raw.iloc[2:, [col_x, col_y]]
            xs = pd.to_numeric(sub.iloc[:, 0], errors="coerce")
            ys = pd.to_numeric(sub.iloc[:, 1], errors="coerce")

            mask = (~xs.isna()) & (~ys.isna())
            pts = [{"x": float(a), "y": float(b)}
                   for a, b in zip(xs[mask], ys[mask])]

            if pts:
                series_out.append({"label": label, "points": pts})

        if series_out:
            return {"series": series_out}

    # ------------------------------------------------------------
    # CASE C — Single-row paired-wide header (Naik format)
    # ------------------------------------------------------------
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()

    if len(cols) % 2 == 0:
        possible = True
        series_out = []
        for i in range(0, len(cols), 2):
            xcol = cols[i]
            ycol = cols[i+1]

            xs = pd.to_numeric(df[xcol], errors="coerce")
            ys = pd.to_numeric(df[ycol], errors="coerce")

            mask = (~xs.isna()) & (~ys.isna())
            if mask.sum() == 0:
                possible = False
                break

            pts = [{"x": float(a), "y": float(b)} for a, b in zip(xs[mask], ys[mask])]
            series_out.append({"label": str(xcol), "points": pts})

        if possible and series_out:
            return {"series": series_out}

    # ------------------------------------------------------------
    # CASE D — Long form
    # ------------------------------------------------------------
    lower_cols = [c.lower() for c in df.columns]
    x_candidates = [c for c in df.columns if c.lower() in ("x", "time", "epoch")]
    y_candidates = [c for c in df.columns if c.lower() in ("y", "value")]
    label_candidates = [c for c in df.columns if any(t in c.lower() for t in ("label", "series", "group"))]

    if x_candidates and y_candidates and label_candidates:
        xcol, ycol, lcol = x_candidates[0], y_candidates[0], label_candidates[0]
        series_out = []
        for lab, g in df.groupby(lcol):
            xs = pd.to_numeric(g[xcol], errors="coerce")
            ys = pd.to_numeric(g[ycol], errors="coerce")
            m = (~xs.isna()) & (~ys.isna())
            pts = [{"x": float(a), "y": float(b)} for a, b in zip(xs[m], ys[m])]
            if pts:
                series_out.append({"label": str(lab), "points": pts})
        if series_out:
            return {"series": series_out}

    # ------------------------------------------------------------
    # CASE E — Wide form (x + many y columns)
    # ------------------------------------------------------------
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        xcol = numeric_cols[0]
        ycols = numeric_cols[1:]
        series_out = []
        for col in ycols:
            xs = pd.to_numeric(df[xcol], errors="coerce")
            ys = pd.to_numeric(df[col], errors="coerce")
            m = (~xs.isna()) & (~ys.isna())
            pts = [{"x": float(a), "y": float(b)} for a, b in zip(xs[m], ys[m])]
            if pts:
                series_out.append({"label": str(col), "points": pts})
        if series_out:
            return {"series": series_out}

    raise ValueError(f"Could not parse ground-truth CSV: {csv_path}")

def _standardize_model_output(model_obj: Any) -> Dict[str, Any]:
    """
    Coerce the extractor's output into the canonical schema expected by the judge:
      {"series": [{"label": str, "points": [{"x": float, "y": float}, ...]}]}
    """
    # Already in place?
    if isinstance(model_obj, dict) and "series" in model_obj:
        return model_obj

    # Try json string
    if isinstance(model_obj, str):
        try:
            model_obj = json.loads(model_obj)
        except Exception as e:
            raise ValueError(f"Extractor returned a string that's not JSON: {e}")

    # Minimal coercion fallback
    if isinstance(model_obj, dict) and "points" in model_obj:
        return {"series": [{"label": model_obj.get("label", "series_1"),
                            "points": model_obj["points"]}]}

    raise ValueError("Extractor output cannot be coerced to the canonical schema.")


def _discover_cases(
    data_dir: str,
    image_glob: str = "*.png",
    csv_suffix: str = ".csv",
) -> List[CaseSpec]:

    """
    Recursively search for case folders.

    For every directory D under data_dir:
        If D contains   D/D.png   AND   D/D.csv,
        we register a CaseSpec.

    Also supports the flat form:
        D.png and D.csv directly in data_dir.
    """
    cases: List[CaseSpec] = []

    # 1. Flat scan (backward-compatible)
    for img_path in glob.glob(os.path.join(data_dir, image_glob)):
        base, _ = os.path.splitext(img_path)
        csv_path = base + csv_suffix
        if os.path.exists(csv_path):
            case_id = os.path.basename(base)
            cases.append(CaseSpec(image_path=img_path, gt_path=csv_path, case_id=case_id))

    # 2. Recursive scan for /X/X.png + /X/X.csv
    for root, dirs, files in os.walk(data_dir):
        # skip root in flat mode to avoid duplicates
        if root.rstrip("/\\") == data_dir.rstrip("/\\"):
            continue

        folder_name = os.path.basename(root.rstrip("/\\"))
        img_path = os.path.join(root, folder_name + ".png")
        csv_path = os.path.join(root, folder_name + csv_suffix)

        if os.path.exists(img_path) and os.path.exists(csv_path):
            cases.append(
                CaseSpec(
                    image_path=img_path,
                    gt_path=csv_path,
                    case_id=folder_name,
                )
            )

    # Sort deterministically
    cases.sort(key=lambda c: c.case_id)
    return cases


# ---------------------------
# Core execution
# ---------------------------
def run_single_case(
    case: CaseSpec,
    extractor_fn: Callable[[str], Any],
    debugging: bool = False,
) -> CaseResult:
    """
    1) Run your extraction function on the image
    2) Standardize both model & real objects
    3) Delegate scoring to judge_model_output(...)
    """
    t0 = time.time()
    try:
        raw = extractor_fn(case.image_path)
        
        extractor_seconds = time.time() - t0

        model_obj = _standardize_model_output(raw)
        gt_obj = _read_csv_ground_truth(case.gt_path)

        # >>> judge call
        t1 = time.time()
        if debugging == True:
            print("gt_obj:", truncate_model_series_points(gt_obj))
            print("model_obj", truncate_model_series_points(model_obj))
        metrics = judge_model_output(model_obj=model_obj, real_obj=gt_obj)
        judge_seconds = time.time() - t1
        return CaseResult(
            case_id=case.case_id,
            image_path=case.image_path,
            gt_path=case.gt_path,
            success=True,
            metrics=metrics,
            extractor_seconds=extractor_seconds,
            judge_seconds=judge_seconds,
            extracted_points=model_obj,      # NEW
            model_series_count=len(model_obj.get("series", [])),
            gt_series_count=len(gt_obj.get("series", [])),
        )
    except Exception as e:
        return CaseResult(
            case_id=case.case_id,
            image_path=case.image_path,
            gt_path=case.gt_path,
            extracted_points=None,
            success=False,
            metrics={},
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
    
def load_completed_case_ids_from_dir(path: str) -> set:
    """
    If `path` is a directory, return all its subfolder names.
    These folder names represent completed case_ids.

    If `path` is a file, fallback to reading JSONL lines.
    """

    import os, json

    completed = set()

    # If it's a directory, read its subfolders
    if os.path.isdir(path):
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isdir(full):
                completed.add(name)
        return completed

    # --- Otherwise treat it as a JSONL file ---
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    cid = obj.get("result", {}).get("case_id")
                    if cid:
                        completed.add(cid)
                except Exception:
                    continue

    return completed
def truncate_model_series_points(
    model_obj: dict,
    max_points: int = 5,
) -> dict:
    """
    Keep only the first `max_points` (by x) for each model series. This is for cleaner debugging prints, not for scoring (the judge gets the full output).
    """

    if not model_obj or "series" not in model_obj:
        return model_obj

    out_series = []

    for s in model_obj["series"]:
        pts = s.get("points", [])
        if not pts:
            out_series.append(s)
            continue

        # sort by x, then take first N
        pts_sorted = sorted(pts, key=lambda p: float(p["x"]))
        pts_trunc = pts_sorted[:max_points]

        out_series.append({
            "label": s.get("label"),
            "points": pts_trunc,
        })

    return {"series": out_series}

def run_benchmark(
    data_dir: str,
    extractor_fn: Callable[[str], Any],
    k: Optional[int] = None,
    image_glob: str = "*.png",
    csv_suffix: str = ".csv",
    save_path: Optional[str] = None,
    redo: bool = True,
) -> Tuple[pd.DataFrame, List[CaseResult]]:
    """
    Execute the benchmark on (up to) k cases found in data_dir.

    Redo indicates if we want to finish the benchmark or restart from scratch.
    If we redo, the pdf will only have the cases that were done recently, but get_metris_from_folders will work.
    Returns:
      (summary_df, results_list)
    - summary_df: one row per case with top-level metrics plus timings
    - results_list: the structured results (for saving or further analysis)
    """
    cases = _discover_cases(data_dir, image_glob=image_glob, csv_suffix=csv_suffix)
    print(f"Discovered {len(cases)} cases in {data_dir}")

    if not redo:
        completed = (
            load_completed_case_ids_from_dir(save_path)
            if save_path else set()
        )

        if completed:
            print(f"Skipping {len(completed)} already-completed cases found in {save_path}")

        cases = [c for c in cases if c.case_id not in completed]
        print(f"{len(cases)} cases remaining to process.")

    cases = cases[:k] if k is not None else cases
    results: List[CaseResult] = []
    rows: List[Dict[str, Any]] = []
    os.makedirs(save_path, exist_ok=True)
    save_jsonl_path = save_path + "benchmark_results.jsonl"

    for case in cases:
        res = run_single_case(case, extractor_fn)
        results.append(res)
        # Flatten top-level metrics (standard keys common in the project)
        row = {
            "case_id": res.case_id,
            "success": res.success,
            "image_path": res.image_path,
            "gt_path": res.gt_path,
            "extractor_seconds": res.extractor_seconds,
            "judge_seconds": res.judge_seconds,
            "model_series_count": res.model_series_count,
            "gt_series_count": res.gt_series_count,
            "metrics": res.metrics,
        }
        if res.success:
            # Pull out common numeric metrics if present
            for k_ in ("MAE", "NMAE", "Bias", "r", "x_overlap_ratio"):
                if k_ in res.metrics:
                    row[k_] = res.metrics[k_]
            if "x_misaligned" in res.metrics:
                row["x_misaligned"] = bool(res.metrics["x_misaligned"])
            # -----------------------------
            # SAVE PER-CASE OUTPUTS
            # -----------------------------
            case_dir = os.path.join(save_path, case.case_id)
            os.makedirs(case_dir, exist_ok=True)
            print(f"Saved outputs to: {case_dir}")
            # Save model-extracted points
            with open(os.path.join(case_dir, "extracted_points.json"), "w", encoding="utf-8") as f:
                json.dump(res.extracted_points, f, indent=2)

            # Save comparison metrics
            with open(os.path.join(case_dir, "comparison_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(res.metrics, f, indent=2)

        else:
            row["error"] = res.error

        rows.append(row)

    df = pd.DataFrame(rows)

    # --------------------------------------------------
    # SAVE / APPEND TO JSONL
    # --------------------------------------------------
    if save_jsonl_path:
        # If redo=True → start a fresh JSONL file
        # If redo=False → append only new case results
        write_mode = "w" if redo else "a"

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_jsonl_path), exist_ok=True)

        with open(save_jsonl_path, write_mode, encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps({"result": asdict(r)}, ensure_ascii=False) + "\n")

    # --------------------------------------------------
    # CLEAN UP STALE FAILURES (redo=False mode)
    # --------------------------------------------------
    if not redo and save_path:
        # Case IDs that failed in THIS run
        failed_this_run = {
            r.case_id for r in results if not r.success
        }

        # All known case IDs on disk
        all_case_ids = {
            c.case_id for c in _discover_cases(
                data_dir,
                image_glob=image_glob,
                csv_suffix=csv_suffix
            )
        }

        # Case IDs with persisted outputs (i.e. successful at least once)
        persisted_success = {
            name for name in os.listdir(save_path)
            if os.path.isdir(os.path.join(save_path, name))
        }

        # Historical failures = known cases without a success folder
        historical_failures = all_case_ids - persisted_success

        # Stale failures = previously failed but NOT failing anymore
        stale_failures = historical_failures - failed_this_run

        if stale_failures:
            print(f"Removing {len(stale_failures)} stale failures:")
            for cid in sorted(stale_failures):
                print(f"  ✓ {cid} is no longer failing")

        # Optional: rewrite JSONL to drop stale failure entries
        jsonl_path = save_path + "benchmark_results.jsonl"
        if os.path.exists(jsonl_path) and stale_failures:
            cleaned = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        cid = obj.get("result", {}).get("case_id")
                        success = obj.get("result", {}).get("success", True)
                        # keep:
                        #  - anything successful
                        #  - failures that are still failing
                        if success or cid not in stale_failures:
                            cleaned.append(line)
                    except Exception:
                        continue

            with open(jsonl_path, "w", encoding="utf-8") as f:
                for line in cleaned:
                    f.write(line)

    return df, results


# ---------------------------
# Convenience: complete_pipeline
# ---------------------------
def complete_pipeline(
    image_path: str,
    extractor_fn: Callable[[str], Any],
) -> Dict[str, Any]:
    """
    Minimal 'do-it-all' helper for a single image:
      - run extractor
      - standardize output
      - return model_obj
    This mirrors the intent of your former `complete_fn`.
    """
    raw = extractor_fn(image_path)
    model_obj = _standardize_model_output(raw)
    return model_obj


# ---------------------------
# CLI entry (optional)
# ---------------------------
if __name__ == "__main__":
    import os
    import json
    from gemini_simple import extract_plot_data_gemini_streaming
    from gpt_simple import single_step_simple
    from pipeline_core import extract_plot_data_gemini
    #from gemini_km import run_unified_plot_extractor
    from gemini_pipeline import run_unified_plot_extractor
    #from gemini_pipeline_km_snapping import run_unified_plot_extractor

    from pathlib import Path
    from plot_extracted_points import plot_extracted_points
    # ---------------------------------------------------------
    # USER-FREE DEFAULTS (no CLI arguments)
    # ---------------------------------------------------------
    DATA_DIR = "data/ExtractedDataMatthias/"                # folder containing <case>/<case>.png + <case>.csv
    IMAGE_GLOB = "*.png"      # glob pattern to find images
    CSV_SUFFIX = ".csv"
    LIMIT_K = 50              # set to an integer to limit the first K cases
    SAVE_JSONL_PATH = os.path.join("data/", "benchmark_results_gpt_simple/")

    # ---------------------------------------------------------
    # Helper to remove ```json / ``` fences
    # ---------------------------------------------------------
    def strip_json_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```json"):
            t = t[len("```json"):].strip()
        elif t.startswith("```"):
            t = t[len("```"):].strip()
        if t.endswith("```"):
            t = t[: -len("```")].strip()
        return t

    # ---------------------------------------------------------
    # GEMINI EXTRACTOR WRAPPER
    # Converts Gemini output → canonical benchmark schema
    # ---------------------------------------------------------
    def extractor_fn(image_path: str):
        """The extraction function.
        It can be a simple LLM call, or a more complex pipeline.
        """
        #raw_text = extract_plot_data_gemini_streaming(image_file_path=image_path,)
        #raw_text = extract_plot_data_gemini(image_path=image_path)
        #raw_text = run_unified_plot_extractor(image_path=image_path)
        raw_text = single_step_simple(image_path=image_path)
        if not isinstance(raw_text, dict):
            # 3. Parse JSON
            try:
                cleaned = strip_json_fences(raw_text)
                obj = json.loads(cleaned)
            except Exception as e:
                raise ValueError(f"Extractor produced invalid JSON: {e}\nRaw output:\n{cleaned}")
        else:
            obj = raw_text
        if "series" not in raw_text:
            raise ValueError(f"Extractor output missing 'series' key: {raw_text}")

        # 4. Return as canonical benchmark schema
        # Benchmark accepts {"series":[{label:"", points:[{x,y}]}]}
        return obj
    # ---------------------------------------------------------
    # RUN BENCHMARK
    # ---------------------------------------------------------
    summary_df, results = run_benchmark(
        data_dir=DATA_DIR,
        extractor_fn=extractor_fn,
        k=LIMIT_K,
        image_glob=IMAGE_GLOB,
        csv_suffix=CSV_SUFFIX,
        save_path=SAVE_JSONL_PATH,
        redo=False,
    )
    plot_extracted_points(directory=Path(SAVE_JSONL_PATH))
    # ---------------------------------------------------------
    # PRINT SUMMARY
    # ---------------------------------------------------------
    print("\nBenchmark complete.")
    print("Saved results to:", SAVE_JSONL_PATH)
    print(summary_df.head())
