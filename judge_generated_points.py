# judge_generated_points.py — axis-aware scoring (X & Y separated, no joint score)

import re, json, math
import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
from collections import defaultdict


def _norm(v, rng):
    """Normalize a single value by the range, with a small epsilon to avoid division by zero."""
    return abs(v) / max(rng, 1e-12)
def _fmt(v, fmt):
    """Safely format a value with a given format string, returning "NA" if formatting fails."""
    if v is None:
        return "NA"
    try:
        return format(v, fmt)
    except Exception:
        return "NA"

def _single_point_cost(ms, rs, x_rng, y_rng, *,
                       is_categorical=False,
                       model_index=None,
                       real_index=None,
                       ordinal_weight=0.1):
    """
    Cost for single-point series.

    Continuous:
        normalized |Δx| + |Δy|

    Categorical:
        |Δy| + ordinal_weight * |index difference|
    """

    my = ms["y"][0]
    ry = rs["y"][0]

    # --------------------------------------------------
    # CATEGORICAL MODE (ignore x magnitude)
    # --------------------------------------------------
    if is_categorical:
        if model_index is None or real_index is None:
            raise ValueError("Categorical matching requires model_index and real_index")

        dy = abs(my - ry)
        dord = abs(model_index - real_index)

        return dy + ordinal_weight * dord

    # --------------------------------------------------
    # CONTINUOUS MODE (original behavior)
    # --------------------------------------------------
    mx = ms["x"][0]
    rx = rs["x"][0]

    return _norm(mx - rx, x_rng) + _norm(my - ry, y_rng)

def _point_to_curve_cost(px, py, cx, cy, x_rng, y_rng):
    """
    Cost of matching a single point (px, py) to a curve defined by points (cx, cy).
    """
    return min(
        _norm(px - x, x_rng) + _norm(py - y, y_rng)
        for x, y in zip(cx, cy)
    )

def expand_duplicate_x_with_continuity(points, eps_frac=0.01):
    """
    Expands points so that:
      - x is strictly increasing
      - for duplicate x:
          * first y is closest to previous y
          * second y is furthest from previous y
          * second point gets +epsilon in x
    """

    if not points:
        return []

    # group by x (preserve original order of points)
    by_x = defaultdict(list)
    for p in points:
        by_x[p["x"]].append(float(p["y"]))

    xs = sorted(by_x.keys())

    # compute epsilon
    x_min, x_max = min(xs), max(xs)
    x_range = x_max - x_min
    eps = eps_frac * x_range if x_range > 0 else eps_frac

    out = []
    y_prev = None

    for x in xs:
        ys = by_x[x]

        if y_prev is None:
            # first x: choose median as stable start
            y0 = float(np.median(ys))
            out.append({"x": x, "y": y0, "synthetic_x": False})
            y_prev = y0
            continue

        if len(ys) == 1:
            y0 = ys[0]
            out.append({"x": x, "y": y0, "synthetic_x": False})
            y_prev = y0
            continue

        # --- CONTINUITY LOGIC ---
        # closest to previous y
        y_closest = min(ys, key=lambda v: abs(v - y_prev))

        # furthest from previous y
        y_furthest = max(ys, key=lambda v: abs(v - y_prev))

        # first point (no shift)
        out.append({
            "x": x,
            "y": y_closest,
            "synthetic_x": False
        })

        # second point (epsilon-shifted)
        if y_furthest != y_closest:
            out.append({
                "x": x + eps,
                "y": y_furthest,
                "synthetic_x": True,
                "x_base": x
            })
            y_prev = y_furthest
        else:
            y_prev = y_closest

    return out


def _safe_greedy_assign(cost):
    """Fallback greedy assignment for cases where Hungarian fails (e.g. all inf costs)."""
    M, R = cost.shape
    row_idx, col_idx = [], []
    used_rows, used_cols = set(), set()

    cost_copy = cost.copy()

    # maximum number of assignments possible
    target = min(M, R)

    while len(row_idx) < target:
        # If no finite values left → stop
        if not np.isfinite(cost_copy).any():
            break

        # Find global minimum finite cost
        idx = np.nanargmin(cost_copy)   # returns int
        i, j = divmod(int(idx), R)

        # If this cell is useless, invalidate and continue
        if i in used_rows or j in used_cols or not np.isfinite(cost_copy[i, j]):
            cost_copy[i, j] = np.inf
            continue

        # Accept pair
        row_idx.append(i)
        col_idx.append(j)
        used_rows.add(i)
        used_cols.add(j)

        # Invalidate row + column
        cost_copy[i, :] = np.inf
        cost_copy[:, j] = np.inf

    return np.array(row_idx, int), np.array(col_idx, int)


def judge_model_output(
    model_obj: dict,
    real_obj: dict,
    verbose: bool = True,
    x_misaligned_overlap_threshold: float = 0.2,
    graph_type: str = "categorical"
):
    """
    Unified scoring function comparing extracted series (model_obj)
    with ground-truth series (real_obj).

    Adds two new metrics per matched series pair:
        • ks: Kolmogorov–Smirnov-like sup norm on matched domain
        • integral_l1: ∫ |m(x) - g(x)| dx over overlapping region

    And two global aggregates:
        • overall_ks
        • overall_integral_l1
    """
    # ============================================================
    # Internal helpers
    # ============================================================

    def _clean_json_string(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            lines = s.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        return s

    def _is_numeric(x):
        try:
            float(x)
            return True
        except Exception:
            return False

    def _extract_model_series(model_obj):
        out = []
        for s in model_obj.get("series", []):
            label = str(s.get("label", f"series_{len(out)+1}"))
            pts = s.get("points", [])
            if not pts:
                continue
            pts = expand_duplicate_x_with_continuity(pts, eps_frac=0.01)
            x = np.array([p.get("x", np.nan) for p in pts], dtype=float)
            y = np.array([p.get("y", np.nan) for p in pts], dtype=float)

            # sort but track original index for exclusion reporting
            order = np.argsort(x, kind="mergesort")
            out.append({
                "label": label,
                "x": x[order],
                "y": y[order],
                "orig_idx": np.arange(len(x))[order]
            })
        return out
        
    def _normalize_real_series(real_obj, *, dedup_x: bool = True):
        """
        real_obj = {"series":[{"label":..., "points":[{x,y},...]}, ...]}

        Returns:
        dict[label] = {"x": np.ndarray|None, "y": np.ndarray}
            - If x is entirely missing/NaN but y exists: x=None and y is finite.
            - Otherwise: x,y are finite arrays sorted by x.
        """
        out = {}

        for s in real_obj.get("series", []):
            name = str(s.get("label", f"real_{len(out)+1}"))
            pts = s.get("points", [])
            if not pts:
                continue
            
            pts = expand_duplicate_x_with_continuity(pts, eps_frac=0.01)
            x = np.array([p.get("x", np.nan) for p in pts], dtype=float)
            y = np.array([p.get("y", np.nan) for p in pts], dtype=float)

            # Keep only finite y 
            y_finite = np.isfinite(y)

            # If x is entirely missing/NaN, fall back to indexwise mode
            if not np.isfinite(x).any():
                y2 = y[y_finite]
                if y2.size == 0:
                    continue
                out[name] = {"x": None, "y": y2}
                continue

            # Otherwise require both x and y to be finite
            keep = np.isfinite(x) & y_finite
            x2, y2 = x[keep], y[keep]
            if x2.size == 0:
                continue

            # Sort by x
            order = np.argsort(x2, kind="mergesort")
            x2, y2 = x2[order], y2[order]

            # deduplicate identical x values by averaging y
            if dedup_x and x2.size > 1:
                ux, inv = np.unique(x2, return_inverse=True)
                if ux.size != x2.size:
                    y_acc = np.zeros_like(ux, dtype=float)
                    cnt = np.zeros_like(ux, dtype=float)
                    np.add.at(y_acc, inv, y2)
                    np.add.at(cnt, inv, 1.0)
                    y2 = y_acc / np.maximum(cnt, 1.0)
                    x2 = ux

            out[name] = {"x": x2, "y": y2}

        return out

    def _interp_mae(model_x, model_y, real_x, real_y):
        """Compute MAE between model and real by interpolating real_y at model_x points."""
        # overlapping domain
        if model_x.size < 2 or real_x.size < 2:
            return (math.inf, 0, (float("nan"), float("nan")), (float("nan"), float("nan")), 
                    np.array([]), np.array([]), np.array([]))  # + my2, ry2, mx2 empty

        x0 = max(np.nanmin(model_x), np.nanmin(real_x))
        x1 = min(np.nanmax(model_x), np.nanmax(real_x))
        if not (x1 > x0):
            return (math.inf, 0, (float("nan"), float("nan")), (x0, x1),
                    np.array([]), np.array([]), np.array([]))

        mask = (model_x >= x0) & (model_x <= x1) & ~np.isnan(model_x) & ~np.isnan(model_y)
        if mask.sum() < 2:
            return (math.inf, int(mask.sum()), (float("nan"), float("nan")), (x0, x1),
                    np.array([]), np.array([]), np.array([]))

        mx2 = model_x[mask]
        my2 = model_y[mask]
        ry2 = np.interp(mx2, real_x, real_y)

        mae = float(np.mean(np.abs(my2 - ry2)))
        ymin, ymax = float(min(my2.min(), ry2.min())), float(max(my2.max(), ry2.max()))
        return mae, int(mask.sum()), (ymin, ymax), (x0, x1), mx2, my2, ry2

    def _indexwise_mae(model_y, real_y):
        """Compute MAE between model and real by comparing y values at matching indices (for x=None cases)."""
        n = min(model_y.size, real_y.size)
        if n == 0:
            return (math.inf, 0, (float("nan"), float("nan")),
                    np.array([]), np.array([]))  # a2, b2 empty

        a = model_y[:n]
        b = real_y[:n]
        nan_mask = np.isnan(a) | np.isnan(b)
        idx_good = np.where(~nan_mask)[0]
        if idx_good.size == 0:
            return (math.inf, 0, (float("nan"), float("nan")),
                    np.array([]), np.array([]))

        a2 = a[idx_good]
        b2 = b[idx_good]
        mae = float(np.mean(np.abs(a2 - b2)))
        ymin, ymax = float(min(a2.min(), b2.min())), float(max(a2.max(), b2.max()))
        return mae, int(idx_good.size), (ymin, ymax), a2, b2

    def _pearsonr(a, b):
        """Compute Pearson correlation, returning NaN if not computable."""
        if a.size < 2 or b.size < 2:
            return float("nan")
        am = a - a.mean()
        bm = b - b.mean()
        denom = np.linalg.norm(am) * np.linalg.norm(bm)
        if denom == 0:
            return float("nan")
        return float(np.dot(am, bm) / denom)

    # ============================================================
    # Parse and extract
    # ============================================================
    if isinstance(model_obj, str):
        model_obj = json.loads(_clean_json_string(model_obj))

    model_series = _extract_model_series(model_obj)
    if not model_series:
        raise ValueError("No series found in model output.")

    real_series = _normalize_real_series(real_obj)
    if not real_series:
        raise ValueError("No series found in real_obj.")

    model_names = [s["label"] for s in model_series]
    real_names = list(real_series.keys())

    M, R = len(model_series), len(real_names)
    cost = np.full((M, R), np.inf, float)
    aux = {}
    
    # ============================================================
    # Global ranges for normalization (needed for single-point)
    # ============================================================
    all_x, all_y = [], []
    for s in model_series:
        all_x.extend(s["x"])
        all_y.extend(s["y"])
    for rs in real_series.values():
        all_x.extend(rs["x"])
        all_y.extend(rs["y"])

    x_rng = max(all_x) - min(all_x) if len(all_x) > 1 else 1.0
    y_rng = max(all_y) - min(all_y) if len(all_y) > 1 else 1.0
    # ============================================================
    # Cost matrix (single-point aware) and aux storage
    # ============================================================
    for i, ms in enumerate(model_series):
        for j, rn in enumerate(real_names):
            rs = real_series[rn]
            # ----------------------------------------------------
            # CASE 1: single point ↔ single point
            # ----------------------------------------------------
            if ms["x"].size == 1 and rs["x"].size == 1:
                cost_ij = _single_point_cost(
                    ms, rs,
                    x_rng, y_rng,
                    is_categorical=True,
                    model_index=i,
                    real_index=j
                )

                cost[i, j] = cost_ij
                aux[(i, j)] = (
                    "single_point",
                    cost_ij,
                    1,
                    (rs["y"][0], ms["y"][0]),
                    np.array([]),
                    np.array([]),
                )
            # ----------------------------------------------------
            # CASE 2: single point ↔ curve
            # ----------------------------------------------------
            elif ms["x"].size == 1 and rs["x"].size > 1:
                cost_ij = _point_to_curve_cost(
                    ms["x"][0], ms["y"][0],
                    rs["x"], rs["y"],
                    x_rng, y_rng
                )

                cost[i, j] = cost_ij
                aux[(i, j)] = ("point_to_curve", cost_ij, 1, None, None, None)
            elif ms["x"].size > 1 and rs["x"].size == 1:
                cost_ij = _point_to_curve_cost(
                    rs["x"][0], rs["y"][0],
                    ms["x"], ms["y"],
                    x_rng, y_rng
                )

                cost[i, j] = cost_ij
                aux[(i, j)] = ("curve_to_point", cost_ij, 1, None, None, None)
               

            # ----------------------------------------------------
            # CASE 3: curve ↔ curve (your existing logic)
            # ----------------------------------------------------
            elif rs["x"] is not None:
                mae, used_n, yrng, (x0, x1), mx2, my2, ry2 = _interp_mae(
                    ms["x"], ms["y"], rs["x"], rs["y"]
                )
                cost[i, j] = mae
                aux[(i, j)] = ("interp", mae, used_n, yrng, (x0, x1), mx2, my2, ry2)
                print("multiples points for both", mae)

            else:
                mae, used_n, yrng, a2, b2 = _indexwise_mae(ms["y"], rs["y"])
                cost[i, j] = mae
                aux[(i, j)] = ("index", mae, used_n, yrng, a2, b2)
                print("failsafe")

    row_idx, col_idx = [], []
    try:
        row_idx, col_idx = hungarian(cost)
    except Exception:
        row_idx, col_idx = _safe_greedy_assign(cost)

    if len(row_idx) == 0:
        print("No feasible assignment between model and GT series.")
        return {
            "pairs": [],
            "unmatched_model_series": [s["label"] for s in model_series],
            "unmatched_real_series": list(real_series.keys()),
            "overall_nmae": float("nan"),
            "overall_ks": float("nan"),
            "overall_integral_l1": float("nan"),
            "overall_x_misaligned": True,
            "model_series_count": len(model_series),
            "gt_series_count": len(real_series),
            "notes": "No feasible assignment between model and GT series."
        }
    # ============================================================
    # Build result pairs
    # ============================================================
    pairs = []
    used_model = set()
    used_real = set()

    for i, j in zip(row_idx, col_idx):
        ms = model_series[i]
        rn = real_names[j]

        info = aux[(i, j)]
        kind = info[0]
        gt_all_y = np.concatenate([rs["y"] for rs in real_series.values()])
        global_ymin = float(np.min(gt_all_y))
        global_ymax = float(np.max(gt_all_y))
        global_y_range = global_ymax - global_ymin

        # Common fields
        mae = info[1]
        n_used = info[2]
        (ymin, ymax) = info[3]
        y_range = abs(ymax - ymin) if (not math.isnan(ymax) and not math.isnan(ymin)) else float("nan")
        nmae = mae / y_range if (y_range and y_range > 0 and math.isfinite(mae)) else float("nan")

        excluded = {}
        ks = None
        ks_scaled = None
        # --------------------------------------------------------
        # SINGLE POINT (categorical)
        # --------------------------------------------------------
        if kind == "single_point":
            m_y = float(ms["y"][0])
            g_y = float(real_series[rn]["y"][0])
            m_x = float(ms["x"][0])
            g_x = float(real_series[rn]["x"][0])

            mae = abs(m_y - g_y)
            bias = m_y - g_y

            # ---- CORRECT single-point NMAE ----
            if global_y_range > 0:
                nmae = mae / global_y_range
            else:
                nmae = float("nan")
            # ---- metrics that do NOT apply ----    
            x_misaligned = None
            # ---- metrics that do NOT apply ----
            corr = None
            ks = None
            integral_l1 = None
            overlap_ratio = None

            ks_scaled = None
            l1_scaled = None
            ks_scaled_y = None
            l1_scaled_y = None

            excluded = {
                "pearson_r": "single_point",
                "ks": "single_point",
                "integral_l1": "single_point",
                "overlap_ratio": "single_point",
            }


        # --------------------------------------------------------
        # INTERPOLATED (continuous curves)
        # --------------------------------------------------------
        elif kind == "interp":
            (x0, x1) = info[4]
            mx2, my2, ry2 = info[5], info[6], info[7]

            union_len = float(
                max(np.nanmax(ms["x"]), np.nanmax(real_series[rn]["x"]))
                - min(np.nanmin(ms["x"]), np.nanmin(real_series[rn]["x"]))
            )
            overlap_len = float(max(0.0, x1 - x0))
            overlap_ratio = (overlap_len / union_len) if union_len > 0 else float("nan")

            if mx2.size > 0:
                corr = _pearsonr(my2, ry2)
                bias = float(np.mean(my2 - ry2))
                ks = float(np.max(np.abs(my2 - ry2)))
                integral_l1 = float(np.trapz(np.abs(my2 - ry2), mx2))

                gt_max = float(np.nanmax(ry2)) if np.isfinite(ry2).any() else None
                gt_yrange = float(np.nanmax(real_series[rn]["y"]) - np.nanmin(real_series[rn]["y"]))
              
                ks_scaled = ks / gt_max if gt_max else None
                l1_scaled = integral_l1 / gt_max if gt_max else None
                ks_scaled_y = ks / gt_yrange if y_range > 0 else None
                l1_scaled_y = integral_l1 / y_range if y_range > 0 else None

            else:
                corr = bias = ks = integral_l1 = None
                ks_scaled = l1_scaled = ks_scaled_y = l1_scaled_y = None

            x_misaligned = (
                overlap_ratio is not None
                and np.isfinite(overlap_ratio)
                and overlap_ratio < x_misaligned_overlap_threshold
            )

            excluded = {}

        # --------------------------------------------------------
        # INDEXWISE
        # --------------------------------------------------------
        else:
            a2, b2 = info[4], info[5]
            overlap_ratio = None

            if a2.size > 0:
                corr = _pearsonr(a2, b2)
                bias = float(np.mean(a2 - b2))
                ks = float(np.max(np.abs(a2 - b2)))
                integral_l1 = float(np.sum(np.abs(a2 - b2)))
            else:
                corr = bias = ks = integral_l1 = None

            ks_scaled = l1_scaled = ks_scaled_y = l1_scaled_y = None
            x_misaligned = False
            excluded = {}


        # x-alignment flag
        x_misaligned = (    overlap_ratio is not None and 
            (not np.isnan(overlap_ratio) and overlap_ratio < x_misaligned_overlap_threshold)
            or (int(n_used) < 2)
        )

        # Append pair
        pairs.append({
            "model_label": ms["label"],
            "real_label": rn,
            "compare_mode": kind,
            "points_compared": int(n_used),

            # core metrics
            "mae": mae,
            "nmae": nmae,
            "bias": bias,
            "pearson_r": corr,

            # unscaled metrics
            "ks": ks,
            "integral_l1": integral_l1,

            # NEW SCALED METRICS
            "ks_scaled_by_gt_max": ks_scaled,
            "l1_scaled_by_gt_max": l1_scaled,
            "ks_scaled_by_gt_yrange": ks_scaled_y,
            "l1_scaled_by_gt_yrange": l1_scaled_y,

            # diagnostics
            "x_overlap_ratio": overlap_ratio,
            "x_misaligned": x_misaligned,
            "excluded": excluded,
        })


        used_model.add(ms["label"])
        used_real.add(rn)

    # ============================================================
    # Unmatched series
    # ============================================================
    unmatched_model = [n for n in model_names if n not in used_model]
    unmatched_real = [n for n in real_names if n not in used_real]

    # ============================================================
    # Global aggregates
    # ============================================================
    valid_nmae = [p["nmae"] for p in pairs if p["nmae"] is not None and math.isfinite(p["nmae"])]
    overall_nmae = float(np.mean(valid_nmae)) if valid_nmae else float("nan")
    overall_x_misaligned = any(p["x_misaligned"] for p in pairs)

    valid_ks = [p["ks"] for p in pairs if p["ks"] is not None and math.isfinite(p["ks"])]
    valid_l1 = [p["integral_l1"] for p in pairs if p["integral_l1"] is not None and math.isfinite(p["integral_l1"])]
    valid_ks_scaled = [p["ks_scaled_by_gt_max"] for p in pairs if p["ks_scaled_by_gt_max"] is not None and math.isfinite(p["ks_scaled_by_gt_max"])]
    valid_l1_scaled = [p["l1_scaled_by_gt_max"] for p in pairs if p["l1_scaled_by_gt_max"] is not None and math.isfinite(p["l1_scaled_by_gt_max"])]
    valid_ks_scaled_y = [p["ks_scaled_by_gt_yrange"] for p in pairs if p["ks_scaled_by_gt_yrange"] is not None and math.isfinite(p["ks_scaled_by_gt_yrange"])]
    valid_l1_scaled_y = [p["l1_scaled_by_gt_yrange"] for p in pairs if p["l1_scaled_by_gt_yrange"] is not None and math.isfinite(p["l1_scaled_by_gt_yrange"])]

    overall_ks = float(np.mean(valid_ks)) if valid_ks else float("nan")
    overall_integral_l1 = float(np.mean(valid_l1)) if valid_l1 else float("nan")
    overall_ks_scaled = float(np.mean(valid_ks_scaled)) if valid_ks_scaled else float("nan")
    overall_l1_scaled = float(np.mean(valid_l1_scaled)) if valid_l1_scaled else float("nan")
    overall_ks_scaled_y = float(np.mean(valid_ks_scaled_y)) if valid_ks_scaled_y else float("nan")
    overall_l1_scaled_y = float(np.mean(valid_l1_scaled_y)) if valid_l1_scaled_y else float("nan")

    # ============================================================
    # Final structure
    # ============================================================
    result = {
        # Full per-series comparison results
        "pairs": [
            {
                "model_label": p["model_label"],
                "real_label": p["real_label"],
                "compare_mode": p["compare_mode"],

                # Sample size
                "points_compared": p["points_compared"],

                # Core error metrics
                "mae": p["mae"],
                "nmae": p["nmae"],
                "bias": p["bias"],
                "pearson_r": p["pearson_r"],

                # Distribution / curve-difference metrics
                "ks": p["ks"],
                "integral_l1": p["integral_l1"],

                # Alignment diagnostics
                "x_overlap_ratio": p["x_overlap_ratio"],
                "x_misaligned": p["x_misaligned"],

                # Informational fields about what was included/excluded
                "excluded": p["excluded"],
            }
            for p in pairs
        ],

        # Ground-truth and model metadata
        "unmatched_model_series": unmatched_model,
        "unmatched_real_series": unmatched_real,

        # Per-case aggregate metrics
        "overall_nmae": float(overall_nmae) if overall_nmae is not None else None,
        "overall_ks": float(overall_ks) if overall_ks is not None else None,
        "overall_integral_l1": float(overall_integral_l1) if overall_integral_l1 is not None else None,
        "overall_x_misaligned": bool(overall_x_misaligned),

        # NEW SCALED METRICS
        "ks_scaled_by_gt_max": overall_ks_scaled,
        "l1_scaled_by_gt_max": overall_l1_scaled,
        "ks_scaled_by_gt_yrange": overall_ks_scaled_y,
        "l1_scaled_by_gt_yrange": overall_l1_scaled_y,

        # Full breakdown per series count
        "model_series_count": len(model_series),
        "gt_series_count": len(real_series),

        # Additional diagnostic information
        "notes": (
            "mae = mean absolute y-error on overlapping x-range; "
            "nmae = mae / (ymax - ymin); "
            "bias = mean(model - gt); "
            "pearson_r = correlation of interpolated curves; "
            "ks = max |model(x) - gt(x)|; "
            "integral_l1 = ∫ |model(x) - gt(x)| dx over overlap; "
            "x_overlap_ratio = overlap length / union length; "
            "x_misaligned = True when overlap too small to trust comparison."
        )
    }


    # ============================================================
    # Optional verbose printing
    # ============================================================
    if verbose:
        print("=== Model vs GT Comparison ===")
        for p in pairs:
            print(f"{p['model_label']} ↔ {p['real_label']} [{p['compare_mode']}, n={p['points_compared']}]")
            print(
            "  "
            f"MAE={_fmt(p.get('mae'), '.6g')}  "
            f"NMAE={_fmt(p.get('nmae'), '.6g')}  "
            f"Bias={_fmt(p.get('bias'), '.6g')}  "
            f"r={_fmt(p.get('pearson_r'), '.3g')}"
            )
            print(
                "  "
                f"KS={_fmt(p.get('ks'), '.6g')}  "
                f"Integral-L1={_fmt(p.get('integral_l1'), '.6g')}"
            )
            print(
                "  "
                f"x_overlap={_fmt(p.get('x_overlap_ratio'), '.3g')}  "
                f"x_misaligned={p.get('x_misaligned', 'NA')}"
            )

        if unmatched_model:
            print("Unmatched model series:", unmatched_model)
        if unmatched_real:
            print("Unmatched real series :", unmatched_real)

        print(f"Overall NMAE={_fmt(overall_nmae, '.6g')}")
        print(f"Overall KS={_fmt(overall_ks, '.6g')}")
        print(f"Overall Integral-L1={_fmt(overall_integral_l1, '.6g')}")
        print(f"Overall x_misaligned={overall_x_misaligned}")


    return result
