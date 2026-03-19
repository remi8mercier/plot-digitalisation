# middle_make_crops.py
from typing import List, Dict, Any, Optional
from interval_crops import make_interval_crops
import shutil
import os

import json
import cv2, numpy as np, math, re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import numpy as np
import math
# ======================
# Axis calibration
# ======================

class AxisCal:
    """Linear or log axis calibration: p = a * f(v) + b, where f(v) is identity or log_k."""
    def __init__(self, a: float, b: float, axis: str, mode: str = "linear"):
        self.a, self.b, self.axis, self.mode = float(a), float(b), axis, mode

    def _f(self, v):
        v = np.asarray(v, float)
        if self.mode.startswith("log"):
            log_base = float(self.mode[3:])
            v = np.clip(v, 1e-12, None)
            return np.log(v) / np.log(float(log_base))
        return v

    def v2p(self, v):
        v = self._f(v)
        return self.a * v + self.b

    def p2v(self, p):
        p = np.asarray(p, float)
        fv = (p - self.b) / (self.a if self.a != 0 else 1e-12)
        if self.mode.startswith("log"):
            log_base = float(self.mode[3:])
            return np.power(float(log_base), fv)
        return fv
    
# ======================
# LLM hints
# ======================
@dataclass
class SeriesSpec:
    name: str
    color: Optional[str] = None
    marker: Optional[str] = None
    size_px: Optional[Tuple[int,int]] = None
    line_connected: Optional[bool] = None
    priority: int = 0
    black_type: str = "anydark"   # "pure"|"darkgray"|"anydark"
    color_closeness: float = 0.7  # 0..1, higher = tighter to hint color
    exclude_zones: List[Dict[str,str]] = field(default_factory=list)  # NEW: per-series zones
    is_open: Optional[bool] = None  # True=open, False=closed, None=unknown
@dataclass
class LLMHints:
    crowded: bool = False
    messy: bool = False
    expected_series_count: Optional[int] = 1
    series: List[SeriesSpec] = field(default_factory=list)
    required_confidence: float = 0.6
    max_candidates: int = 200
    x_axis_location: str = "bottom"
    y_axis_location: str = "left"
    default_closeness: float = 0.7

# ======================
# Helpers
# ======================
def _parse_color(s: Optional[str]) -> Optional[Tuple[int,int,int]]:
    """
    Parse a color string into a BGR tuple for OpenCV.
    Was intended to follow KM curves.
    """
    if not s: return None
    s = s.strip().lower()
    NAMED = {
        "black": (0,0,0), "white": (255,255,255), "red": (0,0,255),
        "green": (0,255,0), "blue": (255,0,0), "gray": (128,128,128),
        "grey": (128,128,128), "yellow": (0,255,255), "magenta": (255,0,255),
        "cyan": (255,255,0), "orange": (0,165,255), "purple": (128,0,128),
        "pink": (203,192,255),
    }
    NAMED = {
    # Original
    "black": (0,0,0), "white": (255,255,255), "gray": (128,128,128), "red":(0,   0,   255),
    "green":   (0,   255, 0), "blue":    (255, 0,   0), "grey":    (128, 128, 128),
    "yellow":  (0,   255, 255), "magenta": (255, 0,   255), "cyan":    (255, 255, 0),
    "orange":  (0,   165, 255), "purple":  (128, 0,   128), "pink":    (203, 192, 255),

    # Newly added recommended chart colors
    "brown":   (19,  69,  139),   # saddle-brown-ish, distinct in charts
    "olive":   (34,  139, 34),    # olive-green variant (OpenCV BGR)
    "teal":    (128, 128, 0),     # teal = blue+green, reduced red (BGR)
    "navy":    (128, 0,   0),     # dark blue
    "maroon":  (0,   0,   128),   # dark red
    "gold":    (0,   215, 255),   # close to metallic yellow
}

    if s in NAMED: return NAMED[s]
    if s.startswith("#") and len(s)==7:
        r,g,b = int(s[1:3],16), int(s[3:5],16), int(s[5:7],16)
        return (b,g,r)  # BGR
    m = re.match(r"rgb\((\d+),(\d+),(\d+)\)", s)
    if m:
        r,g,b = map(int,m.groups())
        return (b,g,r)  # BGR
    return None

# ======================
# Axis detection
# ======================
def _detect_axes(
    img: np.ndarray,
    h_angle_deg: float = 3.0,
    v_angle_deg: float = 3.0,
    canny_lo: int = 50,
    canny_hi: int = 140,
    min_line_frac: float = 0.1,
    max_gap_px: int = 12,
):
    """
    Detect x and y axes using spatially constrained Hough transforms.

    Returns:
      (x_axis_y, y_axis_x, x_axis_span, y_axis_span)
    """

    H, W = img.shape[:2]

    # --- Edge detection ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.bilateralFilter(gray, 5, 20, 20)
    edges = cv2.Canny(gray, canny_lo, canny_hi)

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )

    min_len = int(min(H, W) * min_line_frac)

    # -----------------------
    # Helper metrics
    # -----------------------
    def parallel_support(x0, y0, x1, y1, band=2):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, n).astype(int)
        ys = np.linspace(y0, y1, n).astype(int)

        cnt = 0
        for x, y in zip(xs, ys):
            if 0 <= x < W and 0 <= y < H:
                cnt += np.count_nonzero(
                    edges[
                        max(0, y - band):min(H, y + band + 1),
                        max(0, x - band):min(W, x + band + 1)
                    ]
                )
        return cnt / max(1, n)

    def perpendicular_density_vertical(x, y0, y1, band=6):
        tot = 0
        for y in range(y0, y1):
            if 0 <= y < H:
                tot += np.count_nonzero(
                    edges[y, max(0, x - band):min(W, x + band)]
                )
        return tot / max(1, (y1 - y0))

    def perpendicular_density_horizontal(y, x0, x1, band=6):
        tot = 0
        for x in range(x0, x1):
            if 0 <= x < W:
                tot += np.count_nonzero(
                    edges[max(0, y - band):min(H, y + band), x]
                )
        return tot / max(1, (x1 - x0))

    def local_noise(x0, y0, x1, y1, pad=8):
        xa = max(0, min(x0, x1) - pad)
        xb = min(W, max(x0, x1) + pad)
        ya = max(0, min(y0, y1) - pad)
        yb = min(H, max(y0, y1) + pad)
        area = (xb - xa) * (yb - ya)
        if area <= 0:
            return 1.0
        return np.count_nonzero(edges[ya:yb, xa:xb]) / area

    # -------------------------------------------------
    # 1. Y-axis (left ROI only)
    # -------------------------------------------------
    left_limit = int(0.40 * W)
    edges_left = edges.copy()
    edges_left[:, left_limit:] = 0

    v_lines = cv2.HoughLinesP(
        edges_left,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=min_len,
        maxLineGap=max_gap_px,
    )

    y_axis_x = None
    y_axis_span = None
    best_v_score = 0.0

    if v_lines is not None:
        v_eps = math.tan(math.radians(v_angle_deg))

        for x0, y0, x1, y1 in v_lines.reshape(-1, 4):
            dy = abs(y1 - y0)
            dx = abs(x1 - x0)
            if dy == 0 or dx / dy > v_eps:
                continue

            length = math.hypot(x1 - x0, y1 - y0)

            ps = parallel_support(x0, y0, x1, y1)
            pd = perpendicular_density_vertical(
                int(round((x0 + x1) / 2)), min(y0, y1), max(y0, y1)
            )
            noise = local_noise(x0, y0, x1, y1)

            score = length * ps / ((1.0 + pd) * (1.0 + noise))

            if score > best_v_score:
                best_v_score = score
                y_axis_x = int(round((x0 + x1) / 2))
                y_axis_span = (min(y0, y1), max(y0, y1))

    # -------------------------------------------------
    # 2. X-axis (bottom ROI only)
    # -------------------------------------------------
    bottom_limit = int(0.60 * H)
    edges_bottom = edges.copy()
    edges_bottom[:bottom_limit, :] = 0

    h_lines = cv2.HoughLinesP(
        edges_bottom,
        rho=1,
        theta=np.pi / 180,
        threshold=10,
        minLineLength=min_len,
        maxLineGap=max_gap_px,
    )

    x_axis_y = None
    x_axis_span = None
    best_h_score = 0.0

    if h_lines is not None:
        h_eps = math.tan(math.radians(h_angle_deg))

        for x0, y0, x1, y1 in h_lines.reshape(-1, 4):
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            if dx == 0 or dy / dx > h_eps:
                continue

            length = math.hypot(x1 - x0, y1 - y0)

            ps = parallel_support(x0, y0, x1, y1)
            pd = perpendicular_density_horizontal(
                int(round((y0 + y1) / 2)), min(x0, x1), max(x0, x1)
            )
            noise = local_noise(x0, y0, x1, y1)

            score = length * ps / ((1.0 + pd) * (1.0 + noise))

            if score > best_h_score:
                best_h_score = score
                x_axis_y = int(round((y0 + y1) / 2))
                x_axis_span = (min(x0, x1), max(x0, x1))

    # -------------------------------------------------
    # 3. Fallbacks
    # -------------------------------------------------
    if y_axis_x is None:
        y_axis_x = int(0.10 * W)
        y_axis_span = (0, H)

    if x_axis_y is None:
        x_axis_y = int(0.90 * H)
        x_axis_span = (0, W)

    return x_axis_y, y_axis_x, x_axis_span, y_axis_span



# ------------------------------------------------------------
# Adaptive color-cluster helper
# ------------------------------------------------------------
def circular_mean_hue(hues: np.ndarray) -> float:
    """
    Compute mean hue on a circular scale [0,180) in OpenCV HSV.
    """
    # Convert to radians (scaled for OpenCV’s 0–180 → 0–pi)
    angles = np.deg2rad(hues * 2)  # scale back to [0,360)
    sin_sum = np.mean(np.sin(angles))
    cos_sum = np.mean(np.cos(angles))
    mean_angle = np.arctan2(sin_sum, cos_sum)
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    return (np.rad2deg(mean_angle) / 2.0)  # back to 0–180 scale

def _find_closest_color_mean(
    hsv_img: np.ndarray,
    ref_bgr: tuple,
    plot_mask: np.ndarray | None = None,
    s_min: int = 80,
    v_min: int = 60,
    color="unknown"
) -> np.ndarray:
    """
    Finds the dominant color cluster near a reference BGR color (wrap-safe for red).
    Gives weighted mean favoring high-saturation pixels.
    """
    ref_hsv = cv2.cvtColor(np.uint8([[ref_bgr]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
    h_ref, s_ref, v_ref = ref_hsv

    # restrict to plot region if provided
    hsv_flat = hsv_img[plot_mask > 0].reshape(-1, 3) if plot_mask is not None else hsv_img.reshape(-1, 3)

    # keep only colored pixels
    mask_valid = (hsv_flat[:, 1] > s_min) & (hsv_flat[:, 2] > v_min)
    hsv_colored = hsv_flat[mask_valid]
    if len(hsv_colored) == 0:
        return ref_hsv

    # --- circular hue difference ---
    dh = np.abs(hsv_colored[:, 0] - h_ref)
    dh = np.minimum(dh, 180 - dh)  # <---- key fix for red hues
    ds = np.abs(hsv_colored[:, 1] - s_ref)
    dv = np.abs(hsv_colored[:, 2] - v_ref)
    dist = dh + 0.5 * ds / 255.0 + 0.3 * dv / 255.0

    close = hsv_colored[dist < 30]
    if len(close) == 0:
        return ref_hsv

    # weighted mean (saturation as weight)
    weights = (close[:, 1] / 255.0) ** 2
    h_mean = circular_mean_hue(close[:, 0])
    s_mean = np.average(close[:, 1], weights=weights)
    v_mean = np.average(close[:, 2], weights=weights)

    return np.array([h_mean, s_mean, v_mean], np.uint8)

# ======================
# Main detection
# ======================
def empty_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

        
def follow_one_curve_bidirectional(
    mask: np.ndarray,
    x_start: int,
    y_start_px: int,
    x_axis_y: int,
    y_axis_x: int,
    trim_px: int = 2,
    direction: str = "downward",
    guidance_path: Optional[List[Dict[str, float]]] = None,
    x_cal: AxisCal = None,
    y_cal: AxisCal = None,
    **kwargs
):
    """
    Wrapper around follow_one_curve for handling upward/downward plots.
    For upward mode, the image is flipped vertically before tracking.
    Keeps branch_registry + branch_index propagation intact.

    This is the heuristic KM tracker.
    """
    direction = direction.lower()
    upward = direction.startswith("up")

    gx = None
    gy = None

    if guidance_path:
        # value -> pixel
        pts = [(float(x_cal.v2p(p["x"])), float(y_cal.v2p(p["y"]))) for p in guidance_path]
        pts.sort(key=lambda t: t[0])

        step_pts = [pts[0]]
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            if x1 <= x0:
                continue
            step_pts.append((x1, y0))  # horizontal
            step_pts.append((x1, y1))  # vertical

        # right-continuous y(x)
        gx = []
        gy = []
        for x, y in step_pts:
            if not gx or x > gx[-1]:
                gx.append(x); gy.append(y)
            else:
                gy[-1] = y

        gx = np.asarray(gx, float)
        gy = np.asarray(gy, float)

    # extract branching args from kwargs if present
    branch_registry = kwargs.get("branch_registry", None)
    branch_index = kwargs.get("branch_index", 0)
    if not upward:
        # normal downward mode — just call follow_one_curve directly
        return follow_one_curve(
            mask,
            x_start,
            y_start_px,    
            x_axis_y=x_axis_y,
            y_axis_x=y_axis_x,
            branch_registry=branch_registry,
            branch_index=branch_index,
            guidance_path_y=gy,
            guidance_path_x=gx,
            **{k: v for k, v in kwargs.items() if k not in ("branch_registry", "branch_index", "x_axis_y", "y_axis_x")}
        )

    # --- upward mode via vertical flip ---
    H, W = mask.shape

    # Flip vertically
    mask_flipped = cv2.flip(mask, 0)
    gy_flipped = None
    if gx is not None and gy is not None:
        gy_flipped = H - 1 - gy
    else:
        gx = None
        print("gy is flipped")
        
    # ---- Flip previous series into flipped coordinate frame ----
    prev_ser = kwargs.get("previous_serie", None)
    if prev_ser is not None:
        if isinstance(prev_ser, dict):
            xs_prev = np.asarray(prev_ser["x"], dtype=int)
            ys_prev = np.asarray(prev_ser["y"], dtype=int)
        else:
            xs_prev = np.asarray(prev_ser[0], dtype=int)
            ys_prev = np.asarray(prev_ser[1], dtype=int)

        # Flip Y coordinates exactly like the mask
        ys_prev_flipped = H - 1 - ys_prev

        # Repackage in same structure type
        if isinstance(prev_ser, dict):
            prev_ser_flipped = {"x": xs_prev, "y": ys_prev_flipped}
        else:
            prev_ser_flipped = (xs_prev, ys_prev_flipped, prev_ser[2], prev_ser[3])

        # Overwrite argument for follow_one_curve()
        kwargs["previous_serie"] = prev_ser_flipped

    # Optionally trim a horizontal band around the x-axis
    if trim_px > 0:
        # Remove a horizontal strip around the center line after flipping
        center = x_axis_y
        lower = max(center - trim_px, 0)
        upper = min(center + trim_px + 1, H)
        mask_flipped[lower:upper, :] = 0
        
    # Adjust starting y for flipped coordinates
    y_start_flipped = H - 1 - y_start_px

    # Run the same downward tracker on the flipped mask
    xs, ys_flipped, (fx, fy), (lx, ly), m = follow_one_curve(
        mask_flipped,
        x_start,
        y_start_flipped,
        x_axis_y=x_axis_y,
        y_axis_x=y_axis_x,
        branch_registry=branch_registry,
        branch_index=branch_index,
        guidance_path_y=gy_flipped,
        guidance_path_x=gx,
        **{k: v for k, v in kwargs.items() if k not in ("branch_registry", "branch_index", "x_axis_y", "y_axis_x")}
    )

    if xs is None:
        return None, None, (None, None), (None, None), None
        
    if upward:
        m = cv2.flip(m, 0)
    # Convert Y coordinates back to original orientation
    ys = H - 1 - np.array(ys_flipped)
    fy_orig = H - 1 - fy
    ly_orig = H - 1 - ly

    return xs, ys.astype(int), (fx, fy_orig), (lx, ly_orig), m

def convert_decision_segments_llm_to_px(decision_segments, cal_x, W):
    """
    Convert LLM-provided decision segments (real x-axis units) into the
    pixel-based intervals required by follow_one_curve.

    Input:
        decision_segments: list of {"x0":float, "x1":float, "value":int}
        cal_x: AxisCal for x-axis (used to convert real values → pixels)
        W: image width, used for clamping

    Output:
        list of {"x0_px":int, "x1_px":int, "value":int}
    """

    out = []
    if not decision_segments:
        return out

    for seg in decision_segments:
        x0_val = seg["x0"]
        x1_val = seg["x1"]
        val    = seg["value"]

        # Convert real values → pixel positions
        try:
            x0_px = int(round(cal_x.v2p(x0_val)))
            x1_px = int(round(cal_x.v2p(x1_val)))
        except Exception:
            # fallback: ignore segment if cannot convert
            continue

        # clamp to image bounds
        x0_px = max(0, min(W - 1, x0_px))
        x1_px = max(0, min(W - 1, x1_px))

        # ensure ordering
        if x0_px > x1_px:
            x0_px, x1_px = x1_px, x0_px

        out.append({"x0": x0_px, "x1": x1_px, "value": max(val-1, -1)})

    return out

def get_default_decision_segments_for_series(branch_index):
    """
    Provides default decision segments for each series index as a fallback when LLM hints are missing.

    """

    if branch_index == 0:
        a= [
            {"x0": 0.0,   "x1": 12.6, "value": 1},  # highest
            {"x0": 12.6, "x1": 17, "value": 1},  # triangle becomes 2nd
            {"x0": 18, "x1": 20.0, "value": 1},  # triangle highest again
            {"x0": 20, "x1": 300,  "value": 1},
        ]
        return [{'x0': 0, 'x1': 600, 'value': 0}]
    
    if branch_index == 1:
        a= [
            {"x0": 0.0,  "x1": 10.0,  "value": 2},
            {"x0": 10.0, "x1": 30.0,  "value": 2},
            {"x0": 30, "x1": 300,  "value": 2},
        ]
        return [{'x0': 0, 'x1': 557, 'value': -1}]

    if branch_index == 2:
        return [
            {"x0": 0.0,  "x1": 15.0,  "value": 2},
            {"x0": 15.0, "x1": 30.0,  "value": 2},
            {"x0": 30.0, "x1": 45.0,  "value": 2},
            {"x0": 45.0, "x1": 55.0,  "value": 2},
            {"x0": 55.0, "x1": 70.0,  "value": 2},
            {"x0": 70.0, "x1": 82.0,  "value": 1},  # your special case
            {"x0": 82.0, "x1": 95.0,  "value": 0},
            {"x0": 95.0, "x1": 110.0, "value": 0},
            {"x0": 110.0,"x1": 135.0, "value": 0},
        ]

    return []

def track_km_curves(
    image,
    mask: np.ndarray,
    x_cal,
    y_cal,
    y_axis_x: int,
    x_axis_y: int,
    offset_px: int = 5,
    max_series: int = 2,
    start_value: float = 100.0,
    direction: str = "downward",
    decision_segments: list | None = None,
    guidance_paths: list | None = None,
    dy_search: int = 30,
    smooth_med: int = 9,
    lookahead_px: int = 3,
    band_remove: int = 3,
    colors: list = [],
    debug_save_dir: str | None = None,
):
    """
    Track up to `max_series` Kaplan–Meier curves using follow_one_curve_bidirectional.
    Added logic:
      • keeps a shared branch_registry that records cross-head (x, [band_y...])
      • later series reuse registry to start from the same cross-head and
        choose the next lower branch automatically.
    Set `debug_save_dir` to persist the masks used during tracking for debugging.
    """
    
    H, W, _ = image.shape
    save_dir = debug_save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def _save_mask(name: str, data):
        if not save_dir or data is None:
            return
        mask_to_save = data
        if mask_to_save.dtype != np.uint8:
            mask_to_save = mask_to_save.astype(np.uint8)
        if mask_to_save.max() <= 1:
            mask_to_save = mask_to_save * 255
        cv2.imwrite(os.path.join(save_dir, name), mask_to_save)
    
    m = (mask > 0).astype(np.uint8)
    #_save_mask("track_mask_input.png", mask)
    # Save the base mask after axis line removal (if needed)
    # This shows the plot region free of the X-axis baseline.

    axis_band = 4                      # thickness in pixels; adjust 3–6 as needed
    y0 = max(0, x_axis_y - axis_band)
    y1 = min(H, x_axis_y + axis_band + 1)
    m[y0:y1, :] = 0
    
    # normalize direction flag
    d = (direction or "").strip().lower()
    upward = d.startswith("up")
    downward = not upward
    
    # initial X seed (just right of y-axis)
    x_start = int(np.clip(y_axis_x + max(0, offset_px), 0, W - 1))
    # initial Y seed (top or bottom depending on direction)
    start_v = start_value
    try:
        y_seed = int(round(y_cal.v2p(start_v)))
    except Exception:
        y_seed = H - 3 if upward else 3

    y_seed = int(np.clip(y_seed + (-2 if upward else +2), 0, H - 1))

    series_px = []
    branch_registry = []  # shared across series

    for k in range(max_series):
        previous_serie = series_px[k-1] if k > 0 else None

        dec_seg = convert_decision_segments_llm_to_px(decision_segments[k], x_cal, W) if decision_segments else None
        print(f"▶ tracking series {k+1}/{max_series}")
        guidance_for_k = None
        if guidance_paths is not None and k < len(guidance_paths):
            guidance_for_k = guidance_paths[k]

        if colors != [] and colors[k] != None and colors[k] != 'none':
            print(f"  • using color filter: {colors[k]}")
            m = extract_non_background_mask_color(image, background_tol=20, min_saturation=40, debug=False, keep_color=colors[k])
        
        xs_px, ys_px, (fx, fy), (lx, ly), m = follow_one_curve_bidirectional(
            m,
            x_start=x_start,
            y_start_px=y_seed,
            direction=("upward" if upward else "downward"),
            guidance_path=guidance_for_k,
            x_cal=x_cal,
            y_cal=y_cal,
            dy_search=dy_search,
            x_axis_y=x_axis_y,
            y_axis_x=y_axis_x,
            smooth_med=smooth_med,
            lookahead_px=lookahead_px,
            # these two params are new and required by the current follow_one_curve
            branch_index=k,
            branch_registry=branch_registry,
            previous_serie=previous_serie,
            decisions=dec_seg,
        )
        if xs_px is None or ys_px is None:
            print(f"⚠️ Series {k+1} returned empty path. Trying without color")
            m = (mask > 0).astype(np.uint8)
            xs_px, ys_px, (fx, fy), (lx, ly), m = follow_one_curve_bidirectional(
                m,
                x_start=x_start,
                y_start_px=y_seed,
                direction=("upward" if upward else "downward"),
                guidance_path=guidance_for_k,
                dy_search=dy_search,
                x_axis_y=x_axis_y,
                y_axis_x=y_axis_x,
                smooth_med=smooth_med,
                lookahead_px=lookahead_px,
                # these two params are new and required by the current follow_one_curve
                branch_index=k,
                branch_registry=branch_registry,
                previous_serie=previous_serie,
                decisions=dec_seg,
            )
            if xs_px is None or ys_px is None:
                print("It failed again")
                break
        
        series_px.append((xs_px, ys_px, (fx, fy), (lx, ly)))

        # if this run found a new cross-head, propagate for the next
        if branch_registry:
            last_x, bands = branch_registry[-1]
            print(f"  ↳ branch registered at x={last_x}, bands={bands}")

            # after one series finishes and before running the next
        show = False
        if show:
            plt.figure(figsize=(6, 4))
            plt.imshow(m, cmap="gray", origin="upper")
            plt.title(f"Mask after series {k+1}")
            plt.axis("off")
            plt.show()

    if not series_px:
        print("⚠️ No curves detected; check mask or calibration.")
        return None

    # --- merge early shared trunk ---
    merged_series = []
    for i, (xs, ys, firstp, lastp) in enumerate(series_px):
        if i > 0:
            xs_prev, ys_prev, _, _ = series_px[i - 1]
            x_start_new = xs[0]
            keep_prev = xs_prev < x_start_new
            xs = np.concatenate([xs_prev[keep_prev], xs])
            ys = np.concatenate([ys_prev[keep_prev], ys])
            firstp = (xs_prev[0], ys_prev[0])
        merged_series.append((xs, ys, firstp, lastp))

    # --- pixel → value conversion ---
    series_out = []
    for i, (xs, ys, (fx, fy), (lx, ly)) in enumerate(merged_series, 1):
        x_vals = x_cal.p2v(xs)
        y_vals = y_cal.p2v(ys)
        series_out.append({
            "index": i,
            "x": x_vals.tolist(),
            "y": y_vals.tolist(),
            "first_px": {"x": int(fx), "y": int(fy)},
            "last_px":  {"x": int(lx), "y": int(ly)}
        })

    print(f"✅ {len(series_out)} series extracted.")
    return {"series": series_out, "branch_registry": branch_registry}

def pick_segment(mask, segs, x, y_prev,
                 max_jump=4,          # allowed vertical jump per column
                 energy_hw=(3, 2)):   # (half-height, half-width) for energy window
    """
    Choose the best segment based on continuity + mask energy + reasonable vertical movement.

    segs: list of (mid, height, y0, y1)
    mask: binary mask (H, W)
    x:    current column index
    y_prev: previous y

    Returns: (best_mid, best_height, best_y0, best_y1)
    """

    H, W = mask.shape
    hh, hw = energy_hw

    best = None
    best_score = -1e12

    for (mid, h, y0, y1) in segs:

        # 1) Reject impossible vertical jumps
        vjump = abs(mid - y_prev)
        if vjump > max_jump:
            continue

        # 2) Mask continuity in corridor between prev_y and mid
        ymin = max(0, min(mid, y_prev))
        ymax = min(H, max(mid, y_prev))
        x0 = max(0, x-1)
        x1 = min(W, x+2)
        corridor = mask[ymin:ymax+1, x0:x1]
        cont_score = np.count_nonzero(corridor)

        # 3) Local mask energy near the candidate segment
        ey0 = max(0, mid - hh)
        ey1 = min(H, mid + hh + 1)
        ex0 = max(0, x - hw)
        ex1 = min(W, x + hw + 1)
        energy = np.count_nonzero(mask[ey0:ey1, ex0:ex1])

        # 4) Combine score
        # High continuity   → good
        # High energy       → good
        # Smaller vjump     → good
        score = cont_score * 3.0 + energy * 1.2 - vjump * 2.0

        if score > best_score:
            best_score = score
            best = (mid, h, y0, y1)

    # If everything rejected → fallback to original nearest-midpoint
    if best is None:
        segs.sort(key=lambda s: abs(s[0] - y_prev))
        best = segs[0]

    return best

def follow_one_curve(
    mask: np.ndarray,
    x_start: int,
    y_start_px: int,
    dy_search: int = 12,
    smooth_med: int = 9,
    y_axis_x: int = 50,
    x_axis_y: int = 300,
    lookahead_px: int = 3,
    reconnect_search_h: int = 60,
    reconnect_search_w: int = 12,
    min_line_len: int = 10,
    max_line_gap: int = 3,
    branch_index: int = 0,
    branch_registry: list | None = None,
    band_grow_ratio: float = 1.6,
    band_shrink_ratio: float = 1.2,
    min_gap_between_segments: int = 2,
    previous_serie: tuple | dict | None = None,
    separated_gap_px: int = 6,
    sep_hysteresis_on: int = 3,
    sep_hysteresis_off: int = 5,
    decisions: list | None = None,
    guidance_path_x: np.ndarray | None = None,
    guidance_path_y: np.ndarray | None = None,

    # Persistent guidance cost
    guidance_weight: float = 0.8,
    continuity_weight: float = 1.0,

    # Guidance lookahead for multi-band choice
    guidance_lookahead: int = 6,
    guidance_lookahead_stride: int = 2,

    # Periodic correction snaps
    correction_period_px: int = 8,
    correction_extra_dy: int = 10,

    # Hough safety caps
    max_hough_iters: int = 4,
    max_hough_lines_scored: int = 200,
    hough_min_progress_px: int = 1,
):
    """
    follow_one_curve with:
    - Persistent guidance constraint (Option A)
    - Guidance-based band choice at branches (still explicit)
    - Lookahead scoring to pick the band that matches guidance in the next few px
    - Periodic correction snaps
    - Robust guidance normalization into per-x array (O(1) lookup)
    - Hough reconnect iteration caps + progress guard + line scoring cap

    Keeps original features (decisions, inflation/shrink, separated logic, prev_dist, branch_registry,
    forward reconnect, Hough reconnect).
    """
    # -----------------------------
    # Decisions (unchanged)
    # -----------------------------
    if decisions is None:
        decisions = get_default_decision_segments_for_series(branch_index)

    def choose_band(n_bands: int, x: int) -> int:
        if isinstance(decisions, list) and decisions and isinstance(decisions[0], dict):
            chosen_val = None
            for rule in decisions:
                if rule["x0"] <= x <= rule["x1"]:
                    chosen_val = rule["value"]
            if chosen_val is not None:
                return min(int(chosen_val), n_bands - 1)
        return min(int(branch_index), n_bands - 1)

    if branch_registry is None:
        branch_registry = []

    H, W = mask.shape
    m = (mask > 0).astype(np.uint8)

    def clamp_above_axis(y: int) -> int:
        return int(min(y, x_axis_y - 1))

    # -----------------------------
    # Guidance normalization
    # -----------------------------
    # Goal: build y_guid_per_x length W with right-continuous step function
    y_guid_per_x = None

    def _build_guidance_per_x(gx, gy) -> np.ndarray | None:
        """
        Robustly convert guidance (possibly sparse points) into a per-pixel-x reference array.
        Assumes gx,gy are in PIXEL space. If not, caller should convert before passing.
        We enforce:
          - finite
          - sorted by x
          - clipped to [0, W-1]
          - right-continuous hold
        """
        if gx is None or gy is None:
            return None
        gx = np.asarray(gx).astype(float).ravel()
        gy = np.asarray(gy).astype(float).ravel()
        if gx.size < 2 or gy.size < 2:
            return None

        ok = np.isfinite(gx) & np.isfinite(gy)
        gx = gx[ok]
        gy = gy[ok]
        if gx.size < 2:
            return None

        # If gx is not pixel-like, you will see it immediately in debug if needed:
        # e.g. gx min/max far outside [0,W].
        # We clip anyway, but if it's value-space, it will collapse to boundaries.
        gx = np.clip(gx, 0, W - 1)

        order = np.argsort(gx)
        gx = gx[order]
        gy = gy[order]

        # make x strictly nondecreasing; if duplicates, keep last (right-continuous)
        # We'll compress duplicates.
        x_int = np.round(gx).astype(int)
        y_val = gy.copy()

        # compress by last occurrence
        uniq_x = []
        uniq_y = []
        lastx = None
        for xi, yi in zip(x_int, y_val):
            if lastx is None or xi != lastx:
                uniq_x.append(int(xi))
                uniq_y.append(float(yi))
                lastx = int(xi)
            else:
                uniq_y[-1] = float(yi)

        if len(uniq_x) < 2:
            return None

        uniq_x = np.asarray(uniq_x, dtype=int)
        uniq_y = np.asarray(uniq_y, dtype=float)

        # Build right-continuous array
        yref = np.empty(W, dtype=float)
        yref[:] = uniq_y[0]
        j = 0
        cur = float(uniq_y[0])
        for x in range(W):
            while j + 1 < len(uniq_x) and x >= uniq_x[j + 1]:
                j += 1
                cur = float(uniq_y[j])
            yref[x] = cur
        return yref

    if guidance_path_x is not None and guidance_path_y is not None:
        y_guid_per_x = _build_guidance_per_x(guidance_path_x, guidance_path_y)

    def guidance_y_at_x(x: int) -> float | None:
        if y_guid_per_x is None:
            return None
        x = int(np.clip(x, 0, W - 1))
        return float(y_guid_per_x[x])

    # -----------------------------
    # Persistent cost
    # -----------------------------
    def combined_cost(x: int, y_cand: int, y_prev_local: int) -> float:
        c = continuity_weight * abs(int(y_cand) - int(y_prev_local))
        y_ref = guidance_y_at_x(x)
        if y_ref is not None:
            c += guidance_weight * abs(int(y_cand) - y_ref)
        return float(c)

    # -----------------------------
    # Segment extraction (unchanged)
    # -----------------------------
    def _extract_segments(col, y_low):
        ys = np.where(col > 0)[0]
        if ys.size == 0:
            return []
        gaps = np.where(np.diff(ys) >= min_gap_between_segments)[0]
        segs = []
        s = 0
        for g in np.r_[gaps, len(ys) - 1]:
            a, b = ys[s], ys[g]
            y0, y1 = y_low + a, y_low + b
            segs.append((int((y0 + y1) / 2), y1 - y0 + 1, y0, y1))
            s = g + 1
        return segs

    # -----------------------------
    # prev series distance transform (unchanged)
    # -----------------------------
    prev_dist = None
    if previous_serie is not None:
        if isinstance(previous_serie, dict):
            xs_prev = np.asarray(previous_serie["x"], int)
            ys_prev = np.asarray(previous_serie["y"], int)
        else:
            xs_prev = np.asarray(previous_serie[0], int)
            ys_prev = np.asarray(previous_serie[1], int)

        xs_prev = np.clip(xs_prev, 0, W - 1)
        ys_prev = np.clip(ys_prev, 0, H - 1)

        prev_mask = np.zeros_like(m, dtype=np.uint8)
        for i in range(1, len(xs_prev)):
            cv2.line(prev_mask, (xs_prev[i - 1], ys_prev[i - 1]),
                     (xs_prev[i], ys_prev[i]), 255, 3)

        if prev_mask.max() <= 1:
            prev_mask = prev_mask * 255

        inv = cv2.bitwise_not(prev_mask)
        prev_dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    separated = False
    sep_on_cnt = 0
    sep_off_cnt = 0

    def _update_separated_flag(x_cur, y_cur, gap_thresh=6):
        """
        Update the separated flag based on the current pixel.
        The separated flag is used to determine if the branch we are following is separated from other branches or not.
        """
        nonlocal separated, sep_on_cnt, sep_off_cnt
        if prev_dist is None:
            return
        if not (0 <= x_cur < W and 0 <= y_cur < H):
            return
        d = float(prev_dist[y_cur, x_cur])
        is_sep = (d >= gap_thresh)

        if is_sep:
            sep_on_cnt += 1
            sep_off_cnt = 0
        else:
            sep_off_cnt += 1
            sep_on_cnt = 0

        if (not separated) and sep_on_cnt >= sep_hysteresis_on:
            separated = True
        if separated and sep_off_cnt >= sep_hysteresis_off:
            separated = False

    # -----------------------------
    # Lookahead score for band choice (kept)
    # -----------------------------
    def score_band_by_guidance_lookahead(x0: int, y0: int) -> float:
        """
        Score how well a candidate band at (x0, y0) matches the guidance in the next few pixels.
        """
        y_ref0 = guidance_y_at_x(x0)
        if y_ref0 is None:
            return 0.0
        total = abs(y0 - y_ref0)
        for dx in range(guidance_lookahead_stride, guidance_lookahead + 1, guidance_lookahead_stride):
            xx = x0 + dx
            if xx >= W:
                break
            y_ref = guidance_y_at_x(xx)
            if y_ref is None:
                continue
            total += abs(y0 - y_ref)
        return float(total)

    def choose_band_with_guidance(x: int, bands: list[int], y_prev_local: int) -> int:
        """
        Choose which band to follow at a branch point, using both current cost and lookahead guidance.
        """
        if y_guid_per_x is None:
            return choose_band(len(bands), x)
        costs_now = [combined_cost(x, b, y_prev_local) for b in bands]
        costs_future = [score_band_by_guidance_lookahead(x, b) for b in bands]
        future_w = 0.35
        blended = [cn + future_w * cf for cn, cf in zip(costs_now, costs_future)]
        return int(np.argmin(blended))

    # -----------------------------
    # Main tracking state
    # -----------------------------
    y_track = np.full(W, np.nan)
    y_prev = int(np.clip(y_start_px, 0, H - 1))
    first_x = first_y = last_x = last_y = None
    recent_heights = []
    inflated = False

    half = dy_search // 2
    last_correction_x = -10**9

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    for x in range(max(0, x_start), W):
        found = False

        # Local search with ahead (structure preserved)
        for ahead in range(0, lookahead_px + 1):
            xx = x + ahead
            if xx >= W:
                break

            y_low = max(0, y_prev - half // 4)
            y_high = min(H, y_prev + half)

            col = m[y_low:y_high, xx]
            segs = _extract_segments(col, y_low)
            if not segs:
                continue

            # Choose main segment via persistent combined cost (Option A)
            seg_centers = [s[0] for s in segs]
            best_idx = int(np.argmin([combined_cost(xx, yc, y_prev) for yc in seg_centers]))
            main_y = int(seg_centers[best_idx])
            h = int(segs[best_idx][1])

            y_track[x] = main_y
            if first_x is None:
                first_x, first_y = x, main_y
            last_x, last_y = x, main_y

            y_prev = clamp_above_axis(int(main_y))
            _update_separated_flag(x, y_prev)

            recent_heights.append(h)
            if len(recent_heights) > 30:
                recent_heights.pop(0)
            baseline = np.median(recent_heights[:-1]) if len(recent_heights) > 5 else h

            if baseline > 0 and h / baseline >= band_grow_ratio and not inflated:
                inflated = True
                inflated_h = h

            do_correction = (y_guid_per_x is not None) and ((x - last_correction_x) >= correction_period_px)

            if inflated or (not separated) or do_correction:
                # Correction can look a bit wider
                if do_correction:
                    y_low2 = max(0, y_prev - (half // 4 + correction_extra_dy))
                    y_high2 = min(H, y_prev + (half + correction_extra_dy))
                    col2 = m[y_low2:y_high2, xx]
                    segs_now = _extract_segments(col2, y_low2)
                else:
                    segs_now = _extract_segments(col, y_low)

                if len(segs_now) >= 2:
                    bands = [clamp_above_axis(int(s[0])) for s in segs_now]
                    branch_registry.append((x, bands))

                    chosen = choose_band_with_guidance(x, bands, y_prev)
                    y_prev = clamp_above_axis(int(bands[chosen]))
                    y_track[x] = y_prev
                    last_x, last_y = x, y_prev

                    if do_correction:
                        last_correction_x = x

            if inflated and h <= inflated_h / band_shrink_ratio:
                inflated = False
                segs_now = _extract_segments(col, y_low)
                if len(segs_now) >= 2:
                    bands = [clamp_above_axis(int(s[0])) for s in segs_now]
                    branch_registry.append((x, bands))

                    chosen = choose_band_with_guidance(x, bands, y_prev)
                    y_prev = clamp_above_axis(int(bands[chosen]))
                    y_track[x] = y_prev
                    last_x, last_y = x, y_prev

            found = True
            break

        # ---------------- forward reconnect (guidance-aware) ----------------
        if not found:
            if first_x is not None and not np.any(m[max(0, y_prev - 2):min(H, y_prev + 3), x:x + 3]):
                max_ahead = 15
                for ahead in range(4, max_ahead):
                    xx = x + ahead
                    if xx >= W:
                        break
                    half2 = dy_search // 2
                    y_low = max(0, y_prev - half2 // 2)
                    y_high = min(H, y_prev + half2)

                    col = m[y_low:y_high, xx]
                    if np.any(col):
                        ys = np.where(col > 0)[0]
                        cand_ys = (y_low + ys).astype(int)

                        costs = [combined_cost(xx, yc, y_prev) for yc in cand_ys]
                        y_cur = int(cand_ys[int(np.argmin(costs))])

                        y_track[x:xx + 1] = np.linspace(y_prev, y_cur, xx - x + 1)
                        y_prev = clamp_above_axis(int(y_cur))
                        last_x, last_y = xx, y_prev
                        _update_separated_flag(xx, y_prev)
                        found = True
                        break
                if found:
                    continue

            # ---------------- Hough reconnect (bounded + cached guidance) ----------------
            if first_x is None:
                continue

            iters = 0
            prev_best_drop = None

            while True:
                iters += 1
                if iters > max_hough_iters:
                    break

                x0 = x
                y0 = int(min(y_prev + 3, H - 1))
                x1 = min(x + reconnect_search_w, W)
                y1 = min(y0 + reconnect_search_h, H)

                roi = m[y0:y1, x0:x1]
                if roi.size == 0:
                    break

                lines = cv2.HoughLinesP(
                    roi, 1, np.pi/180, 8,
                    minLineLength=min_line_len,
                    maxLineGap=max_line_gap
                )
                if lines is None:
                    break

                lines = lines.reshape(-1, 4)

                # Cap how many lines we score (NEW)
                if lines.shape[0] > max_hough_lines_scored:
                    # keep the longest/most vertical-ish subset cheaply
                    dx = np.abs(lines[:, 2] - lines[:, 0])
                    dy = np.abs(lines[:, 3] - lines[:, 1])
                    rough = dy - 2 * dx
                    idx = np.argsort(-rough)[:max_hough_lines_scored]
                    lines = lines[idx]

                def score_line(L):
                    dx = abs(L[2] - L[0])
                    dy = abs(L[3] - L[1])
                    base = (dy - dx * 2)

                    if y_guid_per_x is not None:
                        x_mid = x0 + int((L[0] + L[2]) / 2)
                        y_mid = y0 + int((L[1] + L[3]) / 2)
                        y_ref = guidance_y_at_x(x_mid)
                        if y_ref is not None:
                            # bounded influence (NEW)
                            base -= 0.5 * min(abs(y_mid - y_ref), 15)

                    return (base, dy)

                xA, yA, xB, yB = max(lines, key=score_line)

                stripe_x = slice(x0 - max(xA, xB) // 3, x0 + max(xA, xB))
                stripe_y = slice(y_prev, min(H, y_prev + max(yA, yB)))
                stripe = m[stripe_y, stripe_x]
                if stripe.size == 0:
                    break

                row_sums = np.sum(stripe > 0, axis=1)
                if np.count_nonzero(row_sums > 0) / len(row_sums) < 0.25:
                    break

                drop_y = int(y0 + max(yA, yB))

                if y_guid_per_x is not None:
                    x_drop = x0 + int(max(xA, xB))
                    y_ref = guidance_y_at_x(x_drop)
                    if y_ref is not None:
                        cand = [drop_y, int(np.clip(round(y_ref), 0, H - 1))]
                        drop_y = int(min(cand, key=lambda yc: combined_cost(x_drop, yc, y_prev)))

                # progress guard (NEW): if we keep landing on same drop, stop
                if prev_best_drop is not None and abs(drop_y - prev_best_drop) < hough_min_progress_px:
                    break
                prev_best_drop = drop_y

                y_prev = int(np.clip(drop_y, 0, H - 1))
                y_prev = clamp_above_axis(y_prev)
                y_track[x] = y_prev
                last_x, last_y = x, y_prev

                roi_next = m[min(y_prev + 1, H - 1):min(y_prev + reconnect_search_h, H), x0:x1]
                if not np.any(roi_next):
                    break

    # ---------------- finalize (unchanged) ----------------
    if first_x is None or last_x is None:
        return None, None, (None, None), (None, None), None

    xs = np.arange(first_x, last_x + 1)
    ys = y_track[first_x:last_x + 1]
    good = np.isfinite(ys)
    if not np.any(good):
        return None, None, (None, None), (None, None), None

    ys = np.interp(xs, xs[good], ys[good])
    ys = median_filter(ys, size=smooth_med)

    return xs, np.round(ys).astype(int), (first_x, first_y), (last_x, last_y), mask

def extract_non_background_mask(
    bgr_img,
    use_dark_detection=False,
    use_adaptive=False,
    use_wideband_suppression=False,
    use_curve_reinforce=False,
    use_highpass_detail=False,
    background_tol=20,
    min_saturation=40,
    debug=False
):
    """
    Hybrid version: behaves like the OG function by default (HSV-based background detection),
    but also allows optional enhancements toggled by booleans.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_final = np.zeros_like(gray, dtype=np.uint8)

    # OG background detection (always baseline)
    corners = np.concatenate([
        v[:40, :40].ravel(), v[:40, -40:].ravel(),
        v[-40:, :40].ravel(), v[-40:, -40:].ravel()
    ])
    bg_val = np.median(corners)
    bg_mask = (v > bg_val - background_tol) & (v < bg_val + background_tol) & (s < min_saturation)
    bg_mask = (bg_mask.astype(np.uint8) * 255)
    mask_base = cv2.bitwise_not(bg_mask)
    mask_final = cv2.bitwise_or(mask_final, mask_base)

    # (1) Dark detection
    if use_dark_detection:
        _, mask_dark = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        mask_dark = cv2.medianBlur(mask_dark, 3)
        mask_final = cv2.bitwise_or(mask_final, mask_dark)

    # (2) Adaptive threshold
    if use_adaptive:
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 35, 5
        )
        mask_final = cv2.bitwise_or(mask_final, adaptive)

    # (3) Suppress broad bands
    if use_wideband_suppression:
        kernel = np.ones((1, 50), np.uint8)
        wide_vert = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)
        mask_final = cv2.subtract(mask_final, wide_vert)

    # (4) Reinforce continuity
    if use_curve_reinforce:
        mask_final = cv2.dilate(mask_final, np.ones((2,2), np.uint8), iterations=1)

    # (5) High-pass detail
    if use_highpass_detail:
        blurred = cv2.GaussianBlur(gray, (9,9), 0)
        highfreq = cv2.subtract(gray, blurred)
        _, mask_high = cv2.threshold(highfreq, 10, 255, cv2.THRESH_BINARY_INV)
        mask_final = cv2.bitwise_or(mask_final, mask_high)

    # Final cleanup
    mask_final = cv2.medianBlur(mask_final, 3)

    if debug:
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.title("Pure Binary Mask")
        plt.imshow(mask_final, cmap='gray', vmin=0, vmax=255)
        plt.axis("off")
        plt.show()

    return mask_final

def extract_non_background_mask_color(
    bgr_img,
    use_dark_detection=False,
    use_adaptive=False,
    use_wideband_suppression=False,
    use_curve_reinforce=False,
    use_highpass_detail=False,
    background_tol=40,
    min_saturation=40,
    debug=False,
    #
    # Color editing
    #
    remove_color: str | None = None,
    remove_color_closeness: float = 0.6,
    keep_color: str | None = None,            # NEW
    keep_color_closeness: float = 0.6         # NEW
):
    """
    Extended version: behaves like before but supports:
      - remove_color: remove this color from the mask
      - keep_color: keep ONLY this color and suppress everything else

    keep_color takes priority if both are given.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_final = np.zeros_like(gray, dtype=np.uint8)

    # =====================================================
    # BASELINE: original hybrid non-background detection
    # =====================================================
    corners = np.concatenate([
        v[:40, :40].ravel(), v[:40, -40:].ravel(),
        v[-40:, :40].ravel(), v[-40:, -40:].ravel()
    ])
    bg_val = np.median(corners)
    bg_mask = (v > bg_val - background_tol) & (v < bg_val + background_tol) & (s < min_saturation)
    bg_mask = (bg_mask.astype(np.uint8) * 255)
    mask_base = cv2.bitwise_not(bg_mask)
    mask_final = mask_base.copy()

    # Optional enhancements
    if use_dark_detection:
        _, mask_dark = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        mask_dark = cv2.medianBlur(mask_dark, 3)
        mask_final = cv2.bitwise_or(mask_final, mask_dark)

    if use_adaptive:
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 35, 5
        )
        mask_final = cv2.bitwise_or(mask_final, adaptive)

    if use_wideband_suppression:
        kernel = np.ones((1, 50), np.uint8)
        wide_vert = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)
        mask_final = cv2.subtract(mask_final, wide_vert)

    if use_curve_reinforce:
        mask_final = cv2.dilate(mask_final, np.ones((2,2), np.uint8), iterations=1)

    if use_highpass_detail:
        blurred = cv2.GaussianBlur(gray, (9,9), 0)
        highfreq = cv2.subtract(gray, blurred)
        _, mask_high = cv2.threshold(highfreq, 10, 255, cv2.THRESH_BINARY_INV)
        mask_final = cv2.bitwise_or(mask_final, mask_high)

    # ============================================================
    # INTERNAL: small helper to compute a color mask via your HSV logic
    # ============================================================
    def _make_color_mask(color_name: str, closeness: float):
        ref_bgr = _parse_color(color_name)
        if ref_bgr is None:
            return None

        refined_hsv = _find_closest_color_mean(
            hsv_img=hsv,
            ref_bgr=ref_bgr,
            plot_mask=None,
            color=color_name.lower()
        )

        closeness = float(np.clip(closeness, 0.0, 1.0))
        tol_h  = int(35 - 25 * closeness)
        band_s = int(80 - 55 * closeness)
        band_v = int(110 - 70 * closeness)

        h0, s0, v0 = refined_hsv.astype(int)

        lo = np.array([max(0, h0 - tol_h), max(0, s0 - band_s), max(0, v0 - band_v)], np.uint8)
        hi = np.array([min(179, h0 + tol_h), min(255, s0 + band_s), min(255, v0 + band_v)], np.uint8)

        # Hue wrap around
        if h0 < tol_h or h0 > 179 - tol_h:
            lo1 = np.array([0, max(0, s0 - band_s), max(0, v0 - band_v)], np.uint8)
            hi1 = np.array([h0 + tol_h, min(255, s0 + band_s), min(255, v0 + band_v)], np.uint8)
            lo2 = np.array([max(0, 180 - tol_h), max(0, s0 - band_s), max(0, v0 - band_v)], np.uint8)
            hi2 = np.array([179, min(255, s0 + band_s), min(255, v0 + band_v)], np.uint8)
            return cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)

        return cv2.inRange(hsv, lo, hi)

    # ============================================================
    # KEEP COLOR (priority)
    # ============================================================
    if keep_color:
        c_mask = _make_color_mask(keep_color, keep_color_closeness)
        if c_mask is None:
            # If color unknown → empty mask
            mask_final = np.zeros_like(mask_final)
        else:
            # Keep ONLY this color
            mask_final = cv2.bitwise_and(mask_final, c_mask)

        mask_final = cv2.medianBlur(mask_final, 3)

        if debug:
            plt.figure(figsize=(12,5))
            plt.subplot(1,2,1)
            plt.title("Original")
            plt.imshow(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.subplot(1,2,2)
            plt.title(f"Mask (keeping only {keep_color})")
            plt.imshow(mask_final, cmap='gray', vmin=0, vmax=255)
            plt.axis("off")
            plt.show()

        return mask_final

    # ============================================================
    # REMOVE COLOR (only if keep_color is not used)
    # ============================================================
    if remove_color:
        rm = _make_color_mask(remove_color, remove_color_closeness)
        if rm is not None:
            mask_final = cv2.bitwise_and(mask_final, cv2.bitwise_not(rm))

    # final smoothing
    mask_final = cv2.medianBlur(mask_final, 3)

    if debug:
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.subplot(1,2,2)
        ttl = f"Mask (removing {remove_color})" if remove_color else "Pure Binary Mask"
        plt.title(ttl)
        plt.imshow(mask_final, cmap='gray', vmin=0, vmax=255)
        plt.axis("off")
        plt.show()

    return mask_final


if __name__ == "__main__":
    # --- 1. Load the mask ---
    e_graph1 = "graphs/e_graph1.png"
    webp3 = "graphs/3.webp"
    image_used = e_graph1
    img = cv2.imread(image_used)

    # Get the non-background mask
    mask1 = extract_non_background_mask(img, background_tol=20, min_saturation=40, debug=True)
    #mask1= extract_non_background_mask_color(cv2.imread(webp3), background_tol=20, min_saturation=40, debug=True, keep_color="blue")

    # for g4 y pixel start : y_prev = 61
    # --- 3. (Mandatory) use your existing calibrated AxisCal objects ---
    # (Example placeholders; replace with your own calibration values)
    cal_x = AxisCal(a=27.2674, b=85.7674, axis="x", mode="linear")
    cal_y = AxisCal(a=-2.95, b=321.8, axis="y", mode="linear")

    cal_x = AxisCal(a=110.5, b=147.667, axis="x", mode="linear")
    cal_y = AxisCal(a=-4.5, b=342, axis="y", mode="linear")
    colors = ["orange", "blue"]
    # --- 4. Track both Kaplan–Meier series ---
    x_axis_y, y_axis_x, x_span, y_span = _detect_axes(img)

    results = track_km_curves(
        img,
        mask1,
        cal_x,
        cal_y,
        max_series=2,
        y_axis_x=y_axis_x,
        x_axis_y=x_axis_y,
        offset_px=5,      # start 5 px to the right of the y-axis
        start_value=100, # begin from the calibrated top
        #direction="downward"
        #direction="upward",
        #colors=colors
    )
    for s in results["series"]:
        print(f"Detected series {s['index']} with {len(s['x'])} points.")
