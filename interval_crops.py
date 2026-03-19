# interval_crops.py — Axis-detect + OCR (whole image) + LLM-gated calibration
# Then draw dense ticks: base step = scale, labeled ticks every k*scale (anti-overlap),
# unlabeled minor ticks at each scale multiple, labels printed 2 orders finer than scale.
#
# Public API preserved: AxisCal, CropBox, make_interval_crops(...)

from __future__ import annotations
import os, json, math, re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import psutil, os, gc
import numpy as np
import cv2
import ocr  
import gc, ctypes, cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
import os, json, math, base64
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import sklearn
import numpy as np
from itertools import combinations
# =============================
# Data classes
# =============================
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

@dataclass
class CropBox:
    """x0, y0, x1, y1 define a pixel crop box. Coordinates will be clamped to image size."""
    x0: int; y0: int; x1: int; y1: int
    def clamp(self, W: int, H: int):
        self.x0 = max(0, min(self.x0, W-1))
        self.y0 = max(0, min(self.y0, H-1))
        self.x1 = max(self.x0+1, min(self.x1, W))
        self.y1 = max(self.y0+1, min(self.y1, H))
        return self


# =============================
# Utils
# =============================
_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?$", re.I)

def _is_numeric_str(s: str) -> bool:
    """True if s looks like a number (int or float), after stripping and replacing common dash variants."""
    s = str(s).strip().replace("–","-").replace("—","-")
    return bool(_NUM_RE.match(s))

def _tok_text(t: Dict[str, Any]) -> str:
    """
    Extract the text from a token dictionary.

    If the token contains a "text" key with a string value, return that string.
    If the token contains a "value" key, return a string representation of that value.
    Otherwise, return an empty string.
    """
    if "text" in t and isinstance(t["text"], str):
        return t["text"]
    if "value" in t:
        return str(t["value"])
    return ""

def _to_fit_value(v: float, mode: str, base: float = 10.0) -> Optional[float]:
    """
    Convert display-domain value to fit-domain value.

    mode:
      - "linear"
      - "log"  (generic log-k, base provided)
    """
    try:
        v = float(v)
    except Exception:
        return None

    if mode == "linear":
        return v

    if mode.startswith("log"):
        if v <= 0 or base <= 0 or base == 1:
            return None
        return math.log(v, base)

    raise ValueError(f"Unknown axis mode: {mode}")

def _tok_pixel_center(t: Dict[str, Any]) -> Optional[Tuple[float,float]]:
    """
    Extract the pixel center from a token dictionary.

    If the token contains a "pixel" key with a length-2 list/tuple value, return that pixel as a (float, float) tuple.
    If the token contains a "bbox"/"box"/"rect"/"quad" key with a length-4 list/tuple value, return the center of that box as a (float, float) tuple.
    Otherwise, return None.
    """
    p = t.get("pixel")
    if isinstance(p, (list, tuple)) and len(p) == 2:
        try: return float(p[0]), float(p[1])
        except Exception: return None
    B = t.get("bbox") or t.get("box") or t.get("rect") or t.get("quad")
    if isinstance(B, (list, tuple)) and len(B) == 4:
        x0,y0,x1,y1 = map(float, B)
        return (0.5*(x0+x1), 0.5*(y0+y1))
    return None

# We use the Tick model from tick_model.ipynb
class TickCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        
def extract_tick_crop(
    gray: np.ndarray,
    cx: int,
    cy: int,
    size: int = 32
) -> np.ndarray:
    """
    Extract a centered, padded 32×32 crop.
    Assumes gray is float32 in [0,1], white background.
    """
    H, W = gray.shape
    half = size // 2

    crop = np.ones((size, size), dtype=np.float32)

    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half

    sx0, sx1 = max(0, x0), min(W, x1)
    sy0, sy1 = max(0, y0), min(H, y1)

    dx0 = sx0 - x0
    dy0 = sy0 - y0

    crop[
        dy0 : dy0 + (sy1 - sy0),
        dx0 : dx0 + (sx1 - sx0),
    ] = gray[sy0:sy1, sx0:sx1]

    return crop

def _coerce_float_list(x) -> List[float]:
    """
    Coerce a list of values to a list of floats.

    If the input is not a list, an empty list is used.
    If a value in the list cannot be converted to a float, it is skipped.
    """
    out = []
    for v in (x or []):
        try: out.append(float(v))
        except Exception: pass
    return out

# =============================
# Axis detection via edges/Hough (no bbox needed)
# =============================
def _detect_axes(img: np.ndarray,
                 h_angle_deg: float = 3.0,
                 v_angle_deg: float = 3.0,
                 canny_lo: int = 50,
                 canny_hi: int = 140,
                 min_line_frac: float = 0.30,
                 max_gap_px: int = 10
                 ) -> Tuple[int, int, Tuple[float,float], Tuple[float,float]]:
    """
    Return (x_axis_y, y_axis_x, x_axis_span, y_axis_span).

    - x_axis_y: vertical pixel coordinate of the x-axis baseline.
    - y_axis_x: horizontal pixel coordinate of the y-axis baseline.
    - x_axis_span: (x_start, x_end) in pixels along the x-axis line.
    - y_axis_span: (y_start, y_end) in pixels along the y-axis line.

    Falls back to heuristics if not found.
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.bilateralFilter(gray, 5, 20, 20)
    edges = cv2.Canny(gray, canny_lo, canny_hi)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)

    min_len = int(min(H, W) * min_line_frac)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                            minLineLength=min_len, maxLineGap=max_gap_px)

    x_axis_y = None
    y_axis_x = None
    x_axis_span = None
    y_axis_span = None

    if lines is not None:
        lines = lines.reshape(-1,4)
        h_eps = math.tan(math.radians(h_angle_deg))
        v_eps = math.tan(math.radians(v_angle_deg))

        best_h = (0.0, None, None)  # (score, y, (x0,x1))
        best_v = (0.0, None, None)  # (score, x, (y0,y1))

        for x0,y0,x1,y1 in lines:
            dx, dy = x1-x0, y1-y0
            if dx != 0:
                slope_h = abs(dy/dx)
                if slope_h <= h_eps:  # horizontal candidate
                    length = math.hypot(dx, dy)
                    bottom_weight = 1.0 + (y0 + y1) / (2*H)
                    score = length * bottom_weight
                    if score > best_h[0]:
                        best_h = (score,
                                  int(round((y0+y1)/2)),
                                  (min(x0,x1), max(x0,x1)))
            if dy != 0:
                slope_v = abs((x1-x0)/dy)
                if slope_v <= v_eps:  # vertical candidate
                    length = math.hypot(dx, dy)
                    left_weight = 2.0 - (x0 + x1) / (2*W)
                    score = length * left_weight
                    if score > best_v[0]:
                        best_v = (score,
                                  int(round((x0+x1)/2)),
                                  (min(y0,y1), max(y0,y1)))

        if best_h[1] is not None:
            x_axis_y = best_h[1]
            x_axis_span = best_h[2]
        if best_v[1] is not None:
            y_axis_x = best_v[1]
            y_axis_span = best_v[2]

    # Fallbacks
    if x_axis_y is None:
        x_axis_y = int(0.90 * H)
        x_axis_span = (0, W)
    if y_axis_x is None:
        y_axis_x = int(0.10 * W)
        y_axis_span = (0, H)

    return x_axis_y, y_axis_x, x_axis_span, y_axis_span

# =============================
# OCR gating + LLM intersection
# =============================
def _snap_tick_to_axis(
    img: np.ndarray,
    token: dict,
    axis: str,
    x_axis_y: int,
    y_axis_x: int,
    search_frac: float = 0.02,
    dist_bias_weight: float = 0.3,
    d_flat: float = 6.0,
    d_scale: float = 3.0,
    dist_power: float = 2.0,
    model: nn.Module = None,
    device: torch.device = torch.device("cpu"),
    min_confidence: float = 0.5,   # <-- NEW
) -> dict | None:
    """
    Snap a token to a nearby axis line.

    - `img`: Grayscale image of the entire page.
    - `token`: Dictionary containing the token to be snapped.
    - `axis`: String indicating the axis to snap the token to. One of "x" or "y".
    - `x_axis_y`: Integer indicating the vertical pixel coordinate of the x-axis baseline.
    - `y_axis_x`: Integer indicating the horizontal pixel coordinate of the y-axis baseline.
    - `search_frac`: Float indicating the fraction of the image size to search for a better token position.
    - `dist_bias_weight`: Float indicating the weight of the distance bias term.
    - `d_flat`: Float indicating the distance (in pixels) at which the distance bias term is 1.0.
    - `d_scale`: Float indicating the scale of the distance bias term.
    - `dist_power`: Float indicating the power of the distance bias term.
    - `model`: neural network model to use for confidence estimation.

    Returns a dictionary containing the snapped token and its confidence if the confidence is above `min_confidence`. Otherwise, returns the center of the bounding box.
    """
    # ------------------------------------------------------------
    # Validate token
    # ------------------------------------------------------------
    p = token.get("pixel")
    if not (isinstance(p, (list, tuple)) and len(p) == 2):
        return None

    cx, cy = map(int, p)
    H, W = img.shape[:2]

    # ------------------------------------------------------------
    # Grayscale + normalize (MATCH TRAINING)
    # ------------------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # ------------------------------------------------------------
    # Distance bias
    # ------------------------------------------------------------
    def dist_bias(dist):
        if dist <= d_flat:
            return 1.0
        x = (dist - d_flat) / d_scale
        return 1.0 / (1.0 + x ** dist_power)

    def _extract_tick_crop(gray: np.ndarray, cx: int, cy: int, size: int = 32):
        H, W = gray.shape
        half = size // 2

        crop = np.ones((size, size), dtype=np.float32)

        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half

        sx0, sx1 = max(0, x0), min(W, x1)
        sy0, sy1 = max(0, y0), min(H, y1)

        dx0 = sx0 - x0
        dy0 = sy0 - y0

        crop[
            dy0 : dy0 + (sy1 - sy0),
            dx0 : dx0 + (sx1 - sx0),
        ] = gray[sy0:sy1, sx0:sx1]

        return crop

    @torch.no_grad()
    def _model_confidence(crop: np.ndarray):
        """
        Returns the confidence of the token in the given crop using the model.
        Confidence is between 0.0 and 1.0.
        """
        x = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = x.to(device)
        logits = model(x)
        return torch.softmax(logits, dim=1)[0, 1].item()

    # ============================================================
    # Y-axis snapping
    # ============================================================
    if axis == "y":
        axis_x = int(y_axis_x)
        win = max(1, int(H * search_frac))
        y0, y1 = max(0, cy - win), min(H - 1, cy + win)

        best_y = cy
        best_score = -1e9
        best_conf = 0.0

        for y in range(y0, y1 + 1):
            crop = _extract_tick_crop(gray, axis_x, y)
            conf = _model_confidence(crop)

            dist = abs(y - cy)
            score = conf + dist_bias_weight * dist_bias(dist)

            if score > best_score:
                best_score = score
                best_y = y
                best_conf = conf

        # ✅ Reject low-confidence token
        if best_conf < min_confidence:
            #print("LOW CONFIDENCE:", best_conf, "for token:", token)
            token["pixel"] = (float(axis_x), cy)
            token["snap_confidence"] = 0
        else:
            token["pixel"] = (float(axis_x), float(best_y))
            token["snap_confidence"] = float(best_conf)
        return token

    # ============================================================
    # X-axis snapping
    # ============================================================
    if axis == "x":
        axis_y = int(x_axis_y)
        win = max(1, int(W * search_frac))
        x0, x1 = max(0, cx - win), min(W - 1, cx + win)

        best_x = cx
        best_score = -1e9
        best_conf = 0.0

        for x in range(x0, x1 + 1):
            crop = _extract_tick_crop(gray, x, axis_y)
            conf = _model_confidence(crop)

            dist = abs(x - cx)
            score = conf + dist_bias_weight * dist_bias(dist)

            if score > best_score:
                best_score = score
                best_x = x
                best_conf = conf

        if best_conf < min_confidence:
            #print("LOW CONFIDENCE:", best_conf, "for token:", token)
            token["pixel"] = (cx, float(axis_y))
            token["snap_confidence"] = 0
        else:
            token["pixel"] = (float(best_x), float(axis_y))
            token["snap_confidence"] = float(best_conf)
        return token

    return None

def _intersect_with_llm(
    tokens: List[Dict[str, Any]],
    llm_vals: List[float],                 # DISPLAY-domain ticks (authoritative)
    value_tol_abs: float,
    value_tol_rel: float,
    axis_span: Tuple[float, float] = None, # DISPLAY-domain span (v0,v1)
    pixel_span: Tuple[float, float] = None,# (px0,px1)
    axis: str = "x",
    pixel_tol: float = 0.1,               # fraction of image dimension
    breaking: bool = False,
    break_expand: float = 1.3,
    border_factor: float = 1.8,
    mode: str = "linear",                  # "linear" | "log2" | "log10" | "logK"
    log_base: float = 10.0,                # used when mode startswith("log") but no base in mode
    H: int = 500,
    W: int = 500,
):
    """
    LLM ∩ OCR gating.

    - Membership is checked ONLY in DISPLAY space (never transformed).
    - Geometry is checked in FIT space using crude LLM span -> pixel span mapping.
    - Supports logK via mode like "log2"/"log10"/"log" + log_base.
    - Border ticks get extra tolerance.
    - Robust to duplicates: for each tick value, keeps only the best residual token.
    - Optional break-aware three-model logic.
    """
    explanations: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []

    # absolute pixel tolerance
    pixel_tol_px = pixel_tol * (W if axis == "x" else H)

    if not tokens or not llm_vals:
        return [], [{"selected": False, "reason": "No tokens or no LLM tick values"}]

    llm_vals_arr = np.asarray(llm_vals, float)
    vmin = float(np.min(llm_vals_arr))
    vmax = float(np.max(llm_vals_arr))

    # -----------------------------
    # Build crude mapping in FIT space
    # -----------------------------
    have_span = False
    scale = None
    px0 = px1 = None
    fv0 = fv1 = None

    if axis_span is not None and pixel_span is not None:
        v0, v1 = axis_span
        px0, px1 = pixel_span

        fv0 = _to_fit_value(v0, mode, log_base)
        fv1 = _to_fit_value(v1, mode, log_base)

        if (
            fv0 is not None and fv1 is not None
            and np.isfinite(fv0) and np.isfinite(fv1)
            and (fv1 != fv0)
            and px0 is not None and px1 is not None
        ):
            scale = (px1 - px0) / float(fv1 - fv0)
            have_span = True

    # -----------------------------
    # First pass: membership, then residual compute (if possible)
    # -----------------------------
    cand = []

    for t in tokens:
        txt = _tok_text(t)
        p = _tok_pixel_center(t)

        entry = {
            "token": t,
            "text": txt,
            "pixel": p,
            "value": None,
            "selected": False,
            "reason": "",
        }

        if not _is_numeric_str(txt):
            entry["reason"] = f"Rejected: non-numeric text '{txt}'"
            explanations.append(entry)
            continue

        try:
            v = float(txt)
        except Exception:
            entry["reason"] = f"Rejected: cannot parse '{txt}'"
            explanations.append(entry)
            continue

        entry["value"] = v

        # DISPLAY-space membership only
        atol = max(value_tol_abs, value_tol_rel * max(1.0, abs(v)))
        if not np.any(np.isclose(v, llm_vals_arr, rtol=0.0, atol=atol)):
            entry["reason"] = "Rejected: not in LLM ticks"
            explanations.append(entry)
            continue

        # If we cannot do geometry, accept by membership (but note: this can let garbage through)
        if (not have_span) or (p is None):
            entry["selected"] = True
            entry["reason"] = "Selected: numeric match; no usable span/pixel"
            cand.append(entry)  # keep as candidate; we will still dedupe by value
            continue

        # FIT-space value for geometry
        fv = _to_fit_value(v, mode, log_base)
        if fv is None or not np.isfinite(fv):
            entry["reason"] = "Rejected: invalid fit value for axis mode"
            explanations.append(entry)
            continue

        x_px, y_px = p
        px_actual = float(x_px if axis == "x" else y_px)

        # predicted pixel from crude mapping
        if axis == "x":
            px_pred = float(px0 + (fv - fv0) * scale)
        else:
            px_pred = float(px1 - (fv - fv0) * scale)

        if not np.isfinite(px_pred):
            entry["reason"] = "Rejected: non-finite predicted pixel"
            explanations.append(entry)
            continue

        entry["px_actual"] = px_actual
        entry["px_pred"] = px_pred
        entry["residual"] = abs(px_actual - px_pred)

        cand.append(entry)

    if not cand:
        return [], explanations

    # -----------------------------
    # If geometry exists: residual gate (border-aware)
    # else: keep best per value by snap_confidence / area heuristic
    # -----------------------------
    by_value_best = {}  # v -> best entry
    for entry in cand:
        v = entry["value"]

        # Border-aware tol if residual exists
        if "residual" in entry:
            tol = pixel_tol_px * (border_factor if (v == vmin or v == vmax) else 1.0)
            ok = (entry["residual"] <= tol)

            if not ok and not breaking:
                entry["selected"] = False
                entry["reason"] = f"Rejected: residual {entry['residual']:.1f} > {tol:.1f}"
                explanations.append(entry)
                continue

            # If breaking, we postpone decision to break block below
            if not breaking:
                entry["selected"] = True
                entry["reason"] = f"Selected: residual {entry['residual']:.1f} ≤ {tol:.1f}"
        else:
            # No residual path: mark selected but we will dedupe heavily
            entry["selected"] = True
            entry["reason"] = "Selected: membership only (no geometry)"

        # Deduplicate per tick value: keep the best representative
        prev = by_value_best.get(v)
        if prev is None:
            by_value_best[v] = entry
        else:
            # Prefer smaller residual if available; else prefer higher snap_confidence
            r_new = entry.get("residual", float("inf"))
            r_old = prev.get("residual", float("inf"))

            if r_new < r_old:
                by_value_best[v] = entry
            elif r_new == r_old:
                sc_new = float(entry["token"].get("snap_confidence", 0.0))
                sc_old = float(prev["token"].get("snap_confidence", 0.0))
                if sc_new > sc_old:
                    by_value_best[v] = entry

        explanations.append(entry)

    # If breaking logic requested and we have spans, apply it ONLY on the best-per-value set
    if breaking and have_span:
        # Re-run break-aware selection on best representatives only
        best_entries = list(by_value_best.values())
        kept = []

        axis_range_px = abs(px1 - px0)
        shift = 0.2 * axis_range_px
        comp_factor = 0.8
        scale_C = scale * comp_factor

        tol_A = pixel_tol_px * 1.5 * break_expand
        tol_B = pixel_tol_px * 1.5 * break_expand
        tol_C = pixel_tol_px * 2.0 * break_expand

        for entry in best_entries:
            v = entry["value"]
            p = entry["pixel"]
            if p is None:
                continue

            fv = _to_fit_value(v, mode, log_base)
            if fv is None or not np.isfinite(fv):
                continue

            x_px, y_px = p
            pa = float(x_px if axis == "x" else y_px)

            # Model A
            if axis == "x":
                pxA = float(px0 + (fv - fv0) * scale)
                pxB = float((px0 + shift) + (fv - fv0) * scale)
                pxC = float(px0 + (fv - fv0) * scale_C)
            else:
                pxA = float(px1 - (fv - fv0) * scale)
                pxB = float((px1 - shift) - (fv - fv0) * scale)
                pxC = float(px1 - (fv - fv0) * scale_C)

            rA = abs(pa - pxA)
            rB = abs(pa - pxB)
            rC = abs(pa - pxC)

            border_mul = border_factor if (v == vmin or v == vmax) else 1.0

            okA = rA <= (tol_A * border_mul)
            okB = rB <= (tol_B * border_mul)
            okC = rC <= (tol_C * border_mul)

            if okA or okB or okC:
                kept.append(entry["token"])

        return kept, explanations

    # Non-breaking: return best-per-value tokens (already residual-gated if possible)
    kept = [e["token"] for e in by_value_best.values() if e.get("selected")]
    return kept, explanations

def _fit_axis_from_ticks(
    tokens: List[Dict[str,Any]],
    want_axis: str,
    axis_line_px: float,
    mode: str,
    log_base: float = 10.0,
    image=None,
) -> Optional[AxisCal]:
    """
    Fit pixel = a * f(value) + b using gated OCR ticks.
    - x-axis: map x_value -> x_pixel; enforce slope > 0
    - y-axis: map y_value -> y_pixel; enforce slope < 0 (image y grows downward)
    - f is identity or log10 depending on 'mode'
    :param tokens: OCR tokens with pixel information
    :param want_axis: The axis to fit (either "x" or "y")
    :param axis_line_px: The x or y coordinate of the axis line in image pixels
    :param mode: Either "identity" or "log10"
    :param log_base: The base of the logarithm (default 10.0)
    :param image: The image to fit the axis to (for debugging)
    :return: Optional[AxisCal] containing the best fit, or None if the fit is bad
    """
    vals_fit = []
    pixs = []
    print(f"Fitting axis '{want_axis}' with mode '{mode}' and base {log_base}...")
    for t in tokens:
        txt = _tok_text(t)
        if not _is_numeric_str(txt):
            print("Text for token:", t, "is", txt)
            continue

        p = _tok_pixel_center(t)
        if p is None:
            print("Pixel for token:", t, "is bugged")
            continue

        v_disp = float(txt)
        v_fit = _to_fit_value(v_disp, mode, log_base)
        if v_fit is None:
            continue

        if want_axis == "x":
            vals_fit.append(v_fit)
            pixs.append(p[0])
        else:
            vals_fit.append(v_fit)
            pixs.append(p[1])

    if len(vals_fit) < 2:
        print("Less than 2 ticks for axis", vals_fit, len(tokens))
        return None

    vals_fit = np.asarray(vals_fit, float)
    pixs = np.asarray(pixs, float)
    
    a, b = majority_polyfit(vals_fit, pixs, axis=want_axis, image=image)

    if want_axis == "x" and a < 0:
        a = abs(a)
    if want_axis == "y" and a > 0:
        a = -abs(a)

    return AxisCal(float(a), float(b), want_axis, mode)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

def _fit_axis_from_limits(want_axis: str,
                          limits: Optional[Tuple[float,float]],
                          img_hw: Tuple[int,int],
                          x_axis_y: Optional[int],
                          y_axis_x: Optional[int],
                          mode: str) -> Optional[AxisCal]:
    """
    Fit an axis from a set of limits, given as pixel coordinates in the image.

    The limits are a tuple of two floats, which represent the start and end points of the axis.
    The image height and width are given as a tuple of two ints.
    The x and y coordinates of the axis line are given as optional ints.
    The mode is given as a string, which can be either "identity" or "logK".
    The function returns an Optional[AxisCal] containing the best fit, or None if the fit is bad.
    """
    if not limits or len(limits) != 2:
        return None
    H, W = img_hw
    v0, v1 = map(float, limits)
    if mode.startswith("log"):
        if v0 <= 0 or v1 <= 0:
            return None
        log_base = float(mode[3:])
        l = np.log(log_base)
        v0, v1 = math.log(v0)/l, math.log(v1)/l
    if want_axis == "x":
        x0 = y_axis_x if y_axis_x is not None else int(0.1*W)
        x1 = W-1
        M = np.array([[v0,1.0],[v1,1.0]], float)
        Y = np.array([x0, x1], float)
        a,b = np.linalg.lstsq(M, Y, rcond=None)[0]
        if a < 0: a = abs(a)
        return AxisCal(float(a), float(b), 'x', mode)
    else:
        y_top = 0
        y_bottom = x_axis_y if x_axis_y is not None else int(0.9*H)
        M = np.array([[v0,1.0],[v1,1.0]], float)
        Y = np.array([y_bottom, y_top], float)  # up in value = up in image (lower pixel)
        a,b = np.linalg.lstsq(M, Y, rcond=None)[0]
        if a > 0: a = -abs(a)
        return AxisCal(float(a), float(b), 'y', mode)

# =============================
# Scale/precision helpers
# =============================
def _parse_scale_from_intervals(intervals: List[Dict[str, Any]], axis: str) -> Tuple[Optional[float], Optional[int]]:
    """
    Extract (step, label_step) for the given axis ("x" or "y") from scale interval.
    Returns (step, label_step) or (None, None) if missing.
    """
    for it in intervals:
        if str(it.get("id", "")).lower() == "scale":
            v = it.get(axis, None)
            if not isinstance(v, dict):
                return None, None
            try:
                step = float(v.get("step")) if v.get("step") is not None else None
            except Exception:
                step = None
            try:
                label_step = int(v.get("label_step")) if v.get("label_step") is not None else None
            except Exception:
                label_step = None
            return step, label_step
    return None, None

def _tok_quad_or_box(t):
    """Extract a bounding box from the token, handling various possible formats."""
    B = t.get("bbox") or t.get("box") or t.get("rect")

    if B is None:
        return None

    # Handle polygon form: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    if isinstance(B, (list, tuple)) and len(B) == 4 and isinstance(B[0], (list, tuple)):
        xs = [p[0] for p in B]
        ys = [p[1] for p in B]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return int(x0), int(y0), int(x1), int(y1)

    # Handle flat [x0, y0, x1, y1]
    if isinstance(B, (list, tuple)) and len(B) == 4:
        x0, y0, x1, y1 = map(float, B)
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        return int(x0), int(y0), int(x1), int(y1)

    return None


def _extract_crop(img: np.ndarray, box: Tuple[int,int,int,int], pad: int=2) -> Tuple[np.ndarray, Tuple[int,int]]:
    """Extract a cropped region from the image given a bounding box, with optional padding."""
    H,W = img.shape[:2]
    x0,y0,x1,y1 = box
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W-1, x1 + pad); y1 = min(H-1, y1 + pad)
    return img[y0:y1+1, x0:x1+1], (x0,y0)

def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Create a binary mask of the inked regions in the grayscale image."""
    # robust binarization for small labels
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    # light open to remove specks, keep strokes
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)))
    return th

def _projection_profile(mask: np.ndarray, axis: str) -> np.ndarray:
    """Compute the projection profile of the binary mask along the specified axis."""
    # axis == 'x' → vertical projection (sum per column) for x-axis labels
    # axis == 'y' → horizontal projection (sum per row)   for y-axis labels
    if axis == 'x':
        prof = mask.sum(axis=0)
    else:
        prof = mask.sum(axis=1)
    # mild smoothing
    return cv2.blur(prof.astype(np.float32).reshape(-1,1), (3,1)).ravel()

from scipy.signal import find_peaks

def _find_valleys(profile, min_prom_frac=0.05, min_sep_px=2):
    """Find valleys in the projection profile, which correspond to gaps between ticks, which will be used
    later to merge ticks.
    """
    if profile.size < 8:
        return []
    p = (profile - profile.min()) / max(1e-6, (profile.max() - profile.min()))
    ip = 1.0 - p
    peaks, _ = find_peaks(ip, prominence=min_prom_frac, distance=min_sep_px)
    return peaks.tolist()

def _intensity_centroid(mask: np.ndarray) -> Tuple[float,float]:
    """Compute the intensity centroid of the binary mask, which will be used as a fallback tick position if no good peaks are found.
    """
    M = cv2.moments(mask, binaryImage=True)
    if M['m00'] <= 1e-6:
        h,w = mask.shape[:2]
        return w*0.5, h*0.5
    cx = M['m10']/M['m00']; cy = M['m01']/M['m00']
    return cx, cy

def _is_all_digits_or_sign(s: str) -> bool:
    """Check if the string consists only of digits, signs, decimal points, commas, or spaces."""
    if not isinstance(s, str) or not s:
        return False
    return all(ch.isdigit() or ch in "+-., " for ch in s)

def _closest_in_llm(val: float, llm_vals: List[float], atol_abs: float, atol_rel: float) -> Optional[float]:
    """Find the closest value in llm_vals to val within the given absolute and relative tolerances.
     Returns the closest value if found, or None if no value is within the tolerances.
     If no value is within the tolerances, it means that the tick was noise"""
    if not llm_vals:
        return None
    for lv in llm_vals:
        atol = max(atol_abs, atol_rel*max(1.0, abs(lv)))
        if abs(val - lv) <= atol:
            return lv
    return None

def _format_precise(val: float, step: Optional[float], axis_mode: str) -> str:
    """
    Format a DISPLAY-domain numeric value for output.

    Rules:
    - Log axis ("log"): use compact significant-figure formatting.
    - Linear axis: choose decimals relative to step size.
    - Never show unnecessary trailing zeros.
    """

    # --- Log axis: significant figures only ---
    if axis_mode == "log":
        return f"{val:.6g}"

    # --- Linear axis fallback ---
    if step is None or step <= 0 or not math.isfinite(step):
        return f"{val:.6g}"

    # Determine decimals from step magnitude
    try:
        decimals = max(0, int(-math.floor(math.log10(step))) + 2)
    except Exception:
        return f"{val:.6g}"

    s = f"{val:.{decimals}f}"

    # Strip trailing zeros and dangling decimal point
    if "." in s:
        s = s.rstrip("0").rstrip(".")

    return s

# ==========================================================================================
# Tools to select the best matching subset of ticks for axis fitting
# ==========================================================================================
def _vandermonde(x, deg):
    """Generate a Vandermonde matrix for polynomial fitting.
     - x: 1D array of input values
     - deg: degree of the polynomial
     Returns a matrix where each column corresponds to x raised to a power, ordered from highest degree to lowest.
    """
    # Columns in np.polyval order: x^deg ... x^0
    return np.vander(x, N=deg+1, increasing=False)

def _ols_polyfit_on_set(x, y, deg):
    """Perform ordinary least squares polynomial fit on the given (x,y) set.
     - x: 1D array of input values
     - y: 1D array of target values
     - deg: degree of the polynomial
     Returns the polynomial coefficients in np.polyval order (highest degree first)."""
    V = _vandermonde(x, deg)
    beta, *_ = np.linalg.lstsq(V, y, rcond=None)
    return beta  # np.polyval order

def _huber_weights(residuals, delta):
    """
    Compute Huber weights for the given residuals and delta parameter.
    - residuals: 1D array of residuals
    - delta: Huber parameter
    Returns an array of weights corresponding to each residual.
    """
    r = np.abs(residuals)
    w = np.ones_like(r, dtype=float)
    mask = r > delta
    w[mask] = delta / (r[mask] + 1e-12)
    return w

def _irls_huber_on_set(x, y, deg, beta0=None, max_iters=10, delta=None, tol=1e-6):
    """
    IRLS with Huber loss, *only* on the provided (x,y) set.
    Returns beta in np.polyval order (highest degree first).
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if beta0 is None:
        beta = _ols_polyfit_on_set(x, y, deg)
    else:
        beta = beta0.copy()

    if delta is None:
        span = max(1.0, float(np.max(y) - np.min(y)))
        delta = 0.03 * span  # ~3% of span

    for _ in range(max_iters):
        yhat = np.polyval(beta, x)
        r = y - yhat
        w = _huber_weights(r, delta)

        # Weighted least squares on this set
        V = _vandermonde(x, deg)
        Wsqrt = np.sqrt(np.clip(w, 1e-12, None))
        VW = V * Wsqrt[:, None]
        yW = y * Wsqrt
        beta_new, *_ = np.linalg.lstsq(VW, yW, rcond=None)

        if np.linalg.norm(beta_new - beta) <= tol * (1.0 + np.linalg.norm(beta)):
            beta = beta_new
            break
        beta = beta_new
    return beta

def robust_polyfit(
    x,
    y,
    deg=1,
    n_trials=200,
    residual_tol=5.0,
    alpha=0.5,
    lambda_inlier=1.0,
    gamma_span=0.01,          # NEW: span reward
    random_state=None,
):
    """
    Robust polynomial fit with controlled exploration + axis-aware scoring.

    Additions vs original:
      - Huber loss instead of MSE
      - Score evaluated on the SAME set used for fitting
      - Span reward to avoid degenerate local fits
      - Leave-one-out rescue for single catastrophic outliers
    """
    
    print("we have a grand total of ", len(x), " points to fit a degree ", deg, " polynomial.")
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    N = len(x)
    if N <= deg + 1:
        return np.polyfit(x, y, deg)

    s_min = max(2, deg + 1)
    s_max = N

    best_score = -np.inf
    best_beta = None
    best_set = None

    # -------------------------
    # Robust loss
    # -------------------------
    def huber_mean(resid, delta):
        a = np.abs(resid)
        quad = a <= delta
        out = np.empty_like(a, dtype=float)
        out[quad] = 0.5 * (a[quad] ** 2)
        out[~quad] = delta * (a[~quad] - 0.5 * delta)
        return float(out.mean())

    def score_on(beta, xs, ys):
        yhat = np.polyval(beta, xs)
        resid = ys - yhat
        loss = huber_mean(resid, residual_tol)
        n_used = len(xs)
        span = float(xs.max() - xs.min()) if n_used >= 2 else 0.0
        score = lambda_inlier * n_used - alpha * loss + gamma_span * span
        return score, loss, n_used, resid

    # --------------------
    # Case A: N <= 10 (exhaustive)
    # --------------------
    if N <= 10:
        for s in range(s_min, s_max + 1):
            for subset in combinations(range(N), s):
                subset = np.array(subset, int)
                xs, ys = x[subset], y[subset]

                beta0 = _ols_polyfit_on_set(xs, ys, deg)
                yhat = np.polyval(beta0, xs)
                resid = np.abs(ys - yhat)

                if np.any(resid > residual_tol):
                    continue

                score0, _, _, _ = score_on(beta0, xs, ys)
                if score0 > best_score:
                    best_score = score0
                    best_beta = beta0
                    best_set = subset

                beta1 = _irls_huber_on_set(xs, ys, deg, beta0=beta0, max_iters=10)
                score1, _, _, _ = score_on(beta1, xs, ys)
                if score1 > best_score:
                    best_score = score1
                    best_beta = beta1
                    best_set = subset

        if best_beta is None:
            return np.polyfit(x, y, deg)
        return best_beta

    # --------------------
    # Case B: N > 10 (randomized)
    # --------------------
    sizes = np.arange(s_min, s_max + 1)
    weights = np.ones_like(sizes, float)
    weights[sizes <= 4] *= 3.0
    probs = weights / weights.sum()

    for _ in range(int(n_trials)):
        s = int(rng.choice(sizes, p=probs))
        subset_idx = np.sort(rng.choice(N, size=s, replace=False))
        xs, ys = x[subset_idx], y[subset_idx]

        beta0 = _ols_polyfit_on_set(xs, ys, deg)
        yhat = np.polyval(beta0, xs)
        resid = np.abs(ys - yhat)

        if np.any(resid > residual_tol):
            continue

        score0, _, _, _ = score_on(beta0, xs, ys)
        if score0 > best_score:
            best_score = score0
            best_beta = beta0
            best_set = subset_idx

        # --- Expand ---
        yhat_all = np.polyval(beta0, x)
        resid_all = np.abs(y - yhat_all)
        expanded_idx = np.where(resid_all <= residual_tol)[0]

        if expanded_idx.size < s_min:
            continue

        xs2, ys2 = x[expanded_idx], y[expanded_idx]
        beta1 = _irls_huber_on_set(xs2, ys2, deg, beta0=beta0, max_iters=10)
        score1, loss1, n1, resid2 = score_on(beta1, xs2, ys2)

        # --- Leave-one-out rescue (ONLY if one dominates) ---
        if xs2.size >= s_min + 1:
            worst = np.argmax(np.abs(resid2))
            xs3 = np.delete(xs2, worst)
            ys3 = np.delete(ys2, worst)

            beta2 = _irls_huber_on_set(xs3, ys3, deg, beta0=beta1, max_iters=10)
            score2, _, _, _ = score_on(beta2, xs3, ys3)

            if score2 > score1:
                score1 = score2
                beta1 = beta2

        if score1 > best_score:
            best_score = score1
            best_beta = beta1
            best_set = expanded_idx

    if best_beta is None:
        return np.polyfit(x, y, deg)
   
    return best_beta

def majority_polyfit(x, y, deg=1, axis=None, image=None, random_state=None, residual_threshold=None):
    """
    Drop-in replacement for np.polyfit, but robust to outliers using RANSAC.

    Returns coefficients [c0, c1, ..., c_deg] like np.polyfit,
    where c0*x^deg + ... + c_deg is the polynomial.
    """
    new_method = True
    if new_method:
        return robust_polyfit(x, y, deg=deg, random_state=random_state)
    X = x[:, None]
    if len(x) <= deg + 1:
        return np.polyfit(x, y, deg)
    # Handle sklearn API change
    if sklearn.__version__ >= "1.2":
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=residual_threshold,
            random_state=random_state
        )
    else:
        ransac = RANSACRegressor(
            base_estimator=LinearRegression(),
            residual_threshold=residual_threshold,
            random_state=random_state
        )
    model = make_pipeline(PolynomialFeatures(deg, include_bias=True), ransac)
    model.fit(X, y)
    # Get coefficients from the inner linear regressor
    lr = model.named_steps["ransacregressor"].estimator_
    # lr.coef_ has shape (n_features,) = (deg+1,)
    coefs = lr.coef_.ravel()
    intercept = lr.intercept_
    # Combine, but drop the bias term since intercept already handles it
    full_coefs = np.r_[intercept, coefs[1:]]  
    ransac_step = model.named_steps["ransacregressor"]
    inliers = X[ransac_step.inlier_mask_]
    outliers = X[~ransac_step.inlier_mask_]
    return full_coefs[::-1]

# ====================================
# Valley split + LLM aware merge
# ====================================

def _split_merged_digits_with_llm(
    img: np.ndarray,
    token: Dict[str,Any],
    axis: str,
    llm_vals: List[float],
    atol_abs: float = 0.25,
    atol_rel: float = 0.02,
    oversize_ratio: float = 1.8,
    min_prom_frac: float = 0.05,
    min_sep_px: int = 2
) -> List[Dict[str,Any]]:
    """
    If token’s box is oversized (vs typical digit) or text looks run-on (e.g. '9101112'),
    split by projection valleys, then greedily fuse adjacent chunks so that
    concatenated text parses to a number present in llm_vals (within tolerance).
    Each kept chunk gets its intensity centroid and becomes a new tick token.

    Returns: list of refined tokens (could be [token] if no split needed).
    """
    box = _tok_quad_or_box(token)
    if box is None:
        # no box: nothing to split
        return [token]

    # Extract crop & ink
    crop, (ox,oy) = _extract_crop(img, box, pad=1)
    if crop.size == 0:
        return [token]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = _ink_mask(gray)

    h,w = mask.shape[:2]
    txt = str(token.get("text", ""))

    # Heuristics: should we attempt splitting?
    should_try = False
    if _is_all_digits_or_sign(txt) and len(txt.replace(" ","")) >= 3:
        # long run-on looking string
        should_try = True
    # oversized box check vs a “single-digit-ish” size proxy:
    # use height for x-axis (horizontal text), width for y-axis (rotated cases)
    if axis == 'x' and w >= oversize_ratio * max(6, h*0.35):
        should_try = True
    if axis == 'y' and h >= oversize_ratio * max(6, w*0.35):
        should_try = True

    if not should_try:
        return [token]

    # Projection
    prof = _projection_profile(mask, axis)
    cuts = _find_valleys(prof, min_prom_frac=min_prom_frac, min_sep_px=min_sep_px)

    if len(cuts) == 0:
        # No clear valleys → keep as is
        return [token]

    # Build primitive segments between cuts
    segs = []
    if axis == 'x':
        xs = [0] + cuts + [w-1]
        for i in range(len(xs)-1):
            sx, ex = xs[i], xs[i+1]
            if ex - sx < 2: 
                continue
            segs.append(((sx,0,ex,h-1), mask[:, sx:ex]))
    else:
        ys = [0] + cuts + [h-1]
        for i in range(len(ys)-1):
            sy, ey = ys[i], ys[i+1]
            if ey - sy < 2:
                continue
            segs.append(((0,sy,w-1,ey), mask[sy:ey, :]))

    if not segs:
        return [token]

    # OPTIONAL: mini-OCR per segment to get digit strings (but we can also just concatenate raw)
    # Here we stick to concatenating original token.text by segment order using binary bounds.
    # Greedy L→R (or T→B) concatenation checked against LLM ticks
    refined: List[Dict[str,Any]] = []
    i = 0
    while i < len(segs):
        # Try growing window segs[i:j]
        accepted_j = None
        accepted_val = None
        accepted_centroid = None
        accepted_box_local = None

        for j in range(i+1, len(segs)+1):
            # Combine boxes i..j-1
            if axis == 'x':
                x0 = segs[i][0][0]; x1 = segs[j-1][0][2]
                y0 = 0; y1 = h-1
                submask = mask[:, x0:x1]
            else:
                y0 = segs[i][0][1]; y1 = segs[j-1][0][3]
                x0 = 0; x1 = w-1
                submask = mask[y0:y1, :]

            # Estimate centroid inside this combined region
            cx_local, cy_local = _intensity_centroid(submask)

            # We need a candidate numeric value; strategy:
            #  - If token.text is numeric-like, slice by proportional ranges (fallback),
            #  - Otherwise, rely on LLM-only geometry and skip numeric parse (we can’t validate).
            # Simpler: try to OCR submask? (you said “don’t code OCR here”), so we parse from token.text if plausible:
            val_candidate = None
            if _is_all_digits_or_sign(txt):
                # Map segment coverage to substring indices by width/height proportion
                if axis == 'x':
                    frac0, frac1 = x0/float(w), x1/float(w)
                else:
                    frac0, frac1 = y0/float(h), y1/float(h)
                s = txt.replace(" ", "")
                a = max(0, int(round(frac0 * len(s))))
                b = min(len(s), int(round(frac1 * len(s))))
                subs = s[a:b]
                # normalize separators
                subs = subs.replace(",", ".")
                try:
                    # common multi-digit formation; allow “-”, “.” once
                    if subs and (subs.strip("-").replace(".","",1).isdigit()):
                        val_candidate = float(subs)
                except Exception:
                    val_candidate = None

            # If we got a numeric candidate, test against LLM ticks
            if val_candidate is not None:
                match = _closest_in_llm(val_candidate, llm_vals, atol_abs, atol_rel)
                if match is not None:
                    accepted_j = j
                    accepted_val = match
                    if axis == 'x':
                        # centroid in global image coords
                        gx = ox + x0 + cx_local
                        gy = oy + cy_local
                    else:
                        gx = ox + cx_local
                        gy = oy + y0 + cy_local
                    accepted_centroid = (float(gx), float(gy))
                    if axis == 'x':
                        accepted_box_local = (ox + x0, oy + 0, ox + x1, oy + h-1)
                    else:
                        accepted_box_local = (ox + 0, oy + y0, ox + w-1, oy + y1)
            # keep trying to extend (prefer longer valid merges like “10” over “1”)

        if accepted_j is None:
            # Nothing matched LLM ticks; advance by 1 (drop tiny noise)
            i += 1
        else:
            # Emit refined token
            new_t = dict(token)  # copy base fields
            new_t["value"] = float(accepted_val)
            new_t["text"] = str(accepted_val)
            new_t["pixel"] = accepted_centroid
            new_t["bbox"]  = accepted_box_local
            refined.append(new_t)
            i = accepted_j
    

    # If nothing accepted, keep original token as fallback
    return refined if refined else [token]

def _refine_axis_tokens_with_valleys(
    img: np.ndarray,
    tokens: List[Dict[str,Any]],
    axis: str,
    llm_vals: List[float],
    atol_abs: float,
    atol_rel: float
) -> List[Dict[str,Any]]:
    """Apply the split-and-merge logic to all tokens, returning a refined list of tokens where merged digits may have been split and validated against LLM ticks.
    """
    out: List[Dict[str,Any]] = []
    for t in tokens:
        out.extend(
            _split_merged_digits_with_llm(
                img, t, axis, llm_vals,
                atol_abs=atol_abs, atol_rel=atol_rel
            )
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# List-compatible return so existing callers keep working (acts like list)
class IntervalCropResult(list):
    def __init__(self, boxes, calibration, meta):
        super().__init__(boxes)
        self.calibration = calibration  # {"x":{a,b,mode}, "y":{a,b,mode}, "x_axis_y", "y_axis_x"}
        self.meta = meta                # per-crop metadata (paths, bbox, counts, etc.)
# ─────────────────────────────────────────────────────────────────────────────


# =============================
# Main
# =============================
def make_interval_crops(
    image_path: str,
    intervals: List[Dict[str, Any]],
    out_dir: str = "out_interval",
    margin_px: int = 0,
    min_size_px: int = 48,
    save_meta: bool = True,
    # Optional overrides (still supported)
    x_ticks_override: Optional[List[Tuple[float, float]]] = None,
    y_ticks_override: Optional[List[Tuple[float, float]]] = None,
    x_limits_override: Optional[Tuple[float, float]] = None,
    y_limits_override: Optional[Tuple[float, float]] = None,
    # Axis strip paddings
    y_axis_left_pad: int = 40,
    y_axis_right_pad: int = 18,
    x_axis_above_pad: int = 22,
    x_axis_below_pad: int = 28,
    spacer_px: int = 6,
    # Pixel sampling & guardrails
    # gather safeguards
    near_tol: int = 10,
    far_tol: int = 5,
    # Text style (kept, but labels are suppressed in strips)
    font_scale: float = 0.45,
    thickness: int = 1,
    # LLM guidance
    llm_axis_info: Optional[Dict[str, Any]] = None,  # {"x_axis":{"type","range","ticks"}, "y_axis":{...}}
    llm_value_tol_abs: float = 1e-6,
    llm_value_tol_rel: float = 0.02,
    res: Optional[Dict[str, Any]] = None,
    cal_x: Optional[AxisCal] = None,
    cal_y: Optional[AxisCal] = None,
    tokens_all: Optional[List[Dict[str,Any]]] = None,
) -> "IntervalCropResult":
    """
    Main function to create interval crops from an image, with LLM guidance and robust token refinement.
    We first detect axes, then gather tokens, refine them with valley splitting and LLM tick matching, and finally create crops based on the refined tokens and calibration.
    Then we return an IntervalCropResult which is a list of crop metadata, along with calibration and other metadata for downstream use.
    """
    os.makedirs(out_dir, exist_ok=True)
    crops_dir = os.path.join(out_dir, "crops"); os.makedirs(crops_dir, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    H, W = img.shape[:2]

    # 0) Detect axes (no bbox needed)
    x_axis_y, y_axis_x, x_axis_span, y_axis_span = _detect_axes(img)

    if cal_x is None or cal_y is None:
        # 1) LLM axis info (type/range/ticks)
        xi = llm_axis_info.get("x_axis", {}) if llm_axis_info else {}
        yi = llm_axis_info.get("y_axis", {}) if llm_axis_info else {}
        x_rotation = xi.get("rotation_deg", 0.0)
        y_rotation = yi.get("rotation_deg", 0.0)
        x_mode = str(xi.get("type", "linear")).lower() if xi else "linear"
        y_mode = str(yi.get("type", "linear")).lower() if yi else "linear"
        x_break = xi.get("break", False)
        y_break = yi.get("break", False)
        if not x_mode.startswith("log"): x_mode = "linear"
        if not y_mode.startswith("log"): y_mode = "linear"
        x_llm_ticks = _coerce_float_list(xi.get("ticks"))
        y_llm_ticks = _coerce_float_list(yi.get("ticks"))
        x_limits_llm = tuple(xi.get("range", ())) if isinstance(xi.get("range", ()), (list,tuple)) else None
        y_limits_llm = tuple(yi.get("range", ())) if isinstance(yi.get("range", ()), (list,tuple)) else None

        # 2) OCR whole image
        if res == None:
            res = ocr.process_image(image_path, x_axis_y=x_axis_y, y_axis_x=y_axis_x, per_plot_ocr=False, x_rotation=x_rotation, y_rotation=y_rotation)
        tokens_all: List[Dict[str, Any]] = []
        for plot in res.get("plots", []):
            tokens_all.extend(plot.get("ticks", []))
            tokens_all.extend(plot.get("labels", []))

        if not x_limits_llm and x_llm_ticks:
            lo, hi = float(min(x_llm_ticks)), float(max(x_llm_ticks))
            pad = 0.05 * max(1.0, hi - lo)
            x_limits_llm = (lo - pad, hi + pad)
        if not y_limits_llm and y_llm_ticks:
            lo, hi = float(min(y_llm_ticks)), float(max(y_llm_ticks))
            pad = 0.05 * max(1.0, hi - lo)
            y_limits_llm = (lo - pad, hi + pad)

        # 3) Gather likely tick-label tokens near each axis    
        def gather_axis_with_explanation(
            tokens,
            axis,              # "x" or "y"
            axis_px,           # pixel coordinate of this axis (y-axis x-position, or x-axis y-position)
            other_axis_px,     # pixel coordinate of the other axis
            near_tol=10,       # allowed distance from target axis in %
            far_tol=5         # must be farther than this from the other axis in %
        ):
            """
            Geometry-based axis tick selection with detailed explanations.

            Returns:
                selected_tokens, explanations_list

            Explanation for each token:
                {
                    "text": "...",
                    "pixel": (x,y),
                    "selected": True/False,
                    "reason": "clear English justification"
                }
            """
            if axis  == "x":
                near_tol = near_tol * H/100
                far_tol = far_tol  * W/100
            if axis == "y":
                near_tol = near_tol * W/100
                far_tol = far_tol  * H/100

            selected = []
            explanations = []

            for t in tokens:
                txt = str(t.get("text"))
                p   = t.get("pixel")

                # --- 1. must have pixel ---
                if p is None:
                    explanations.append({
                        "text": txt, "pixel": None,
                        "selected": False,
                        "reason": "Rejected: token has no pixel coordinate"
                    })
                    continue

                x, y = p

                # --- 2. must be numeric ---
                try:
                    _ = float(txt)
                except:
                    explanations.append({
                        "text": txt, "pixel": p,
                        "selected": False,
                        "reason": f"Rejected: text '{txt}' is not numeric"
                    })
                    continue

                # --- 3. compute axis distances ---
                if axis == "x":
                    d_axis = abs(y - axis_px)       # y-distance from x-axis baseline
                    d_other = abs(x - other_axis_px)
                else: # axis == "y"
                    d_axis = abs(x - axis_px)       # x-distance from y-axis baseline
                    d_other = abs(y - other_axis_px)

                near = d_axis <= near_tol
                far  = d_other > far_tol

                # --- 4. decision ---
                if near and far:
                    explanations.append({
                        "text": txt, "pixel": p, "selected": True,
                        "reason": (
                            f"Selected: near {axis}-axis (dist={d_axis} ≤ {near_tol}) "
                            f"and far from other axis (dist={d_other} > {far_tol})"
                        )
                    })
                    selected.append(t)
                else:
                    reason = []
                    if not near:
                        reason.append(f"too far from {axis}-axis (dist={d_axis} > {near_tol})")
                    if not far:
                        reason.append(f"too close to other axis (dist={d_other} ≤ {far_tol})")
                    explanations.append({
                        "text": txt, "pixel": p, "selected": False,
                        "reason": "Rejected: " + "; ".join(reason)
                    })

            return selected, explanations

        def gather_x(tokens, x_axis_y, y_axis_x, near_tol=near_tol, far_tol=far_tol):
            return gather_axis_with_explanation(tokens, "x", x_axis_y, y_axis_x, near_tol, far_tol)

        def gather_y(tokens, y_axis_x, x_axis_y, near_tol=near_tol, far_tol=far_tol):
            return gather_axis_with_explanation(tokens, "y", y_axis_x, x_axis_y, near_tol, far_tol)

        raw_x, expl_x = gather_x(tokens_all, x_axis_y, y_axis_x)
        raw_y, expl_y = gather_y(tokens_all, y_axis_x, x_axis_y)
        if False:
            for e in expl_x:
                print(f"x-axis tick '{e['text']}' at {e['pixel']}: {e['reason']}")
        if False:
            for e in expl_y:
                print(f"y-axis tick '{e['text']}' at {e['pixel']}: {e['reason']}")
        # 3b) NEW: valley-split + LLM-aware merge (per axis)
        raw_x = _refine_axis_tokens_with_valleys(img, raw_x, "x", x_llm_ticks or [], llm_value_tol_abs, llm_value_tol_rel)
        raw_y = _refine_axis_tokens_with_valleys(img, raw_y, "y", y_llm_ticks or [], llm_value_tol_abs, llm_value_tol_rel)
        raw_x = [t for t in raw_x if t.get("axis") == "x"]
        raw_y = [t for t in raw_y if t.get("axis") == "y"]
    

        device = "cuda" if torch.cuda.is_available() else "cpu"

        tick_model = TickCNN()
        tick_model.load_state_dict(torch.load("tick_model.pth", map_location=device))
        tick_model.to(device)
        tick_model.eval()

        # Snap OCR ticks to actual axis stubs (closeness + stub length)
        raw_x = [
            t for t in (
                _snap_tick_to_axis(img, tok, "x", x_axis_y, y_axis_x,
                                model=tick_model, device=device)
                for tok in raw_x
            )
            if t is not None
        ]

        raw_y = [
            t for t in (
                _snap_tick_to_axis(img, tok, "y", x_axis_y, y_axis_x,
                                model=tick_model, device=device)
                for tok in raw_y
            )
            if t is not None
        ]

        # 4) LLM ∩ OCR gating (for calibration only)
        explaining_x = False
        explaining_y = False
        if x_mode.startswith("log"):
            log_base_x = float(x_mode[3:])
        else:
            log_base_x = 10
        if y_mode.startswith("log"):
            log_base_y = float(y_mode[3:])
        else:
            log_base_y = 10
        if x_llm_ticks and x_limits_llm:
            raw_x, explain_x = _intersect_with_llm(
                                    raw_x,
                                    x_llm_ticks,
                                    llm_value_tol_abs,
                                    llm_value_tol_rel,
                                    axis_span=x_limits_llm,
                                    pixel_span=x_axis_span,
                                    axis="x",
                                    breaking=x_break,
                                    H=H,
                                    W=W,
                                    mode=x_mode,
                                    log_base=log_base_x
                                )

            if explaining_x:
                print("we are explaining x")
                for e in explain_x:
                    print(e)
        if y_llm_ticks and y_limits_llm:
            raw_y, explain_y = _intersect_with_llm(
                                    raw_y,
                                    y_llm_ticks,
                                    llm_value_tol_abs,
                                    llm_value_tol_rel,
                                    axis_span=y_limits_llm,
                                    pixel_span=y_axis_span,
                                    axis="y",
                                    breaking=y_break,
                                    H=H,
                                    W=W,
                                    mode=y_mode,
                                    log_base=log_base_y
                                )
            if explaining_y:
                print("we are explaining y")
                for e in explain_y:
                    print(e)

        # 5) Calibrate (ticks → AxisCal), then fallbacks if needed
        cal_y = _fit_axis_from_ticks(raw_y, 'y', x_axis_y, y_mode, log_base_y, image=image_path)
        cal_x = _fit_axis_from_ticks(raw_x, 'x', y_axis_x, x_mode, log_base_x, image=image_path)
        # -------------------------------
        # X-axis override calibration
        # -------------------------------
        if cal_x is None and x_ticks_override and len(x_ticks_override) >= 2:
            vals_disp = np.array([v for (v, _p) in x_ticks_override], dtype=float)
            pixs      = np.array([p for (_v, p) in x_ticks_override], dtype=float)

            vals_fit = []
            pixs_fit = []

            for v, p in zip(vals_disp, pixs):
                fv = _to_fit_value(v, x_mode, log_base_x)
                if fv is not None and np.isfinite(fv):
                    vals_fit.append(fv)
                    pixs_fit.append(p)

            if len(vals_fit) >= 2:
                vals_fit = np.asarray(vals_fit, dtype=float)
                pixs_fit = np.asarray(pixs_fit, dtype=float)
                a, b = majority_polyfit(vals_fit, pixs_fit, axis="x")
                if a < 0:
                    a = abs(a)

                cal_x = AxisCal(
                    float(a),
                    float(b),
                    axis="x",
                    mode=x_mode,
                    base=log_base_x
                )
        # -------------------------------
        # Y-axis override calibration
        # -------------------------------
        if cal_y is None and y_ticks_override and len(y_ticks_override) >= 2:
            vals_disp = np.array([v for (v, _p) in y_ticks_override], dtype=float)
            pixs      = np.array([p for (_v, p) in y_ticks_override], dtype=float)

            vals_fit = []
            pixs_fit = []

            for v, p in zip(vals_disp, pixs):
                fv = _to_fit_value(v, y_mode, log_base_y)
                if fv is not None and np.isfinite(fv):
                    vals_fit.append(fv)
                    pixs_fit.append(p)

            if len(vals_fit) >= 2:
                vals_fit = np.asarray(vals_fit, dtype=float)
                pixs_fit = np.asarray(pixs_fit, dtype=float)

                a, b = majority_polyfit(vals_fit, pixs_fit, axis="y")
                if a > 0:
                    a = -abs(a)

                cal_y = AxisCal(
                    float(a),
                    float(b),
                    axis="y",
                    mode=y_mode,
                    base=log_base_y
                )

        # Final fallback: if calibration failed, try to fit a line from the LLM-provided axis limits (if any)
        if cal_x is None:
            print("Warning: x-axis calibration failed")
            cal_x = _fit_axis_from_limits('x', x_limits_override or x_limits_llm, (H,W), x_axis_y, y_axis_x, x_mode)
        if cal_y is None:
            print("Warning: y-axis calibration failed")
            cal_y = _fit_axis_from_limits('y', y_limits_override or y_limits_llm, (H,W), x_axis_y, y_axis_x, y_mode)

        if cal_x is None or cal_y is None:
            missing = []
            if cal_x is None: missing.append('x-axis')
            if cal_y is None: missing.append('y-axis')
            raise ValueError('Cannot calibrate ' + ' & '.join(missing))
        
    # 6) Determine base scale per axis
    step_x, label_k_x = _parse_scale_from_intervals(intervals, 'x')
    step_y, label_k_y = _parse_scale_from_intervals(intervals, 'y')

    x_mode, y_mode = cal_x.mode, cal_y.mode

    def _derive_step_from_calibration(
        cal: AxisCal,
        min_px: int = 10,
    ) -> Optional[float]:
        """
        Derive a step from calibration such that the change corresponds to at least `min_px` pixels.

        Assumes AxisCal encodes:
            pixel = a * fit(value) + b

        Returns:
        - linear: additive step in DISPLAY units
        - log:    multiplicative factor (>1) in DISPLAY units
        """

        if cal is None or cal.a is None:
            return None

        a = float(cal.a)
        if not math.isfinite(a) or abs(a) < 1e-12:
            return None

        # a is px per fit-unit  => delta_fit = min_px / |a|
        delta_fit = float(min_px) / abs(a)
        if not math.isfinite(delta_fit) or delta_fit <= 0:
            return None

        # ----------------------
        # Linear axis (additive)
        # ----------------------
        if cal.mode == "linear":
            return float(delta_fit)

        # ----------------------
        # Log axis (multiplicative)
        # ----------------------
        if isinstance(cal.mode, str) and cal.mode.startswith("log"):
            # Prefer explicit stored base if present
            base = float(cal.mode[3:])
            if not math.isfinite(base) or base <= 0 or base == 1.0:
                return None

            # A min_px pixel move corresponds to a multiplicative factor
            # in display space: value' = value * base**delta_fit
            return float(base ** delta_fit)

        return None

    if step_x is None:
        step_x = _derive_step_from_calibration(cal_x, min_px=10)

    if step_y is None:
        step_y = _derive_step_from_calibration(cal_y, min_px=10)

    if label_k_y is None:
        label_k_y = 5
    if label_k_x is None:
        label_k_x = 5

    # 7) Helpers (local) — ticks-only axis strips + label masking in crops
    def draw_x_axis_linear(
        strip: np.ndarray,
        n_ticks: int,
        safe_margin: int = 5,
    ):
        """
        Draw a linear x-axis with tick indices 1..n_ticks.

        IMPORTANT:
        - Tick indices shown to the LLM are ALWAYS 1..n_ticks
        - Calibration is fit using the TRUE tick indices actually drawn
        (no renumbering if endpoints are clipped)

        Returns:
            tick_px: list of pixel x positions (LOCAL to strip)
            a, b: affine calibration such that pixel_x = a * tick + b
        """
        strip[:] = 255
        H, W = strip.shape[:2]
        ay = 0

        # axis line
        cv2.line(strip, (0, ay), (W - 1, ay), (0, 0, 0), 1)

        if n_ticks <= 0:
            return [], 0.0, 0.0

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.35, 1
        tick_len = 3

        tick_ids = []   # TRUE tick numbers (1..n_ticks)
        tick_px  = []   # LOCAL pixel positions

        for i in range(n_ticks):
            tick = i + 1

            # ideal evenly spaced position
            if n_ticks > 1:
                px = int(round(i / (n_ticks - 1) * (W - 1)))
            else:
                px = W // 2

            if safe_margin <= px < W - safe_margin:
                # draw tick
                cv2.line(strip, (px, ay), (px, ay + tick_len), (0, 0, 0), 1)

                # draw label
                label = str(tick)
                (tw, th_text), _ = cv2.getTextSize(label, font, fs, th)
                x0 = max(1, int(px - tw / 2))
                y0 = min(H - 2, ay + tick_len + 4 + th_text)
                cv2.putText(strip, label, (x0, y0), font, fs, (0, 0, 0), th, cv2.LINE_AA)

                tick_ids.append(tick)
                tick_px.append(px)

        # Fit calibration using TRUE tick indices
        if len(tick_ids) >= 2:
            ticks = np.asarray(tick_ids, dtype=float)
            pxs   = np.asarray(tick_px, dtype=float)
            a, b = np.polyfit(ticks, pxs, 1)
        else:
            a, b = 0.0, 0.0

        return tick_px, float(a), float(b)

    def draw_y_axis_linear(
        strip: np.ndarray,
        n_ticks: int,
        safe_margin: int = 5,
    ):
        """
        Draw a linear y-axis with tick indices 1..n_ticks.

        Convention:
        - tick = 1 is the BOTTOM tick
        - tick increases upward

        Returns:
            tick_px: list of pixel y positions (LOCAL to strip)
            a, b: affine calibration such that pixel_y = a * tick + b
        """
        strip[:] = 255
        H, W = strip.shape[:2]
        ax = W - 1

        # axis line
        cv2.line(strip, (ax, 0), (ax, H - 1), (0, 0, 0), 1)

        if n_ticks <= 0:
            return [], 0.0, 0.0

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.35, 1
        tick_len = 3

        tick_ids = []   # TRUE tick numbers (1..n_ticks)
        tick_px  = []   # LOCAL pixel positions

        for i in range(n_ticks):
            tick = i + 1

            # evenly spaced, bottom -> top
            if n_ticks > 1:
                py = int(round((1.0 - i / (n_ticks - 1)) * (H - 1)))
            else:
                py = H // 2

            if safe_margin <= py < H - safe_margin:
                # draw tick
                cv2.line(strip, (ax - tick_len, py), (ax, py), (0, 0, 0), 1)

                # draw label
                label = str(tick)
                (tw, th_text), _ = cv2.getTextSize(label, font, fs, th)
                x0 = max(2, ax - tick_len - 4 - tw)
                y0 = min(H - 2, py + th_text // 2)
                cv2.putText(strip, label, (x0, y0), font, fs, (0, 0, 0), th, cv2.LINE_AA)

                tick_ids.append(tick)
                tick_px.append(py)

        # Fit calibration using TRUE tick indices
        if len(tick_ids) >= 2:
            ticks = np.asarray(tick_ids, dtype=float)
            pys   = np.asarray(tick_px, dtype=float)
            a, b = np.polyfit(ticks, pys, 1)
        else:
            a, b = 0.0, 0.0
        return tick_px, float(a), float(b)

    def _mask_labels_in_crop(main_crop: np.ndarray, crop_xy0: Tuple[int,int], tokens: List[Dict[str,Any]],
                             fg: float = 0.80, bg: float = 0.20) -> np.ndarray:
        if not tokens: return main_crop
        x0c, y0c = crop_xy0
        overlay = main_crop.copy()
        Hc, Wc = main_crop.shape[:2]
        for t in tokens:
            B = t.get("bbox") or t.get("box") or t.get("rect") or t.get("quad")
            if not (isinstance(B, (list, tuple)) and len(B) == 4): continue

            # flatten polygon if needed
            if isinstance(B[0], (list, tuple)):
                xs = [p[0] for p in B]
                ys = [p[1] for p in B]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
            else:
                x0, y0, x1, y1 = map(float, B)

            # map global bbox → crop-local coords and clip
            x0-=x0c; x1-=x0c; y0-=y0c; y1-=y0c
            x0 = max(0, min(Wc-1, x0)); x1 = max(0, min(Wc-1, x1))
            y0 = max(0, min(Hc-1, y0)); y1 = max(0, min(Hc-1, y1))
            if x1<=x0 or y1<=y0: continue
            cv2.rectangle(overlay, (x0,y0), (x1,y1), (255,255,255), -1)
        return cv2.addWeighted(overlay, fg, main_crop, bg, 0.0)

    # Generate crops per interval 
    out_boxes: List[CropBox] = []
    meta_out: List[Dict[str, Any]] = []

    # plot region implied by axes to image edges
    plot_x0, plot_x1 = y_axis_x, W-1
    plot_y0, plot_y1 = 0, x_axis_y
    
    for k, it in enumerate(intervals):
        iid = str(it.get('id', f'int_{k:02d}'))
        if iid.lower() == 'scale':
            continue

        xval = it.get('x', None)
        yval = it.get('y', None)

        # map value windows to pixel windows, intersect with plot region
        if xval is None:
            print(f"Warning: interval {iid} missing x field")
            xwin = (plot_x0, plot_x1)
        else:
            xv0, xv1 = float(xval[0]), float(xval[1])
            xp0, xp1 = cal_x.v2p(xv0), cal_x.v2p(xv1)
            xwin = (max(plot_x0, int(math.floor(min(xp0, xp1)))),
                    min(plot_x1, int(math.ceil(max(xp0, xp1)))))
            if xwin[1] <= xwin[0]:
                print("X Axis inverted")
                continue

        if yval is None:
            print(f"Warning: interval {iid} missing y field")
            ywin = (plot_y0, plot_y1)
        else:
            yv0, yv1 = float(yval[0]), float(yval[1])
            yp0, yp1 = cal_y.v2p(yv0), cal_y.v2p(yv1)
            ywin = (max(plot_y0, int(math.floor(min(yp0, yp1)))),
                    min(plot_y1, int(math.ceil(max(yp0, yp1)))))
            if ywin[1] <= ywin[0]:
                print("there is the issue")
                print(ywin[0], ywin[1])
                print("Y Axis inverted")
                continue
            
        
        # Prevent absurd spans
        if (xwin[1] - xwin[0]) > 1.5 * W: xwin = (plot_x0, plot_x1)
        if (ywin[1] - ywin[0]) > 1.5 * H: ywin = (plot_y0, plot_y1)

        # main crop with margin + minimum size
        xx0 = max(plot_x0, int(xwin[0]) - margin_px)
        xx1 = min(plot_x1, int(xwin[1]) + margin_px)
        yy0 = max(plot_y0, int(ywin[0]) - margin_px)
        yy1 = min(plot_y1, int(ywin[1]) + margin_px)
        
        main = CropBox(xx0, yy0, xx1, yy1).clamp(W, H)

        # Build labeled & minor ticks from scale with anti-overlap
        y_strip_h = max(1, main.y1 - main.y0)
        x_strip_w = max(1, main.x1 - main.x0)
        # Prepare strips (NO LABEL TEXT: ticks only)
        max_dim = 5000  # cap at ~5k px per side
        y_strip_h = min(y_strip_h, max_dim)
        x_strip_w = min(x_strip_w, max_dim)

        if y_strip_h <= 0 or x_strip_w <= 0 or y_strip_h > max_dim or x_strip_w > max_dim:
            print(f"[skip] {iid}: invalid strip size {x_strip_w}×{y_strip_h}")
            continue

        safe_margin = 10

        ya_x0 = max(0, y_axis_x - y_axis_left_pad)
        ya_x1 = min(W, y_axis_x + 18)
        y_axis_strip = np.full((y_strip_h, max(1, ya_x1 - ya_x0), 3), 255, np.uint8)

        xa_y0 = max(0, x_axis_y - x_axis_above_pad)
        xa_y1 = min(H, x_axis_y + x_axis_below_pad)
        x_axis_strip = np.full((max(1, xa_y1 - xa_y0), x_strip_w, 3), 255, np.uint8)
        
        N_X = 6
        N_Y = 6

        # draw + get calibration
        y_tick_px, a_ty, b_ty = draw_y_axis_linear(y_axis_strip, N_Y)
        x_tick_px, a_tx, b_tx = draw_x_axis_linear(x_axis_strip, N_X)


        # Main crop (from original image)
        main_crop = img[main.y0:main.y1, main.x0:main.x1].copy()

        # MASK printed labels that fell inside the crop (OCR bboxes → crop coords)
        if tokens_all:
            main_crop = _mask_labels_in_crop(main_crop, (main.x0, main.y0), tokens_all, fg=0.80, bg=0.20)

        # Compose L-shape (y-axis strip | main) on top row; x-axis strip on bottom
        spacer_v = np.full((y_axis_strip.shape[0], spacer_px, 3), 255, np.uint8)
        top_row = np.hstack([y_axis_strip, spacer_v, main_crop])

        spacer_h = np.full((spacer_px, top_row.shape[1], 3), 255, np.uint8)
        x_axis_padded = np.hstack([
            np.full((x_axis_strip.shape[0], y_axis_strip.shape[1] + spacer_px, 3), 255, np.uint8),
            x_axis_strip
        ])
        if x_axis_padded.shape[1] < top_row.shape[1]:
            pad = top_row.shape[1] - x_axis_padded.shape[1]
            x_axis_padded = np.hstack([x_axis_padded, np.full((x_axis_padded.shape[0], pad, 3), 255, np.uint8)])
        elif x_axis_padded.shape[1] > top_row.shape[1]:
            pad = x_axis_padded.shape[1] - top_row.shape[1]
            top_row = np.hstack([top_row, np.full((top_row.shape[0], pad, 3), 255, np.uint8)])

        composite = np.vstack([top_row, spacer_h, x_axis_padded])

        out_path = os.path.join(crops_dir, f"ival_{k:02d}.png")
        ok = cv2.imwrite(out_path, composite)        
        out_boxes.append(main)

        meta_out.append({
            "id": iid,
            "image_path": out_path,
            "bbox_main_xyxy": [int(main.x0), int(main.y0), int(main.x1), int(main.y1)],
            "tick_calibration": {
                "x": {
                    "a": a_tx,
                    "b": b_tx,
                    "n_ticks": N_X
                },
                "y": {
                    "a": a_ty,
                    "b": b_ty,
                    "n_ticks": N_Y
                }
            },

            "axis_calibration": {
                "x": {"a": cal_x.a, "b": cal_x.b, "mode": cal_x.mode},
                "y": {"a": cal_y.a, "b": cal_y.b, "mode": cal_y.mode}
            },

            "axes_pixels": {
                "x_axis_y": int(x_axis_y),
                "y_axis_x": int(y_axis_x)
            },
        })

    # 9) Save meta + return calibration bundle
    calibration_info = {
        "x": {"a": cal_x.a, "b": cal_x.b, "mode": cal_x.mode},
        "y": {"a": cal_y.a, "b": cal_y.b, "mode": cal_y.mode},
        "x_axis_y": int(x_axis_y),
        "y_axis_x": int(y_axis_x)
    }

    if save_meta:
        meta_path = os.path.join(out_dir, "interval_crops_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)

    n = len(meta_out)
    if n == 0:
        print(f"Saved 0 interval crops. Output dir: {crops_dir}")
    else:
        #print(f"Saved {n} interval crops to {crops_dir}")
        ex = ", ".join(m["image_path"] for m in meta_out[:min(3,n)])


    return IntervalCropResult(out_boxes, calibration_info, meta_out)
