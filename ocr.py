"""
ocr_boost.py — Enhance step 1 with EasyOCR + per-plot filtering + pixel-aware outputs

What this provides
------------------
1) Robust OCR of chart text with EasyOCR (ticks, axis labels, titles, legend entries).
2) Automatic plot-region discovery (handles multi-plot figures / subplots).
3) Per-plot OCR pass (crop-based) to reduce noise and let you process one plot at a time.
4) Structured outputs with NUMERIC tick labels mapped to pixel coordinates.
5) A compact, downsampled edge "vibe grid" (e.g., 32×32 heatmap) that captures the overall
   shape/evolution of the plot without doing curve extraction. You can pass this grid to an LLM
   to help it reason about trends/shapes while keeping tokens low.

Notes
-----
- This module is a drop-in helper for step 1 of your 2-step approach. You can call it before your
  LLM extraction step to provide pixel-calibrated context and per-plot isolation.
- The plot detection is heuristic (edge density + rectangular contour search). It works well on
  clean exported figures; noisy scans may need parameter tuning.
- EasyOCR is used in English by default; add languages if needed.

Dependencies
------------
    pip install easyocr opencv-python-headless numpy scikit-image

Example
-------
    from ocr_boost import process_image
    result = process_image("/path/to/figure.png", per_plot_ocr=True)
    # result is a dict you can JSON-serialize and feed into your pipeline / LLM

"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects

try:
    import easyocr  # type: ignore
except Exception as e:
    raise ImportError(
        "EasyOCR is required. Install with `pip install easyocr`. Original error: %r" % (e,)
    )

# -----------------------------
# Data containers
# -----------------------------
@dataclass
class TextDetection:
    text: str
    conf: float
    box: List[Tuple[int, int]]  # 4 points [(x1,y1),...]
    center: Tuple[float, float]

@dataclass
class TickLabel:
    value: float
    text: str
    pixel: Tuple[float, float]
    axis: str  # "x" or "y" or "unknown"
    box: Optional[List[Tuple[int, int]]]  # optional box

@dataclass
class PlotRegion:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    ticks: List[TickLabel]
    labels: List[TextDetection]  # axis/legend/title (raw OCR chunks)
    legend_texts: List[TextDetection]

# -----------------------------
# OCR helpers
# -----------------------------

def _to_num_tokens(txt: str) -> List[str]:
    """Extract numeric-like tokens. Handles integers, floats, sci-notation, and simple ranges."""
    t = txt.replace("–", "-").replace("—", "-")
    return re.findall(r"-?\d+(?:[\.,]\d+)?(?:e[+-]?\d+)?", t, flags=re.I)


def _center_of_box(box: List[Tuple[int, int]]) -> Tuple[float, float]:
    """Center of a 4-point box."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (float(sum(xs)) / 4.0, float(sum(ys)) / 4.0)


def run_easyocr(img_bgr: np.ndarray, lang: List[str], enhance: bool = True, allowlist: str | None = None, scale_up: float = 1.5) -> List[TextDetection]:
    """Run EasyOCR and return structured detections.

    Optional:
      - enhance: apply CLAHE
      - allowlist: restrict characters (e.g., digits for tick OCR)
      - scale_up: resize before OCR to help tiny text
    """
    proc = img_bgr.copy()
    if enhance:
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        proc = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    if scale_up and scale_up != 1.0:
        proc = cv2.resize(proc, None, fx=scale_up, fy=scale_up, interpolation=cv2.INTER_CUBIC)

    reader = easyocr.Reader(lang, gpu=True)  # set gpu=False if needed
    results = reader.readtext(proc[:, :, ::-1], allowlist=allowlist)  # expects RGB

    out: List[TextDetection] = []
    for box, text, conf in results:  # box is 4 points
        try:
            text_clean = str(text).strip()
            if not text_clean:
                continue
            # Reverse the scale if we resized
            poly = []
            for (x, y) in box:
                if scale_up and scale_up != 1.0:
                    x = int(round(x / scale_up))
                    y = int(round(y / scale_up))
                poly.append((int(x), int(y)))
            out.append(
                TextDetection(
                    text=text_clean,
                    conf=float(conf),
                    box=poly,
                    center=_center_of_box(poly),
                )
            )
        except Exception:
            continue
    return out



def _digits_margin_pass(
    img_bgr: np.ndarray,
    lang: List[str],
    conf_min: float = 0.2,
    scale_up: float = 3.0,
    band_frac: float = 0.15,
    x_axis_y: int | None = None,
    y_axis_x: int | None = None,
    x_rotation: float | None = None,
    y_rotation: float | None = None,
) -> List[TickLabel]:
    """
    OCR for numeric ticks around axes.

    - For the y-axis: crop a vertical strip around y_axis_x and run EasyOCR 3 times:
      0°, +90° (clockwise), and -90° (counter-clockwise). Map coords back to full image.
    - For the x-axis: crop a horizontal strip around x_axis_y (as before).
    - If an axis anchor is missing, fall back to legacy left/bottom margins.

    Returns a list[TickLabel].
    """
    H, W = img_bgr.shape[:2]
    ticks: List[TickLabel] = []

    # ---------- helpers ----------
    def _is_good_num(td) -> Optional[float]:
        if td.conf < conf_min:
            return None
        nums = _to_num_tokens(td.text)
        if not nums:
            return None
        try:
            return float(nums[0].replace(",", "."))
        except Exception:
            return None

    def _dedup_ticks(items: List[TickLabel], px_tol: int = 8, val_tol: float = 1e-6) -> List[TickLabel]:
        out: List[TickLabel] = []
        for t in items:
            x, y = t.pixel
            keep = True
            for u in out:
                ux, uy = u.pixel
                if abs(t.value - u.value) <= val_tol and (abs(x-ux) + abs(y-uy) <= px_tol):
                    keep = False
                    break
            if keep:
                out.append(t)
        return out

    # Map rotated ROI centers back to the original crop (and then to full image).
    def _map_box_from_rot(poly_rot: List[Tuple[int,int]], rot_code: int, orig_h: int, orig_w: int) -> List[Tuple[int,int]]:
        mapped = []
        for (u, v) in poly_rot:
            if rot_code == cv2.ROTATE_90_CLOCKWISE:
                # inverse of (u = h-1 - y, v = x)  ->  (x = v, y = h-1 - u)
                x = v
                y = (orig_h - 1) - u
            elif rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
                # inverse of (u = y, v = w-1 - x)  ->  (x = w-1 - v, y = u)
                x = (orig_w - 1) - v
                y = u
            else:
                x, y = u, v
            mapped.append((int(x), int(y)))
        return mapped
    
    def _map_center_from_rot(center, rot_code, orig_h, orig_w):
        x_rot, y_rot = center  # EasyOCR centers are (x, y)
        if rot_code == cv2.ROTATE_90_CLOCKWISE:
            # same mapping as _map_box_from_rot for CLOCKWISE
            x = y_rot
            y = (orig_h - 1) - x_rot
        elif rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
            # same mapping as _map_box_from_rot for COUNTERCLOCKWISE
            x = (orig_w - 1) - y_rot
            y = x_rot
        else:
            x, y = x_rot, y_rot
        return float(x), float(y)

    def _run_ocr_on_roi_with_rotations(roi_bgr: np.ndarray, axis: str, x0_off: int, y0_off: int) -> List[TickLabel]:

        h0, w0 = roi_bgr.shape[:2]

        def _ocr0():
            dets = run_easyocr(roi_bgr, lang, enhance=True,
                            allowlist="0123456789.-", scale_up=scale_up)
            out = []
            for td in dets:
                val = _is_good_num(td)
                if val is None:
                    continue
                cx, cy = td.center
                box0 = [(px + x0_off, py + y0_off) for (px, py) in td.box]
                out.append(TickLabel(val, td.text,
                                    (cx + x0_off, cy + y0_off),
                                    axis=axis, box=box0))
            return out

        def _ocr_cw():
            roi_cw = cv2.rotate(roi_bgr, cv2.ROTATE_90_CLOCKWISE)
            dets = run_easyocr(roi_cw, lang, enhance=True,
                            allowlist="0123456789.-", scale_up=scale_up)
            out = []
            for td in dets:
                val = _is_good_num(td)
                if val is None:
                    continue
                u, v = td.center
                x_local, y_local = _map_center_from_rot((u, v),
                                                        cv2.ROTATE_90_CLOCKWISE,
                                                        h0, w0)
                box_local = _map_box_from_rot(td.box,
                                            cv2.ROTATE_90_CLOCKWISE,
                                            h0, w0)
                box_full = [(px + x0_off, py + y0_off)
                            for (px, py) in box_local]
                out.append(TickLabel(val, td.text,
                                    (x_local + x0_off,
                                    y_local + y0_off),
                                    axis=axis, box=box_full))
            return out

        def _ocr_ccw():
            roi_ccw = cv2.rotate(roi_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            dets = run_easyocr(roi_ccw, lang, enhance=True,
                            allowlist="0123456789.-", scale_up=scale_up)
            out = []
            for td in dets:
                val = _is_good_num(td)
                if val is None:
                    continue
                u, v = td.center
                x_local, y_local = _map_center_from_rot((u, v),
                                                        cv2.ROTATE_90_COUNTERCLOCKWISE,
                                                        h0, w0)
                box_local = _map_box_from_rot(td.box,
                                            cv2.ROTATE_90_COUNTERCLOCKWISE,
                                            h0, w0)
                box_full = [(px + x0_off, py + y0_off)
                            for (px, py) in box_local]
                out.append(TickLabel(val, td.text,
                                    (x_local + x0_off,
                                    y_local + y0_off),
                                    axis=axis, box=box_full))
            return out

        with ThreadPoolExecutor(max_workers=3) as ex:
            f0  = ex.submit(_ocr0)
            f90 = ex.submit(_ocr_cw)
            f_90 = ex.submit(_ocr_ccw)

            out0  = f0.result()
            out90 = f90.result()
            out_90 = f_90.result()


        all_results = out0 + out90 + out_90

        return _dedup_ticks(all_results, px_tol=8, val_tol=1e-6)

    # ---------- Y-axis strip (vertical numbers) ----------
    if y_axis_x is not None:
        # Asymmetric band: more to the left (outside) than to the right (inside plot)
        band_w_left  = int(band_frac * W)            # outside the plot
        band_w_right = int(band_frac * W * 0.5)      # slightly into the plot
        x0 = max(0, y_axis_x - band_w_left)
        x1 = min(W, y_axis_x + band_w_right)
        y_roi = img_bgr[:, x0:x1]
        ticks.extend(_run_ocr_on_roi_with_rotations(y_roi, axis="y", x0_off=x0, y0_off=0))
    else:
        # Legacy left margin
        left_w = int(band_frac * W)
        x0, x1 = 0, left_w
        y_roi = img_bgr[:, x0:x1]
        ticks.extend(_run_ocr_on_roi_with_rotations(y_roi, axis="y", x0_off=x0, y0_off=0))

    # ---------- X-axis strip (horizontal numbers) ----------
    if x_axis_y is not None:
        # Asymmetric band: more below (outside) than above (into the plot)
        band_h_below = int(band_frac * H)
        band_h_above = int(band_frac * H * 0.5)
        y0 = max(0, x_axis_y - band_h_above)
        y1 = min(H, x_axis_y + band_h_below)
        x_roi = img_bgr[y0:y1, :]
        # One orientation is enough for x-axis (labels are horizontal), but keep consistent call:
        ticks.extend(_run_ocr_on_roi_with_rotations(x_roi, axis="x", x0_off=0, y0_off=y0))
    else:
        # Legacy bottom margin
        bot_h = int(band_frac * H)
        y0, y1 = H - bot_h, H
        x_roi = img_bgr[y0:y1, :]
        ticks.extend(_run_ocr_on_roi_with_rotations(x_roi, axis="x", x0_off=0, y0_off=y0))

    return ticks


def debug_visualize(image_path: str, result: Dict[str, Any], out_path: str = "ocr_debug_overlay.png") -> str:
    """Draw plot boxes, tick centers, and OCR text on top of the image for sanity checks.
    Returns the output path.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, p in enumerate(result.get("plots", [])):
        x, y, w, h = p["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"plot {i}", (x, max(0, y - 5)), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # ticks
        for t in p.get("ticks", []):
            cx, cy = map(int, t["pixel"])
            cv2.circle(img, (cx, cy), 3, (255, 0, 0), -1)
            cv2.putText(img, f"{t['text']}({t['axis']})", (cx + 4, cy - 4), font, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
        # label centers
        for td in p.get("labels", []):
            cx, cy = map(int, td["center"])
            cv2.circle(img, (cx, cy), 2, (0, 165, 255), -1)
    cv2.imwrite(out_path, img)
    return out_path

def process_image(
    image_path: str,
    lang: List[str] | None = None,
    per_plot_ocr: bool = True,
    numeric_conf_min: float = 0.3,
    min_plot_area: int = 12_000,
    suppress_border: bool = True,
    border_frac: float = 0.05,
    ocr_enhance: bool = True,
    ocr_outer_pad_frac: float = 0.08,
    tick_inclusion_pad_frac: float = 0.12,
    digits_fallback: bool = True,
    margins_only: bool = True,
    x_axis_y: int | None = None,
    y_axis_x: int | None = None,
    x_rotation: float | None = None,
    y_rotation: float | None = None,
) -> Dict[str, Any]:
    """
    Full pipeline for step 1 : extracting all possible ticks.

    Only scan strips around the given x_axis_y and y_axis_x
      to extract numeric tick labels.

    Returns a JSON-serializable dict.
    """
    if lang is None:
        lang = ["en"]

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img_bgr.shape[:2]

    plots_json: List[Dict[str, Any]] = []

    if x_axis_y is None or y_axis_x is None:
        raise ValueError("margins_only=True requires x_axis_y and y_axis_x")

    ticks = _digits_margin_pass(
        img_bgr, lang,
        conf_min=numeric_conf_min,
        x_axis_y=x_axis_y,
        y_axis_x=y_axis_x,
        x_rotation=x_rotation,
        y_rotation=y_rotation
    )
    # No labels/legends in margins-only mode
    labels: List[str] = []
    legends: List[str] = []
    full_box = (0, 0, W, H)
    plots_json.append(
        asdict(
            PlotRegion(
                bbox=full_box,
                ticks=ticks,
                labels=labels,
                legend_texts=legends,
            )
        )
    )

    out = {
        "image_path": image_path,
        "image_size": {"width": int(W), "height": int(H)},
        "plots": plots_json,
        "meta": {
            "lang": lang,
            "per_plot_ocr": per_plot_ocr,
            "margins_only": margins_only,
        },
    }
    
    return out


def save_json(result: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EasyOCR-enhanced step 1 for chart pipeline")
    ap.add_argument("image", help="Path to input chart image")
    ap.add_argument("--lang", nargs="*", default=["en"], help="EasyOCR language codes")
    ap.add_argument("--grid", type=int, default=32, help="Vibe grid size (e.g., 32)")
    ap.add_argument("--no-per-plot", action="store_true", help="Disable per-plot OCR (use global)")
    ap.add_argument("--min-area", type=int, default=12000, help="Minimum plot area in pixels")
    ap.add_argument("--conf", type=float, default=0.3, help="Minimum OCR confidence for numeric ticks")
    ap.add_argument("--no-border-suppress", action="store_true", help="Do not suppress frame edges in vibe grid")
    ap.add_argument("--border-frac", type=float, default=0.05, help="Border band fraction to suppress (0..0.2)")
    ap.add_argument("--no-ocr-enhance", action="store_true", help="Disable CLAHE-based OCR enhancement")
    ap.add_argument("--outer-pad-frac", type=float, default=0.08, help="OCR crop padding as fraction of plot box")
    ap.add_argument("--tick-pad-frac", type=float, default=0.12, help="Pad when assigning OCR to plot (fraction of max(w,h))")
    ap.add_argument("--no-digits-fallback", action="store_true", help="Disable digits-only margin OCR fallback")
    ap.add_argument("--out", default="ocr_step.json", help="Output JSON path")

    args = ap.parse_args()

    res = process_image(
        args.image,
        lang=args.lang,
        per_plot_ocr=not args.no_per_plot,
        numeric_conf_min=args.conf,
        min_plot_area=args.min_area,
        suppress_border=not args.no_border_suppress,
        border_frac=args.border_frac,
        ocr_enhance=not args.no_ocr_enhance,
        ocr_outer_pad_frac=args.outer_pad_frac,
        tick_inclusion_pad_frac=args.tick_pad_frac,
        digits_fallback=not args.no_digits_fallback,
    )
    save_json(res, args.out)
    print(f"Saved: {args.out}")
