import os
import json
import csv
import math
import re
import threading
import queue
import base64
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2
import concurrent.futures
import time
import shutil
import warnings
from pathlib import Path
from scipy.ndimage import median_filter
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import json
import pandas as pd

import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import numpy as np
import time as _time
from google.genai import types
from time import time
from concurrent.futures import ThreadPoolExecutor
import threading
import os
import cv2
from pathlib import Path

from typing import List, Dict, Any, Optional
from interval_crops import make_interval_crops
import shutil
from pipeline_core import run_full_pipeline_async, run_step2
from gemini_simple import extract_plot_data_gemini_streaming
load_dotenv()

from ocr import process_image # For generating initial ticks/meta
from extraction_tools import _detect_axes, extract_non_background_mask
from interval_crops import AxisCal
# Placeholder for LLM configuration, assuming Google GenAI is available
try:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
except ImportError:
    warnings.warn("Google GenAI client not installed. LLM functionality will be disabled.", ImportWarning)
    # Define Mock classes/client here for file integrity

# --- GLOBAL LLM PROMPTS AND TOOL DEFINITIONS ---

# The LLM must be directed to produce these structures via prompt engineering
# (The LLM prompt bodies are omitted for brevity but required for actual use)
UNIFIED_USER_PROMPT = "Analyze the image and the current state to perform the next mandatory extraction step."
UNIFIED_SYSTEM_PROMPT = """
You are a high-level plot extraction router and analyst.

Your job is to SELECT THE EXTRACTION PIPELINE based solely on the chart’s
visual structure. You do NOT perform calibration and you do NOT reason
about axis fitting.

### Pipeline Selection Rules (STRICT):
1. **Path A - Categorical Extraction
    - Bars or mixed structures
    - cateogrical information
    - each point is about an interval, so the axis isn't linear anymore
   - Action:
     - Call `categorical_tool`
        
2. **Path B — Kaplan–Meier / Survival / Stepwise Curves (HIGHEST PRIORITY)**
   - If the chart visually resembles ANY of the following:
     - Kaplan–Meier survival curves
     - Cumulative incidence plots
     - Monotonic stepwise lines (upward or downward)
     - Survival probability / cumulative incidence on y-axis
     - Censor marks or step drops
   - You MUST choose Path B.
   - Actions: 
     - call 'km_detection_tool'

3. **Path C — Generic Numeric Extraction (Fallback)**
   - Use ONLY if the plot is clearly does NOT belong to one of the other path.
   - Examples:
     - Scatter plots
     - Non-monotonic multi-line charts
   - Action:
     - Call `crop_generation_tool`

### Output Rules:
- Output ONE and ONLY one tool call.
- No commentary.
- No markdown.
"""

CALIBRATION_SYSTEM_PROMPT = """
You are an axis calibration specialist.

Your sole task is to infer accurate axis properties from the chart image.

You do NOT select pipelines.
You do NOT extract data points.
You do NOT reason about chart semantics beyond axis meaning.

### Your task:
- Identify X and Y axis properties
- Determine:
  - axis type (linear, logK with K being an integerer so 10, 2, 5, ...)
  - visible tick values
  - approximate numeric range
  - presence of breaks or discontinuities (if any)

### Axis breaks:
- If an axis has a break, describe breaks using a separate `breaks` field.
- Each break must specify the numeric values it separates.
    Example:
    {
    "ticks": [0, 10, 50, 100],
    "breaks": [[10, 50], [50, 100]]
    }
    
### Rules:
- Always call `axis_properties_tool`.
- Provide the most complete AXIS_INFO possible.
- Do NOT output commentary.
- Do NOT repeat calibration after it has been done.

"""

CALIBRATION_USER_PROMPT = """
Analyze the chart image and extract full axis calibration information.
Infer axis ticks, and numeric range as precisely as possible.
"""

categorical_prompt = """You are extracting data from a CATEGORICAL GRAPH.
The graph contains discrete categories.
Each category has exactly one numeric value per series
(e.g. bar height or dot position).

You MUST:
- Identify each category
- Extract one numeric value per (category, series)
- Use ONLY the provided series ids
- Extract error bars if present

You MUST NOT:
- interpolate values
- treat this as a continuous curve
- invent categories or series

Return ONLY valid JSON following this schema:
{
  "data": [
    {
      "category": "string",
      "series_id": "string",
      "value": number,
      "error_low": number | null,
      "error_high": number | null
    }
  ]
}
"""
# Helper to create Gemini Tool definitions (as per previous response)
def create_function_declaration(tool_dict):
    return types.Tool(function_declarations=[types.FunctionDeclaration(name=tool_dict["name"], description=tool_dict["description"], parameters=types.Schema(**tool_dict["parameters"]))])

# 1. Axis Calibration (A) - Uses imported axis_properties_tool logic
axis_properties_tool_def = {
    "name": "axis_properties_tool",
    "description":  "Store axis properties and produce calibrated AxisCal objects.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "axis_info": {
                "type": "OBJECT",
                "properties": {
                    "x_axis": {"type": "OBJECT", "properties": {"type": {"type": "STRING"}, "ticks": {"type": "ARRAY", "items": {"type": "NUMBER"}}, "break": {"type": "ARRAY", "items": {"type": "NUMBER"}}}, "required": ["type", "ticks"]}, 
                    "y_axis": {"type": "OBJECT", "properties": {"type": {"type": "STRING"}, "ticks": {"type": "ARRAY", "items": {"type": "NUMBER"}}, "break": {"type": "ARRAY", "items": {"type": "NUMBER"}}}, "required": ["type", "ticks"]},
                },
                "required": ["x_axis", "y_axis"],
            }
        },
        "required": ["axis_info"]
    }
}

categorical_tool_description = """
Use this tool to extract data from CATEGORICAL GRAPHS.

A categorical graph has a finite set of discrete categories
(e.g. bars, dots, or markers), where each category has
one representative numeric value per series.

This tool is appropriate when:
- the x-axis represents named categories, not a continuous scale
- each category has a single bar, dot, or marker per series
- there is no curve continuity between categories

Do NOT use this tool for continuous line plots or curves.

The tool extracts one value per (category, series) pair,
optionally with error bars.
"""

crop_gen_description = """
The crop_generation_tool generates focused image crops of a plot to support
series-specific extraction in downstream steps.

Key constraints:
- The tool does NOT perform axis detection or axis calibration.
- The tool does NOT extract numeric values.
- The tool DOES generate spatial crops (image regions) that are likely to contain:
  (a) visually ambiguous series interactions (overlaps, crossings),
  (b) important geometric events (starts, sharp turns, peaks, valleys, knees, drops),
  (c) representative local segments that help confirm series identity.
- The tool relies on your generated series descriptions (series_info) so that each
  crop can be interpreted as belonging to a specific series.

Inputs:
- image_path: path to the full plot image.

- series_info: list of visual series descriptions. The goal is to recognize,
  without ambiguity and using only local visual cues, which series a curve belongs to.
  Descriptions must focus on local appearance, not global trends.

Each series must include:
    id: a unique identifier (e.g., "s1", "s2")
    name: a clear, human-readable name
    color: detailed color description (hue, brightness, saturation, gradients if any), and a comparison explaining how this color differs from other series
    line_style: line pattern (solid, dashed, dotted, step-like, etc.), including cadence or regularity, and an explanation of how this pattern distinguishes it from similar series
    marker: marker presence and shape (circle, square, triangle, cross, etc.), size if relevant, and explicit comparison to other markers
    width: relative stroke thickness, including whether it is noticeably thicker or thinner than others
    disambiguation_notes: a short reasoning section explaining how to identify this series when multiple series appear similar (e.g., “This is not s2 because s2 uses a dashed line and square markers, whereas this series uses a solid line with circular markers.”)

- intervals: REQUIRED. MUST be non-empty.
  Intervals define the EXCLUSIVE regions of the image to be cropped.
  If intervals are missing, empty, or malformed, downstream extraction WILL FAIL.

  Interval format:
  - Each interval is a rectangular crop defined by:
      • x: [min, max] numeric range along the horizontal axis
      • y: [min, max] numeric range along the vertical axis
  - Ranges MUST be numeric, finite, and local to the data shown in that crop.
  - Full-axis default ranges are not valid unless explicitly justified by the data.

  Interval selection rules:
  - Intervals MUST collectively cover the entire horizontal extent of the plot,
    from the start to the final event. Do not stop early.
  - Intervals MUST NOT use uniform spacing along X.
    • Use narrow X ranges where series change rapidly, overlap, or cross.
    • Use wider X ranges for long, stable plateau regions.
  - You have a maximum budget of 6 intervals, and you must use atleast 3. Allocate them where visual
    ambiguity or complexity is highest, don't hesitate to several interval for a single x range (looking at different y values).
  - Intervals should ideally be contiguous along X. Small gaps are allowed only
    in simple plateau regions with no visible change.
  - X and Y resolution are independent. Increasing Y resolution MUST NOT introduce
    additional X splits.

Outputs:
- crops: list of image crops (sub-images)
- crops_meta: metadata describing crop positions in the original image.
  Each entry should include a series_id, or a list of candidate series_ids
  when ambiguity exists.

Guarantees:
- Crops preserve enough local visual context to distinguish series according
  to series_info.
- Interval selection is intended to maximize downstream point-extraction
  reliability, not to uniformly tile or exhaustively cover the plot.
"""

crop_generation_tool_def = {
    "name": "crop_generation_tool",
    "description": crop_gen_description,
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the full plot image."
            },
            "description": {
                "type": "string",
                "description": "High-level visual description of the plot."
            },
            "series_info": {
                "type": "array",
                "description": "Visual series descriptors for disambiguation within crops.",
                "items": {
                    "type": "string",
                }
            },
            "intervals": {
                "type": "array",
                "description": "Spatial targeting hints for crop generation.",
                "items": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "y": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        }
                    },
                    "required": ["x", "y"]
                }
            },
        },
        "required": ["image_path", "intervals", "series_info", "description"]
    }
}

km_description = """The km_detection_tool extracts Kaplan–Meier (survival-style)
curves from plots where ALL series evolve continuously over the x-axis
(time, months, years, follow-up).

This tool MUST be used whenever:
- all series are continuous or stepwise-continuous,
- curves represent survival, cumulative incidence, or progression over time,
- multiple series start at a common baseline and diverge gradually or in steps.

This includes standard Kaplan–Meier plots, cumulative survival plots,
and any medical time-to-event curves with step-like geometry.

The tool is the DEFAULT extraction mechanism for continuous multi-series plots.

--------------------------------------------------------------------
INTERNAL BEHAVIOR (for clarity — not controlled by the language model):

1) Optional internal guidance extraction:
   - The tool may internally extract sparse, approximate guidance paths
     (e.g. via a language model or exploratory logic).
   - Guidance is used ONLY to bias curve following, never as final output.

2) Deterministic image-based curve snapping:
   - Final curves are snapped directly to image pixels.
   - Each series is tracked independently.
   - Guidance NEVER causes series to merge or cross.

--------------------------------------------------------------------
CRITICAL CONSTRAINTS (MUST FOLLOW):

- The language model MUST call km_detection_tool whenever all series are continuous.
- The language model MUST NOT attempt to extract final points itself.
- The language model MUST NOT call crop_generation_tool separately.
- The language model MUST provide series_info, intervals, and scale
  exactly as required (same rules as crop_generation_tool).
- Series identities (id + name) MUST remain stable throughout.

--------------------------------------------------------------------
KAPLAN–MEIER–SPECIFIC ASSUMPTIONS:

- Curves are monotonic and piecewise constant or piecewise monotonic.
- Series may overlap, cross visually, or share long common prefixes.
- All series typically start at the same or similar y-value.
- Curves may terminate early (censoring).

--------------------------------------------------------------------
ROLE OF THE LANGUAGE MODEL:

The language model is responsible ONLY for:
- Declaring that the plot contains continuous survival-like series.
- Listing ALL series with stable ids and local visual descriptors.
- Providing correct intervals and scale metadata.
- Specifying starting_y_value and direction.

The language model MUST NOT:
- invent series,
- merge series,
- emit pixel coordinates,
- emit final data points,
- or attempt to simplify continuous plots into categorical data.

--------------------------------------------------------------------
GOAL:

Whenever the plot contains continuous or stepwise-continuous series,
this tool MUST be called to ensure robust, pixel-accurate extraction.
"""

km_detection_tool_def = {
    "name": "km_detection_tool",
    "description": km_description,
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "image_path": {"type": "string"},
            "description": {"type": "string"},

            # ---------------- GLOBAL KM PARAMETERS ----------------
            "starting_y_value": {
                "type": "NUMBER",
                "description": (
                    "Expected starting value of the Kaplan–Meier curves "
                    "(typically 1.0 for survival probability)."
                )
            },
            "direction": {
                "type": "string",
                "enum": ["downward", "upward"],
                "description": (
                    "Direction of survival evolution. "
                    "'downward' means survival decreases over time "
                    "(standard Kaplan–Meier). "
                    "'upward' means inverted survival plots."
                )
            },
            "max_series": {
                "type": "number",
                "description": (
                    "Maximum number of Kaplan–Meier curves expected in the plot. "
                    "Used as a safety bound during exploration and tracking."
                )
            },

            # ---------------- SERIES DESCRIPTORS ----------------
            "series_info": {
                "type": "array",
                "description": (
                    "Visual series descriptors for Kaplan–Meier curves. "
                    "These descriptors are used BOTH for crop-based exploration "
                    "and for bridging exploratory guidance to final tracking."
                ),
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique identifier for the series (e.g. 's1')."
                        },
                        "name": {
                            "type": "string",
                            "description": "Human-readable name of the series."
                        },

                        # KM-specific color (RESTRICTED ENUM)
                        "color": {
                            "type": "string",
                            "enum": [
                                "none",
                                "red",
                                "green",
                                "blue",
                                "yellow",
                                "magenta",
                                "cyan",
                                "orange",
                                "purple",
                                "pink",
                                "brown",
                                "olive",
                                "teal",
                                "navy",
                                "maroon",
                                "gold"
                            ],
                            "description": (
                                "The visual color of the Kaplan–Meier curve. "
                                "Do NOT invent colors. "
                                "If the curve is grey, black, dark, or visually "
                                "ambiguous with other series, use 'none'."
                            )
                        },

                        # SAME LOCAL VISUAL DESCRIPTORS AS CROP TOOL
                        "line_style": {
                            "type": "string",
                            "description": (
                                "Description of the line pattern (solid, dashed, step-like, etc.) "
                                "and what makes it different from other series, if applicable."
                            )
                        },
                        "marker": {
                            "type": "string",
                            "description": (
                                "Marker shape if present (cross, square, circle, etc.) "
                                "and what makes it different from other series, if applicable."
                            )
                        },
                        "width": {
                            "type": "string",
                            "description": (
                                "Relative stroke thickness and what makes it different "
                                "from other series, if applicable."
                            )
                        }
                    },
                    "required": ["id", "name", "color", "line_style", "marker", "width"]
                }
            },
        },
        # ---------------- REQUIRED FIELDS ----------------
        "required": [
            "image_path",
            "description",
            "series_info",
            "starting_y_value",
            "direction",
            "max_series"
        ]
    }
}

categorical_tool_schema = {
  "type": "object",
  "properties": {
    "graph_type": {
      "type": "string",
      "description": "The type of categorical graph it is, bar chart, grouped bar, ...",
    },
  },
  "required": ["graph_type"]
}
categorical_tool_def = {
    "name": "categorical_tool", "description": categorical_tool_description,
    "parameters": categorical_tool_schema
}

UNIFIED_GEMINI_TOOLS = [
    create_function_declaration(crop_generation_tool_def),
    create_function_declaration(categorical_tool_def),
    create_function_declaration(km_detection_tool_def),  
]


def categorical_llm_output_to_canonical(
    llm_obj: dict,
    category_to_x: dict | None = None,
):
    """
    Convert categorical extraction into benchmark-canonical series format.

    Each category becomes its own series with >=1 point.
    """

    series_out = []

    for item in llm_obj["data"]:
        cat = item["category"]
        y = float(item["value"])

        # Numeric x (preferred: provided or inferred)
        if category_to_x and cat in category_to_x:
            x = float(category_to_x[cat])
        else:
            # Fallback: parse midpoint from label like "21 – 30"
            nums = [float(s) for s in re.findall(r"\d+\.?\d*", cat)]
            x = sum(nums) / len(nums) if nums else float("nan")

        points = [{"x": x, "y": y}]

        # Optional error bars → extra points at same x
        if "error_low" in item and item["error_low"] is not None:
            points.append({"x": float(x), "y": float(item["error_low"])})
        if "error_high" in item and item["error_high"] is not None:
            points.append({"x": float(x), "y": float(item["error_high"])})

        series_out.append({
            "label": cat,
            "points": points,
        })

    return {"series": series_out}

def categorical_graph_extraction_tool(
    image_path: str,
    model: str = "gemini-3-pro-preview"
):
    prompt = categorical_prompt

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _img_to_data_url(image_path)}},
        ]
    }]
    response = gemini_chat(model, messages=messages, is_json=True)
    raw_stripped = response.strip()
    try:
        return categorical_llm_output_to_canonical(json.loads(raw_stripped))
    except Exception:
        return _balanced_json_extract(raw_stripped)



def axis_properties_tool(ocr_results: Dict[str, Any], image_path: str, axis_info: dict, out_dir: str = "out_interval") -> dict:
    """
    1) Store AXIS_INFO (including rotation_deg for both axes).
    2) Build calibration exactly like your pipeline:
       - Pass llm_axis_info into make_interval_crops() to calibrate.
       - Use a minimal 'intervals' list containing only {"id":"scale"} so no crops are forced.
    Returns: {"ok": True, "axis_info":..., "calibration": {...}}
    """
    # 0) Persist axis info
    global last_axis_info, last_calibration, last_crops_obj
    last_axis_info = axis_info or {}

    intervals = [{"id": "scale"}]
    print("LLM axis information is ", axis_info)
    y_axis = axis_info["y_axis"]
    x_axis = axis_info["x_axis"]
    x_axis = normalize_axis_if_needed(x_axis)
    y_axis = normalize_axis_if_needed(y_axis)

    files, crops = run_step2(
        chart_path=image_path,
        intervals=intervals,
        out_dir=out_dir,
        llm_axis_info=axis_info,
        res=ocr_results
    )
    # --- Execute Calibration (Logic Fusion) ---
    x, y, a, b = crops.calibration.items()
    cal_x, cal_y, x_axis_y, y_axis_x = x[1], y[1], a[1], b[1]
    cal_x = AxisCal(a=cal_x.get("a"), b=cal_x.get("b"), mode=cal_x.get("mode"), axis="x")
    cal_y = AxisCal(a=cal_y.get("a"), b=cal_y.get("b"), mode=cal_y.get("mode"), axis="y")
    print(f"Calibration results: X-axis: a={cal_x.a}, b={cal_x.b}, mode={cal_x.mode}; Y-axis: a={cal_y.a}, b={cal_y.b}, mode={cal_y.mode}")
    # 3. Update Global State with Calibration
    GLOBAL_CALIBRATION_RESULT["x_cal"] = cal_x
    GLOBAL_CALIBRATION_RESULT["y_cal"] = cal_y
    
    return cal_x, cal_y # Return the calculated AxisCal objects

def expand_duplicate_x_with_continuity_final(
    result: dict,
    eps_frac: float = 0.01,
) -> dict:
    """
    Post-process final extractor output to:
      - keep all points
      - make x strictly increasing
      - resolve duplicate x using continuity with previous y
      - encode vertical steps as x + k*epsilon

    This is SAFE to run at the very end of the pipeline.
    """

    if not result or "series" not in result:
        return result

    out_series = []

    for series in result["series"]:
        pts = series.get("points", [])
        if len(pts) <= 1:
            out_series.append(series)
            continue

        # sort by original x, preserve relative order for ties
        pts_sorted = sorted(
            enumerate(pts),
            key=lambda t: (float(t[1]["x"]), t[0])
        )
        pts_sorted = [p for _, p in pts_sorted]

        xs = np.array([float(p["x"]) for p in pts_sorted])
        ys = np.array([float(p["y"]) for p in pts_sorted])

        x_min, x_max = xs.min(), xs.max()
        x_range = x_max - x_min
        eps = eps_frac * x_range if x_range > 0 else eps_frac

        new_pts = []
        y_prev = None

        i = 0
        while i < len(pts_sorted):
            x = xs[i]

            # collect block with same x
            j = i
            block = []
            while j < len(xs) and xs[j] == x:
                block.append(ys[j])
                j += 1

            if len(block) == 1:
                y = block[0]
                new_pts.append({"x": x, "y": y})
                y_prev = y
            else:
                if y_prev is None:
                    # no history: keep original order, no epsilon
                    for k, y in enumerate(block):
                        new_pts.append({
                            "x": x + k * eps,
                            "y": y,
                            "synthetic_x": k > 0,
                            "x_base": x,
                        })
                    y_prev = block[-1]
                else:
                    # sort by distance to previous y
                    block_sorted = sorted(
                        block,
                        key=lambda v: abs(v - y_prev)
                    )

                    # closest first (no shift)
                    y0 = block_sorted[0]
                    new_pts.append({"x": x, "y": y0})

                    # remaining: furthest first, epsilon-shifted
                    rest = sorted(
                        block_sorted[1:],
                        key=lambda v: abs(v - y_prev),
                        reverse=True,
                    )

                    for k, y in enumerate(rest, start=1):
                        new_pts.append({
                            "x": x + k * eps,
                            "y": y,
                            "synthetic_x": True,
                            "x_base": x,
                        })

                    y_prev = new_pts[-1]["y"]

            i = j

        out_series.append({
            "id": series.get("id"),
            "points": new_pts,
        })

    return {"series": out_series}

# ================================
# Tool 2: KM detection plan → track_km_curves
# ================================
from interval_crops import AxisCal

from itertools import product


def km_detection_tool(
    img,
    starting_y_value: float,
    direction: str,
    series: list,
    x_cal,
    y_cal,
    exploration_result: list | None = None,
    max_series: int | None = None,
    offset_px: int = 5,
    tool_args: dict = {},
    state: dict = {},
):
    """
    Joint KM snapping with INDEPENDENT per-series exploration ranges.

    - Each series snaps only within its own exploration range
    - Series stop independently
    - Interaction applies only when both series are active
    - No extrapolation outside exploration
    - some parameters are not used, this is because i tried several variants and this ended up being the most robust, but the prompt is still generic and can accomodate other variants.
    """

    if exploration_result is None or not series:
        return {"series": []}

    S = len(series)

    # ------------------------------------------------------------
    # Mask + axes
    # ------------------------------------------------------------
    mask = extract_non_background_mask(
        img,
        background_tol=20,
        min_saturation=40,
        debug=False,
    )

    H, W = mask.shape
    _, y_axis_x, _, _ = _detect_axes(img)
    x_axis_limit = int(y_axis_x + offset_px)

    # ------------------------------------------------------------
    # Parse exploration guidance
    # ------------------------------------------------------------
    y_ref = np.full((S, W), np.nan)
    series_x_start = np.zeros(S, dtype=int)
    series_x_end = np.zeros(S, dtype=int)

    global_x_min = W
    global_x_max = 0

    for s, guid in enumerate(exploration_result):
        if guid is None:
            return {"series": []}

        gx, gy = guid
        gx = np.asarray(gx, float)
        gy = np.asarray(gy, float)

        if len(gx) == 0:
            return {"series": []}

        order = np.argsort(gx)
        gx = gx[order].astype(int)
        gy = gy[order].astype(int)

        gx = np.clip(gx, 0, W - 1)
        gy = np.clip(gy, 0, H - 1)

        series_x_start[s] = gx[0]
        series_x_end[s] = gx[-1]

        global_x_min = min(global_x_min, gx[0])
        global_x_max = max(global_x_max, gx[-1])

        # right-continuous fill ONLY inside this series span
        j = 0
        cur = gy[0]
        for x in range(gx[0], gx[-1] + 1):
            while j + 1 < len(gx) and x >= gx[j + 1]:
                j += 1
                cur = gy[j]
            y_ref[s, x] = cur

    # ------------------------------------------------------------
    # Global x loop (union of all ranges)
    # ------------------------------------------------------------
    x_start = max(x_axis_limit, global_x_min)
    x_end = global_x_max + 1

    # ------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------
    snap_dy = 6
    snap_expand = 2
    max_expand = 25
    snap_dx = 4

    def candidates_for_series(s, x):
        if x < series_x_start[s] or x > series_x_end[s]:
            return None  # inactive

        y0 = y_ref[s, x]
        if not np.isfinite(y0):
            return []

        y0 = int(round(y0))
        dy = snap_dy

        while dy <= max_expand:
            y_lo = max(0, y0 - dy)
            y_hi = min(H, y0 + dy + 1)

            ys = []
            for xx in range(x, min(W, x + snap_dx)):
                col = mask[y_lo:y_hi, xx]
                if np.any(col):
                    ys.extend(y_lo + np.where(col > 0)[0])

            if ys:
                return [(int(y), dy) for y in np.unique(ys)]

            dy += snap_expand

        # KM flat segment allowed
        return []

    # ------------------------------------------------------------
    # Beam search with per-series activity
    # ------------------------------------------------------------
    beam_size = 12
    smooth_w = 0.6
    interact_w = 1.0
    interact_sigma = 6.0

    # initialize at first active x
    init_ys = []
    active_init = []

    for s in range(S):
        if x_start <= series_x_end[s]:
            init_ys.append(int(y_ref[s, max(x_start, series_x_start[s])]))
            active_init.append(True)
        else:
            init_ys.append(None)
            active_init.append(False)

    beam = [(0.0, tuple(init_ys), [])]

    for x in range(x_start, x_end):
        new_beam = []

        for base_cost, last_ys, hist in beam:
            cand_lists = []
            active_flags = []

            for s in range(S):
                cands = candidates_for_series(s, x)

                if cands is None:
                    # inactive series
                    cand_lists.append([(last_ys[s], 0)])
                    active_flags.append(False)
                elif not cands:
                    # active but flat → hold y
                    cand_lists.append([(last_ys[s], max_expand)])
                    active_flags.append(True)
                else:
                    cand_lists.append(cands)
                    active_flags.append(True)

            for choice in product(*cand_lists):
                ys = [c[0] for c in choice]
                dys = [c[1] for c in choice]

                cost = base_cost

                # smoothness only if active
                for s in range(S):
                    if active_flags[s]:
                        cost += dys[s]
                        cost += smooth_w * abs(ys[s] - last_ys[s])

                # interaction only when both active
                for i in range(S):
                    if not active_flags[i]:
                        continue
                    for j in range(i + 1, S):
                        if not active_flags[j]:
                            continue
                        d = abs(ys[i] - ys[j])
                        cost += interact_w * np.exp(-d / interact_sigma)

                new_beam.append((cost, tuple(ys), hist + [ys]))

        if not new_beam:
            break

        new_beam.sort(key=lambda t: t[0])
        beam = new_beam[:beam_size]

    if not beam:
        return {"series": []}

    # ------------------------------------------------------------
    # Extract best path
    # ------------------------------------------------------------
    _, _, hist = min(beam, key=lambda t: t[0])
    ys_track = np.asarray(hist).T  # [S, T]
    xs_px = np.arange(x_start, x_start + ys_track.shape[1])

    # ------------------------------------------------------------
    # Emit per-series outputs (trimmed individually)
    # ------------------------------------------------------------
    series_out = []

    for s, series_def in enumerate(series):
        xs_s = []
        ys_s = []

        for i, x in enumerate(xs_px):
            if series_x_start[s] <= x <= series_x_end[s]:
                xs_s.append(x)
                ys_s.append(ys_track[s, i])

        if not xs_s:
            continue

        xs_val = x_cal.p2v(np.array(xs_s))
        ys_val = y_cal.p2v(median_filter(np.array(ys_s), size=5))

        series_out.append({
            "label": series_def.get("name", series_def.get("id", f"s{s}")),
            "points": [
                {"x": float(x), "y": float(y)}
                for x, y in zip(xs_val, ys_val)
            ],
        })

    return {"series": series_out}

# --- Helper function definitions (Non-tool specific, minimal utility) ---
def truncate_model_series_points(
    model_obj: dict,
    max_points: int = 1,
) -> dict:
    """
    Keep only the first `max_points` (by x) for each model series.
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

def _get_mime_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    return 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'

def _load_image_and_mask(image_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(f"Cannot read image: {image_path}")    
    return img
def _execute_axis_tool_sync(args: Dict[str, Any], ocr_results: Dict[str, Any], image_path: str) -> Tuple[AxisCal, AxisCal]:
    
    # Call the imported axis_properties_tool logic
    return axis_properties_tool(ocr_results, image_path, **args)

# --- Tool Executor Wrappers (These are the new functions that orchestrate) ---
def _execute_km_tool(tool_args, state, img, model):
    """
    KM executor with inline LLM guidance extraction (guidance-first).

    Flow:
    1) Call Gemini directly on the full image to extract rough KM guidance
    2) Align guidance strictly by series id
    3) Convert guidance to PIXEL SPACE
    4) Pass guidance into km_detection_tool (guidance snapping)
    """

    image_path = state["image_path"]
    series_info = tool_args["series_info"]

    # ------------------------------------------------------------
    # 1) Build strict LLM prompt (series identity enforced)
    # ------------------------------------------------------------
    series_lines = []
    for s in series_info:
        series_lines.append(
            f"- id: {s['id']}, name: {s.get('name','')}, color: {s.get('color','none')}"
        )

    prompt_text = f"""
You are extracting Kaplan–Meier survival curves from a medical plot.

The plot contains EXACTLY the following series.
You MUST use these ids exactly and MUST NOT invent new series.

SERIES:
{chr(10).join(series_lines)}

RULES:
- Each series is independent.
- Do NOT merge or align curves across series.
- Curves are monotonic step functions.
- Prefer visual accuracy over smoothness.
- Return only clearly visible points.
- Points are for GUIDANCE ONLY, not final output.

TASK:
For each series id, extract a rough set of (x, y) points following the KM steps.

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "series": [
    {{
      "id": string,
      "points": [{{"x": number, "y": number}}]
    }}
  ]
}}
"""

    # ------------------------------------------------------------
    # 2) Load image + call Gemini (inline, streaming)
    # ------------------------------------------------------------
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    contents = [
        types.Content(
            parts=[
                types.Part(text=prompt_text),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png",
                        data=img_bytes,
                    ),
                    media_resolution={"level": "media_resolution_high"},
                ),
            ]
        )
    ]

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )

    t0 = _time.time()
    response_stream = client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    )

    raw_text = ""
    for chunk in response_stream:
        if not chunk.candidates:
            continue
        for part in chunk.candidates[0].content.parts or []:
            if part.text:
                raw_text += part.text

    t1 = _time.time()
    print(f"KM LLM inference time: {t1 - t0:.2f}s")
    
    # ------------------------------------------------------------
    # 3) Parse + validate LLM guidance (by id)
    # ------------------------------------------------------------
    try:
        llm_result = json.loads(raw_text)
    except Exception as e:
        raise RuntimeError("KM LLM guidance is not valid JSON") from e
    
    # id → guidance points (value space)
    guidance_by_id = {s["id"]: None for s in series_info}

    for s in llm_result.get("series", []):
        sid = s.get("id")
        if sid not in guidance_by_id:
            continue

        pts = []
        for p in s.get("points", []):
            try:
                pts.append({
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                })
            except Exception:
                continue

        if pts:
            guidance_by_id[sid] = pts

    # ------------------------------------------------------------
    # 4) Convert guidance VALUE → PIXEL (ordered by series_info)
    # ------------------------------------------------------------
    guidance_px_per_index = []

    for s in series_info:
        pts = guidance_by_id[s["id"]]
        if not pts:
            guidance_px_per_index.append(None)
            continue

        gx_val = np.array([p["x"] for p in pts], dtype=float)
        gy_val = np.array([p["y"] for p in pts], dtype=float)

        # VALUE → PIXEL conversion (CRITICAL)
        gx_px = np.array([state["x_cal"].v2p(v) for v in gx_val], dtype=float)
        gy_px = np.array([state["y_cal"].v2p(v) for v in gy_val], dtype=float)

        guidance_px_per_index.append((gx_px, gy_px))

    # ------------------------------------------------------------
    # 5) Call km_detection_tool (guidance snapping)
    # ------------------------------------------------------------
    final_result = km_detection_tool(
        img=img,
        starting_y_value=tool_args["starting_y_value"],
        direction=tool_args["direction"],
        series=series_info,
        x_cal=state["x_cal"],
        y_cal=state["y_cal"],
        exploration_result=guidance_px_per_index,  # <<< PIXEL guidance
        max_series=tool_args.get("max_series"),
        offset_px=tool_args.get("offset_px", 5),
        tool_args=tool_args,
        state=state
    )

    # ------------------------------------------------------------
    # 6) Fallback: retry without color filtering (optional)
    # ------------------------------------------------------------
    if final_result is None:
        for s in series_info:
            s["color"] = "none"

        final_result = km_detection_tool(
            img=img,
            starting_y_value=tool_args["starting_y_value"],
            direction=tool_args["direction"],
            series=series_info,
            x_cal=state["x_cal"],
            y_cal=state["y_cal"],
            exploration_result=guidance_px_per_index,
            max_series=tool_args.get("max_series"),
            offset_px=tool_args.get("offset_px", 5),
            tool_args=tool_args,
            state=state
        )

    return final_result


def _execute_crop_gen_tool(args: Dict[str, Any], state: Dict[str, Any], image_path: str) -> Tuple[List[Dict[str, Any]], str]:
    # Safely extract intervals, description, llm_axis_info
    intervals = args["intervals"]
    cal_x, cal_y = state["x_cal"], state["y_cal"]
    
    print("Running crop generation tool...")
    files, crops = run_step2(
        chart_path=image_path,
        intervals=intervals,
        llm_axis_info=[],
        x_cal=cal_x,
        y_cal=cal_y,
    )
    # Return the metadata and the path to the crop directory
    return crops.meta, crops, os.path.dirname(files[0]) if files else None


def _balanced_json_extract(text: str) -> dict:
    """Extract the first top-level JSON object via brace counting."""
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found.")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i+1])
    raise ValueError("Unbalanced JSON in model output.")

def _img_to_data_url(path: str) -> str:
    """Converts a local file path to a base64 data URL."""
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except FileNotFoundError:
        # Create dummy file if not found, to avoid crash when loading image
        try:
            from PIL import Image
            Image.new('RGB', (1, 1), color = 'red').save(path)
        except:
             pass
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"


def convert_to_gemini_part(part: Dict[str, Any]) -> types.Part:
    """Converts the pipeline's image_url or text format to the Gemini SDK's Part objects."""
    if part.get("type") == "image_url":
        data_url = part["image_url"]["url"]
        if data_url.startswith("data:"):
            try:
                mime_type = data_url.split(":")[1].split(";")[0]
                b64_data = data_url.split(",", 1)[1]
                data_bytes = base64.b64decode(b64_data)
                # Use simple Blob/Part structure to avoid v1alpha compatibility issues
                return types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=data_bytes
                    )
                )
            except Exception as e:
                raise ValueError(f"Failed to decode data URL to Part: {e}")
    elif part.get("type") == "text":
        # FIX: Use keyword argument to resolve TypeError (Part.from_text takes 1 positional)
        return types.Part.from_text(text=part["text"])
    
    raise TypeError(f"Unsupported part type in convert_to_gemini_part: {part.get('type')}")
    
def _sync_client_chat(model: str, contents: List[types.Part], is_json: bool, thinking_level: str = "low") -> str:
    """Synchronous API call wrapper for the Gemini SDK with thinking_level."""
    
    config_args = {'thinking_config': types.ThinkingConfig(thinking_level=thinking_level)}
    if is_json:
        config_args['response_mime_type'] = "application/json"
    
    config = types.GenerateContentConfig(**config_args)
    
    # ⚠️ ADD RETRY LOGIC for 503 errors 
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response.text
        except APIError as e:
            if e.response.status_code in (500, 503, 429) and attempt < 2:
                print(f"Server error {e.response.status_code}. Retrying in {2**attempt}s...")
                time.sleep(2**attempt)
            else:
                raise e
    return ""

def gemini_chat(model: str, messages: List[Dict[str, Any]], is_json=False) -> str:
    """Synchronous Gemini Chat Wrapper."""
    content_parts = messages[0].get("content", [])
    converted_parts = [convert_to_gemini_part(p) for p in content_parts]
    return _sync_client_chat(model, converted_parts, is_json)


# --- Main Orchestration Function ---
# --- GLOBAL STATE for concurrent execution ---
# Executor for running OCR concurrently
ocr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Global storage for the first, successful Axis Calibration results
GLOBAL_CALIBRATION_RESULT = {
    "x_cal": None, "y_cal": None, "ocr_res": None, 
    "x_axis_y": None, "y_axis_x": None
}

def save_extracted_points(result, out_path=None, fmt="csv"):
    """Save the extracted points to a file.
    Args:
        result (Dict): A dict with all series and their points.
        out_path (_type_, optional): Defaults to None.
        fmt (str, optional): The format is either a csv or a json. Defaults to "csv".
    """
    series_list = result.get("series", [])
    print(f"Found {len(series_list)} series")

    if not series_list:
        print("⚠️ No series found in result.")
        return

    fmt = fmt.lower()
    if out_path is None:
        out_path = f"extracted_points.{fmt}"

    # -------- JSON --------
    if fmt == "json":
        data = []
        for i, s in enumerate(series_list):
            xs = [float(p["x"]) for p in s.get("points", [])]
            ys = [float(p["y"]) for p in s.get("points", [])]

            data.append({
                "series_index": i,
                "label": s.get("label", ""),
                "x": xs,
                "y": ys
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Saved {len(data)} series to JSON: {out_path}")
        return

    # -------- CSV --------
    if fmt == "csv":
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["series_index", "label", "x", "y"])

            total_points = 0
            for i, s in enumerate(series_list):
                label = s.get("label", "")
                for p in s.get("points", []):
                    x = round(float(p["x"]), 4)
                    y = round(float(p["y"]), 4)
                    writer.writerow([i, label, x, y])
                    total_points += 1

        print(f"✅ Saved CSV with {total_points} points to {out_path}")
        return

    raise ValueError("fmt must be either 'csv' or 'json'")

def normalize_axis_breaks(
    ticks: List[float],
    breaks: Union[List[Any], List[List[Any]]],
) -> Dict[str, Any]:
    """
    Normalize axis breaks.

    Supported break formats:
      - breaks = [lo, hi]                      (single break)
      - breaks = [[lo, hi], [lo2, hi2], ...]   (multiple breaks)

    Endpoints may be str/int/float; they are converted to float.

    Rules:
    - A break is kept ONLY if there are at least 2 ticks strictly below `lo`
      AND at least 2 ticks strictly above `hi`.
    - If either side has <= 1 tick:
        - remove the break
        - remove the isolated tick(s) on that side
    - Ticks remain numeric only.
    """

    if not ticks:
        return {"ticks": [], "breaks": []}

    # Ensure numeric ticks
    ticks_f = [float(t) for t in ticks]
    ticks_f.sort()

    # Normalize breaks into a list of [lo, hi] pairs
    break_pairs: List[List[Any]] = []
    if not breaks:
        return {"ticks": ticks_f, "breaks": []}

    # Case A: single break like [10, 50]
    if isinstance(breaks, (list, tuple)) and len(breaks) == 2 and not isinstance(breaks[0], (list, tuple)):
        break_pairs = [list(breaks)]
    # Case B: list of breaks like [[10, 50], [60, 120]]
    elif isinstance(breaks, (list, tuple)):
        break_pairs = list(breaks)  # may still contain malformed entries; filtered below
    else:
        # Unsupported type
        return {"ticks": ticks_f, "breaks": []}

    kept_breaks: List[List[float]] = []

    for br in break_pairs:
        if not isinstance(br, (list, tuple)) or len(br) != 2:
            continue

        try:
            lo = float(br[0])
            hi = float(br[1])
        except (TypeError, ValueError):
            continue

        if lo >= hi:
            continue

        below = [t for t in ticks_f if t <= lo]
        above = [t for t in ticks_f if t >= hi]
        print("Break", br, "has", len(below), "ticks below and", len(above), "ticks above.")
        # Case 1: meaningful break → keep
        if len(below) >= 2 and len(above) >= 2:
            kept_breaks.append([lo, hi])
            continue

        # Case 2: trivial break → prune sparse side(s)
        if len(below) <= 1:
            ticks_f = above

        if len(above) <= 1:
            ticks_f = below

    return {
        "ticks": ticks_f,
        "breaks": kept_breaks,
    }
def normalize_axis_if_needed(axis: dict) -> dict:
    """
    Normalize axis breaks only if they exist.
    Axis is modified in-place and also returned.
    """

    if not axis or "ticks" not in axis:
        return axis

    breaks = axis.get("break")

    # No breaks → nothing to do
    if breaks is None:
        return axis

    if isinstance(breaks, list) and len(breaks) == 0:
        axis["break"] = []
        return axis

    # Normalize
    norm = normalize_axis_breaks(axis["ticks"], breaks)

    axis["ticks"] = norm["ticks"]
    axis["break"] = norm["breaks"]
    print(axis["ticks"])

    return axis

def draw_calibrated_axes_from_axis_info(
    ax,
    x_cal,
    y_cal,
    axis_info,
    W,
    H,
    tick_len_px: int = 5,
    fontsize: int = 6,
    axis_lw: float = 0.6,
    tick_lw: float = 0.6,
    color: str = "#444444",
    alpha: float = 0.65,
):
    """
    Draw calibrated axes using ONLY ticks provided in axis_info
    (after normalize_axis_if_needed).

    No inferred ticks, no spacing heuristics.
    """

    # ------------------------------------------------------------
    # Normalize axis info (caller wanted this explicitly)
    # ------------------------------------------------------------
    x_axis = normalize_axis_if_needed(axis_info["x_axis"])
    y_axis = normalize_axis_if_needed(axis_info["y_axis"])

    x_ticks = x_axis.get("ticks", []) or []
    y_ticks = y_axis.get("ticks", []) or []

    # ------------------------------------------------------------
    # X AXIS (bottom)
    # ------------------------------------------------------------
    y0 = H - 1
    ax.plot(
        [0, W - 1],
        [y0, y0],
        lw=axis_lw,
        color=color,
        alpha=alpha,
        zorder=5,
    )

    for xv in x_ticks:
        try:
            px = x_cal.v2p(float(xv))
        except Exception:
            continue

        if 0 <= px < W:
            ax.plot(
                [px, px],
                [y0, y0 - tick_len_px],
                lw=tick_lw,
                color=color,
                alpha=alpha,
                zorder=5,
            )
            ax.text(
                px,
                y0 - tick_len_px - 1,
                f"{xv:.3g}",
                ha="center",
                va="top",
                fontsize=fontsize,
                color=color,
                alpha=alpha,
                zorder=6,
            )

    # ------------------------------------------------------------
    # Y AXIS (left)
    # ------------------------------------------------------------
    x0 = 0
    ax.plot(
        [x0, x0],
        [0, H - 1],
        lw=axis_lw,
        color=color,
        alpha=alpha,
        zorder=5,
    )

    for yv in y_ticks:
        try:
            py = y_cal.v2p(float(yv))
        except Exception:
            continue

        if 0 <= py < H:
            ax.plot(
                [x0, x0 + tick_len_px],
                [py, py],
                lw=tick_lw,
                color=color,
                alpha=alpha,
                zorder=5,
            )
            ax.text(
                x0 + tick_len_px + 1,
                py,
                f"{yv:.3g}",
                ha="left",
                va="center",
                fontsize=fontsize,
                color=color,
                alpha=alpha,
                zorder=6,
            )


def save_extracted_series_images(
    result: dict,
    image_path: str,
    x_cal,
    y_cal,
    axis_info,
    out_dir: str,
    prefix: str,
    figsize=(6, 4),
    dpi=150,
):
    """
    Save extracted series plots with strict calibration consistency.

    Outputs:
      - per-series overlays on original image
      - combined overlay on original image
      - CSVs of extracted data in:
          * value space  (x, y)
          * pixel space  (x_px, y_px)

    Directory structure:
      out_dir/
        overlays/
        csv/
          series_<i>_values.csv
          series_<i>_pixels.csv
          all_series_values.csv
          all_series_pixels.csv

    Folder is deleted and recreated on each call.
    """

    # ============================================================
    # CLEAN OUTPUT DIRECTORY (STRICT)
    # ============================================================
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    overlays_dir = os.path.join(out_dir, "overlays")
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    series_list = result.get("series", [])
    if not series_list:
        print("⚠️ No series to plot.")
        return

    # ============================================================
    # LOAD ORIGINAL IMAGE
    # ============================================================
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # ============================================================
    # ACCUMULATORS FOR COMBINED CSVs
    # ============================================================
    all_values_rows = []
    all_pixels_rows = []

    # ============================================================
    # PER-SERIES PROCESSING
    # ============================================================
    for i, s in enumerate(series_list):
        pts = s.get("points", [])
        if not pts:
            continue

        label = s.get("label", f"Series {i}")

        # -------------------------
        # RAW VALUES
        # -------------------------
        x_raw = np.array([float(p["x"]) for p in pts])
        y_raw = np.array([float(p["y"]) for p in pts])

        # -------------------------
        # PIXEL VALUES
        # -------------------------
        x_px = x_cal.v2p(x_raw)
        y_px = y_cal.v2p(y_raw)

        # ========================================================
        # SAVE PER-SERIES CSVs
        # ========================================================
        df_values = pd.DataFrame({
            "x": x_raw,
            "y": y_raw,
        })

        df_pixels = pd.DataFrame({
            "x_px": x_px,
            "y_px": y_px,
        })

        values_path = os.path.join(csv_dir, f"{prefix}_series_{i}_values.csv")
        pixels_path = os.path.join(csv_dir, f"{prefix}_series_{i}_pixels.csv")

        df_values.to_csv(values_path, index=False)
        df_pixels.to_csv(pixels_path, index=False)

        # accumulate combined
        for xv, yv, xp, yp in zip(x_raw, y_raw, x_px, y_px):
            all_values_rows.append({
                "series_id": i,
                "series_label": label,
                "x": xv,
                "y": yv,
            })
            all_pixels_rows.append({
                "series_id": i,
                "series_label": label,
                "x_px": xp,
                "y_px": yp,
            })

        # ========================================================
        # PER-SERIES OVERLAY
        # ========================================================
        fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax.imshow(img_rgb)
        ax.plot(x_px, y_px, "-o", lw=2)
        draw_calibrated_axes_from_axis_info(ax, x_cal, y_cal, axis_info, W, H)
        ax.set_title(label)
        ax.axis("off")

        fig.savefig(
            os.path.join(overlays_dir, f"{prefix}_series_{i}_overlay.png"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

    # ============================================================
    # SAVE COMBINED CSVs
    # ============================================================
    if all_values_rows:
        pd.DataFrame(all_values_rows).to_csv(
            os.path.join(csv_dir, f"{prefix}_all_series_values.csv"),
            index=False,
        )

    if all_pixels_rows:
        pd.DataFrame(all_pixels_rows).to_csv(
            os.path.join(csv_dir, f"{prefix}_all_series_pixels.csv"),
            index=False,
        )

    # ============================================================
    # COMBINED OVERLAY
    # ============================================================
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(img_rgb)

    for i, s in enumerate(series_list):
        pts = s.get("points", [])
        if not pts:
            continue

        x_raw = np.array([float(p["x"]) for p in pts])
        y_raw = np.array([float(p["y"]) for p in pts])

        ax.plot(
            x_cal.v2p(x_raw),
            y_cal.v2p(y_raw),
            lw=2,
            label=s.get("label", f"{i}"),
        )

    draw_calibrated_axes_from_axis_info(ax, x_cal, y_cal, axis_info, W, H)
    ax.legend(fontsize=8)
    ax.axis("off")

    fig.savefig(
        os.path.join(overlays_dir, f"{prefix}_all_overlay.png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    print(f"✅ Saved overlays and CSVs to: {out_dir}")


def analyze_single_image_async(
    img_path: str,
    series_descriptions: List[Any] | None = None,
    global_image_path: str = None, # Keeping parameter for signature compatibility, but unused
    model: str = "gemini-3-pro-preview",
) -> Dict[str, Any]:
    """
    Optimized Gemini analysis focusing on extraction precision.
    Uses the logic of Prompt 2: Read Axes -> Identify -> Validate.
    """
    name = Path(img_path).name
    # Normalize series_descriptions to help the model distinguish between lines
    if not series_descriptions:
        series_desc_text = "No specific series labels provided. Use colors (Red, Cyan, etc.) or relative positions."
    elif isinstance(series_descriptions, (list, tuple)):
        series_desc_text = "\n".join(f"- {str(s)}" for s in series_descriptions)
    else:
        series_desc_text = str(series_descriptions)
    # REWRITTEN SYSTEM PROMPT (Prompt 2 Spirit)
    system_prompt = f"""
You are an expert system extracting NUMERICAL data from a HIGH-QUALITY scientific plot crop.
Return all visible data series. Work in three internal steps:
1) Read axes: Locate tick labels and grid lines. Determine if x/y are linear or log10.
2) Identify series: Use the descriptions below to map visual lines to series 'id' field extracted from the descriptions (ex: s1, s2, sk ...).
3) Validate: Ensure all points are within the visible frame and match the tick progression.

TARGET SERIES DESCRIPTIONS:
{series_desc_text}

TASK:
- Extract coordinates relative to the tick labels (e.g., if a point is halfway between tick 1 and 2, x_tick is 1.5 (if axis isn't logarithmic)).
- Capture all discrete markers and major inflection points on curves.
- Be precise: Multiple series may be very close or overlapping. Distinguish them by using color and relative vertical rank.

RULES:
- Do NOT guess data hidden by labels or the frame.
- ALWAYS assign a detected series ID from the provided descriptions.
- Do NOT extrapolate beyond what is visually certain.
- Be careful to not get tricked by grid lines, confidence intervals, or other chart elements. 
- Output at least 5 points per series if visible.
- STRICT JSON OUTPUT ONLY.

OUTPUT FORMAT:
OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "series": [
    {{
      "id": string,      // MUST match the input series id exactly
      "label": string,   // Preserve label if provided
      "points": [
        {{"x": number, "y": number}}
      ]
    }}
  ]
}}
"""

    try:
        img_url = _img_to_data_url(img_path)

        # Removed the global image to reduce token noise and processing time
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": {"url": img_url}}
            ]
        }]
        # Set is_json=True to enforce the schema
        out = gemini_chat(model=model, messages=messages, is_json=True)

        parsed = json.loads(out)
        
        # Standardize output handling
        if isinstance(parsed, list):
            return {"filename": name, "series": parsed}
        if isinstance(parsed, dict):
            # Handle cases where the model wraps the list in a "series" key
            return {"filename": name, "series": parsed.get("series", parsed.get("data", []))}

        return {"filename": name, "error": "Invalid JSON structure"}

    except Exception as e:
        print(f"🚨 Exception in analyze_single_image_async for {name}: {e}")
        return {"filename": name, "error": str(e)}
    

def run_unified_plot_extractor(
    image_path: str,
    model="gemini-3-pro-preview",
    saving=False,
) -> dict:
    """
    Unified plot extractor with:

    - Concurrent LLM planning (calibration + routing)
    - Ordered, dependency-gated tool execution
    - Calibration is attempted first
    - If calibration fails:
        → categorical tool is still allowed
        → axis-dependent saving is disabled
    - HARD fallback to Gemini pure extraction if nothing usable works
    """

    # ============================================================
    # HARD FALLBACK (single exit)
    # ============================================================
    def _fallback():
        return extract_plot_data_gemini_streaming(image_path)

    start_time = time()

    # ============================================================
    # Initialization & CV pre-pass
    # ============================================================
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to load image")
    except Exception:
        return _fallback()

    # ------------------------------------------------------------
    # Detect axes (may fail on categorical plots)
    # ------------------------------------------------------------
    try:
        x_axis_y, y_axis_x, x_axis_span, y_axis_span = _detect_axes(img)

        GLOBAL_CALIBRATION_RESULT["x_axis_y"] = x_axis_y
        GLOBAL_CALIBRATION_RESULT["y_axis_x"] = y_axis_x

    except Exception:
        # Axis detection failed early → categorical may still work
        x_axis_y, y_axis_x = None, None

    # ============================================================
    # Shared mutable state
    # ============================================================
    state = {
        "axis_ready": False,
        "x_cal": None,
        "y_cal": None,
        "image_path": image_path,
        "crops_meta": None,
    }

    # ============================================================
    # Concurrency primitives
    # ============================================================
    llm_executor = ThreadPoolExecutor(max_workers=2)

    axis_plan_ready = threading.Event()
    router_plan_ready = threading.Event()

    axis_plan = {}
    router_response = {}

    calibration_failed = False

    # ============================================================
    # OCR (only meaningful if axes exist)
    # ============================================================
    ocr_future = None
    if x_axis_y is not None and y_axis_x is not None:
        try:
            ocr_future = ocr_executor.submit(
                process_image,
                image_path,
                x_axis_y=x_axis_y,
                y_axis_x=y_axis_x,
                per_plot_ocr=False,
            )
        except Exception:
            calibration_failed = True
    else:
        calibration_failed = True

    # ============================================================
    # Build Gemini image part
    # ============================================================
    image_part = types.Part.from_bytes(
        data=Path(image_path).read_bytes(),
        mime_type=_get_mime_type(image_path),
    )

    # ============================================================
    # LLM CALL A — CALIBRATION PLANNER
    # ============================================================
    def calibration_llm_call():
        try:
            response = client.models.generate_content(
                model=model,
                contents=[image_part, CALIBRATION_USER_PROMPT],
                config=types.GenerateContentConfig(
                    system_instruction=CALIBRATION_SYSTEM_PROMPT,
                    tools=[create_function_declaration(axis_properties_tool_def)],
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )

            for call in response.function_calls:
                if call.name == "axis_properties_tool":
                    axis_plan["args"] = dict(call.args)
                    axis_plan_ready.set()
                    return

            axis_plan["error"] = "No axis_properties_tool call"
            axis_plan_ready.set()

        except Exception as e:
            axis_plan["error"] = str(e)
            axis_plan_ready.set()

    # ============================================================
    # LLM CALL B — ROUTER PLANNER
    # ============================================================
    def router_llm_call():
        try:
            response = client.models.generate_content(
                model=model,
                contents=[image_part, UNIFIED_USER_PROMPT],
                config=types.GenerateContentConfig(
                    system_instruction=UNIFIED_SYSTEM_PROMPT,
                    tools=UNIFIED_GEMINI_TOOLS,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            router_response["response"] = response
            router_plan_ready.set()

        except Exception as e:
            router_response["error"] = str(e)
            router_plan_ready.set()

    # ============================================================
    # Launch both LLM calls concurrently
    # ============================================================
    llm_executor.submit(calibration_llm_call)
    llm_executor.submit(router_llm_call)

    # ============================================================
    # CALIBRATION PHASE (optional)
    # ============================================================
    axis_plan_ready.wait()

    if "error" in axis_plan:
        calibration_failed = True

    if not calibration_failed and ocr_future is not None:
        try:
            ocr_result = ocr_future.result()
            x_cal, y_cal = _execute_axis_tool_sync(
                axis_plan["args"],
                ocr_result,
                image_path,
            )
            print(f"Calibration result: x={x_cal}, y={y_cal}")
            if x_cal is None or y_cal is None:
                calibration_failed = True
            else:
                state["x_cal"] = x_cal
                state["y_cal"] = y_cal
                state["axis_ready"] = True

        except Exception:
            calibration_failed = True

    if state["axis_ready"]:
        print(f"Calibration completed in {time() - start_time:.2f}s")
    else:
        print("Calibration failed → categorical extraction still allowed")

    # ============================================================
    # ROUTER PHASE (always runs)
    # ============================================================
    router_plan_ready.wait()

    if "error" in router_response:
        return _fallback()

    response = router_response["response"]
    final_result = None

    # ============================================================
    # TOOL EXECUTION LOOP
    # ============================================================
    for call in response.function_calls:

        tool_name = call.name
        tool_args = dict(call.args)

        print(f"Running tool: {tool_name}")

        # --------------------------------------------------------
        # If calibration failed → only categorical is safe
        # --------------------------------------------------------
        if calibration_failed and tool_name != "categorical_tool":
            print(
                f"Skipping tool {tool_name} because calibration failed "
                "(categorical-only safe path)"
            )
            continue

        try:
            # ====================================================
            # CROP TOOL (requires axis calibration)
            # ====================================================
            if tool_name == "crop_generation_tool":
                cropping_pipeline = True
                if cropping_pipeline:
                    crops_meta, crops, crop_dir = _execute_crop_gen_tool(
                        tool_args,
                        state,
                        image_path,
                    )
                    state["crops_meta"] = crops_meta
                    print(tool_args["series_info"])
                    final_result = run_full_pipeline_async(
                        image_path,
                        crops,
                        model=model,
                        series_descriptions=tool_args["series_info"],
                    )
                else:
                    final_result = analyze_single_image_async(
                    img_path=image_path,
                    model=model,
                    series_descriptions=tool_args["series_info"],
                    )
                break

            # ====================================================
            # CATEGORICAL TOOL (works without calibration)
            # ====================================================
            elif tool_name == "categorical_tool":

                print("Categorical extraction triggered")

                final_result = categorical_graph_extraction_tool(
                    image_path,
                    model,
                )

                final_result = truncate_model_series_points(final_result)
                break

            # ====================================================
            # KM TOOL (requires calibration)
            # ====================================================
            elif tool_name == "km_detection_tool":

                final_result = _execute_km_tool(
                    tool_args,
                    state,
                    img,
                    model=model,
                )
                break

        except Exception as e:
            print("Tool execution failed:", tool_name, e)
            return _fallback()

    # ============================================================
    # Post-processing
    # ============================================================
    if final_result and "series" in final_result:
        final_result = expand_duplicate_x_with_continuity_final(
            final_result,
            eps_frac=0.01,
        )

    # ============================================================
    # If router produced nothing usable → fallback
    # ============================================================
    if not final_result:
        print("Router produced no valid output → fallback Gemini extraction")
        try:
            final_result = _fallback()
        except Exception as e:
            return {"error": f"Fallback failed: {e}"}

    # ============================================================
    # SAVING LOGIC
    # ============================================================
    if saving:

        # Axis-dependent saving ONLY if calibration succeeded
        if state["axis_ready"]:
            save_extracted_series_images(
                result=final_result,
                image_path=image_path,
                x_cal=state["x_cal"],
                y_cal=state["y_cal"],
                axis_info=axis_plan.get("args", {}).get("axis_info", {}),
                out_dir="extracted_plots",
                prefix=os.path.splitext(os.path.basename(image_path))[0],
            )

        # Always safe: save extracted points
        save_extracted_points(
            final_result,
            out_path="extracted_points.csv",
            fmt="csv",
        )

    return final_result

dataset_dir = "data/ExtractedDataMatthias/"
name_g1 = "Data-Naik2016-6MonthPSA-Strat-MetastasisFreeSurvival-as-cLMA-G8dv"
name_g3 = "Data-Naik2016-6MonthPSA-Strat-ProstateCancerSpecificMortality-as-AXhW-IH9H"
name_g5 = "Data-ProstateCancer-BiochemicalRecurrence-TestosteroneNadir-as-kPUM-ja2E"
image_path_g3 = dataset_dir + name_g3 + "/" + name_g3+ ".png"
image_path_base = "graphs/graph_test1.jpeg"
image_path_3 = "graphs/graph1.png"
image_path_g6 = "graphs/servier.png"
name_g7 = "Data-ProstateCancer-BiochemicalRecurrence-TestosteroneNadir-as-kPUM-ja2E"
name_g9 = "Data-ProstateCancer-TimeToCastrationResistance-CastrationLevel-as-G8iP-2p8H"
name_g10 = "Data-ProstateCancer-10YearMetastasis-TestosteroneNadir-as-HUFM-k4RT"
name_g11 = "Data-ProstateCancer-OverallSurvival-TestosteroneSuppression-as-gIUH-EPAn"
name_g12 = "Data-ProstateCancer-CastrationLevel-OverallSurvival-LocallyAdvanced-Metastatic-as-HN9j-M4NK"
name_g13 = "Data-ProstateCancer-CastrationLevel-TimeToProgression-LocallyAdvanced-Metastatic-as-aWvR-2wlo"
name_4 = "Data-ProstateCancer-CastrationLevel-TimeToProgression-BiochemicalRecurrence-as-4dag-RoWD"
name_g14 =  "Data-as-YpSF-UNAQ"
name_g15 = "Data-as-CdWI-s6OK"
name_g16 = "/Data-as-cfVO-vMjk"
name_g19 = "Data-as-Atl6-ihFg"
name_g20 = "Data-ProstateCancer-Survival-7MonthPsaResponse-as-7II1-HQ1y"
name_g21 = "Data-ProstateCancer-TimeToCauseSpecificSurvival-TestosteroneNadir-as-zQ4P-rWVB"
image_path_g4 = dataset_dir + name_4 + "/" + name_4 + ".png"
#name = "Data-ProstateCancer-CastrationLevel-OverallSurvival-BiochemicalRecurrence-as-KBNL-4YCX"
image_path_g1 = dataset_dir + name_g1 + "/" + name_g1 + ".png"
image_path_g2 = "graphs/graph2.webp"
image_path_g5 = dataset_dir + name_g5 + "/" + name_g5 + ".png"
image_path_g7 = dataset_dir + name_g7 + "/" + name_g7 + ".png"
image_path_g9 = dataset_dir + name_g9 + "/" + name_g9 + ".png"
image_path_g10 = dataset_dir + name_g10 + "/" + name_g10 + ".png" 
image_path_g11 = dataset_dir + name_g11 + "/" + name_g11 + ".png"
image_path_g12 = dataset_dir + name_g12 + "/" + name_g12 + ".png"
image_path_g13 = dataset_dir + name_g13 + "/" + name_g13 + ".png"
image_path_g14 = dataset_dir + name_g14 + "/" + name_g14 + ".png"
image_path_g15 = dataset_dir + name_g15 + "/" + name_g15 + ".png"
image_path_g16 = dataset_dir + name_g16 + "/" + name_g16 + ".png"
image_path_g19 = dataset_dir + name_g19 + "/" + name_g19 + ".png"
image_path_g20 = dataset_dir + name_g20 + "/" + name_g20 + ".png"
image_path_g21 = dataset_dir + name_g21 + "/" + name_g21 + ".png"
e_graph1 = "graphs/e_graph1.png"
e_graph2 = "graphs/e_graph2.png"
e_graph3 = "graphs/e_graph3.png"
e_graph4 = "graphs/e_graph4.png"
r_graph1 = "graphs/graph3.webp"
ri_graph1 = "graphs/riad_1.png"

if __name__ == "__main__":
    test_image_path = "graphs/e_graph1.png"  # Replace with your test image path
    test_image_path = image_path_g21
    from time import time
    start_time = time()
    print(test_image_path)
    result = run_unified_plot_extractor(test_image_path, model="gemini-3-pro-preview", saving=True)
    end_time = time()
    print(f"Extraction completed in {end_time - start_time:.2f} seconds.")
    print(result)
