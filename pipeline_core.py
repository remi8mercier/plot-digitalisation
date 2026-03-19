"""
This module contains the core logic for the LLM-based chart digitization pipeline.
Step 1: Interval and Axis Extraction (Synchronous)
Step 2: Using those to compute the axis calibration, and return the crops to analyze.
Step 3: Concurrent Crop Analysis (Async).

"""


# helpers_llm_pipeline.py
import os, re, json, base64
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import itertools
from typing import List, Dict, Any, Optional
import shutil
import re
import json
import base64
import time
import asyncio
import math
import nest_asyncio
import concurrent.futures # <-- New Import for Threading
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from interval_crops import IntervalCropResult

import shutil
import os
from interval_crops import AxisCal
from dotenv import load_dotenv
from interval_crops import make_interval_crops

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

prompt_agent_1 = """
You will study a chart image and produce FOUR sections, in this exact order:

=== INTERVALS (JSON) ===
<json array only>

=== AXIS_INFO (JSON) ===
{
  "x_axis": {
    "label": "<verbatim axis label text if readable, else null>",
    "type": "linear" | "log10" | "unknown",
    "range": [xmin, xmax],           // approximate numeric range
    "ticks": [v1, v2, v3, ...]       // ordered major tick values you can read/estimate
  },
  "y_axis": {
    "label": "<verbatim axis label text if readable, else null>",
    "type": "linear" | "log10" | "unknown",
    "range": [ymin, ymax],
    "ticks": [v1, v2, v3, ...]
  }
}

=== DESCRIPTION ===
<freeform text>

Guidelines:
- The REAL image is the authority (primary truth).
- If provided, vibe_grid is only for coarse shape confirmation.
- If provided, axis_ocr only corroborates where ticks likely are; do not override the image.
- Axes can be linear or log10; infer from spacing and labels (e.g., 10^-3, 10^-2, 10^-1, 1, 10, 10^2).
- For AXIS_INFO.ticks: list the visible/clear **major** ticks in order; omit ambiguous ones.
- Ranges in INTERVALS are for cropping; be compact (≈5–10 items) including a tail/coverage region.
- Always include a final helper interval with id="scale", giving two fields:
  {"id":"scale",   "x":{"step":"<numeric step>","label_step":<int>}, "y":{"step":"<numeric step>","label_step":<int>}}
  - `step` = the degree of precision needed to extract the graph faithfully (not just the smallest tick shown).  
    Choose the smallest numeric increment that is **meaningful for visible variation**, not merely for numerical neatness.  
    If no visible variation occurs below a certain scale, keep `step` coarser (e.g. 0.1, 1, 5, 10).  
    Only use finer steps (e.g. 0.01 or 0.001) when clear sub-tick details are *visibly* resolved on the chart itself.  
    As a rule of thumb, `step` should normally not exceed 1/20 of the total axis span unless the graph shows finer detail.
  - `label_step` = how many steps to skip between labels so that labels don’t overlap when redrawing. 
    This is a layout safeguard, not necessarily what is shown in the image.

— FEW-SHOT EXAMPLES —

Example A
=== INTERVALS (JSON) ===
[
  {"id":"start",   "x":[0.0, 0.5], "y":[0.0, 0.3]},
  {"id":"ramp1",   "x":[0.8, 1.2], "y":[0.8, 1.8]},
  {"id":"knee",    "x":[1.8, 2.2], "y":[3.5, 4.8]},
  {"id":"midrise", "x":[2.8, 3.2], "y":[4.8, 6.2]},
  {"id":"high",    "x":[4.5, 5.5], "y":[6.2, 7.4]},
  {"id":"tail",    "x":[5.8, 6.1], "y":[7.2, 8.2]},
]

=== AXIS_INFO (JSON) ===
{
  "x_axis": {"label":"Time (s)","type":"linear","range":[0,6.2],"ticks":[0,1,2,3,4,5,6]},
  "y_axis": {"label":"Amplitude","type":"linear","range":[0,8.5],"ticks":[0,2,4,6,8]}
}

=== DESCRIPTION === 
A single rising line. It begins near the baseline, climbs gently at first, then steepens around the second x tick, reaching a mid-level band before easing into a high region. Toward the right edge it approaches its top band without a sharp peak, ending high and still trending upward slightly.

Example B
=== INTERVALS (JSON) ===
[
  {"id":"start",   "x":[-2, 2], "y":[-0.1, 0.1]},
  {"id":"riseA",   "x":[0.6, 0.9], "y":[0.6, 0.9]},
  {"id":"peak1",   "x":[1.4, 1.7], "y":[0.9, 1.1]},
  {"id":"dip1",    "x":[3.0, 3.2], "y":[-0.1, 0.1]},
  {"id":"bump",    "x":[3.5, 3.7], "y":[0.5, 0.8]},
  {"id":"expA",    "x":[4.0, 4.3], "y":[1.3, 2.1]},
  {"id":"surge",   "x":[5.0, 5.2], "y":[5.5, 7.0]},
  {"id":"near0",                 "y":[-0.1, 0.1]},
  {"id":"plateau_hint","x":[1.9, 2.5]},
]

=== AXIS_INFO (JSON) ===
{
  "x_axis": {"label":"Frequency (Hz)","type":"log10","range":[0.1,1000],"ticks":[0.1,1,10,100,1000]},
  "y_axis": {"label":"Power","type":"linear","range":[-0.2,7.5],"ticks":[0,2,4,6]}
}

=== DESCRIPTION ===
A wave-like start: quick rise to a modest crest, soft decline to baseline, then a renewed upswing. After mid-range, growth accelerates sharply, transitioning into an exponential-looking surge toward the far right. The earliest and mid-right behaviors look smoother; the late segment is much steeper and dominates the overall scale.
"""


#----------------------------------------------------------------------
# 1. CORE LLM CLIENT AND HELPERS (Gemini SDK)
# ----------------------------------------------------------------------

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    
    # Initialize client once for synchronous calls
    # NOTE: Assuming GOOGLE_API_KEY is defined in the execution environment
    try:
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    except AttributeError:
        # Fallback if genai.Client doesn't support api_key argument directly
        client = genai.Client()
    
    # REMOVED get_async_gemini_client() and all async wrappers/calls

    def _img_to_data_url(path: str) -> str:
        """Converts a local file path to a base64 data URL."""
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{b64}"
        except FileNotFoundError:
            # Return a minimal placeholder data URL on failure
            return "data:image/png;base64," 

    # --- REVISED convert_to_gemini_part (FIXED) ---
    def convert_to_gemini_part(part: Dict[str, Any]) -> types.Part:
        """Converts the pipeline's image_url or text format to the Gemini SDK's Part objects."""
        if part.get("type") == "image_url":
            data_url = part["image_url"]["url"]
            if data_url.startswith("data:"):
                try:
                    mime_type = data_url.split(":")[1].split(";")[0]
                    b64_data = data_url.split(",", 1)[1]
                    data_bytes = base64.b64decode(b64_data)
                    
                    # --- FIX: Removed the unsupported media_resolution field ---
                    return types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=data_bytes
                        )
                        # REMOVED: media_resolution={"level": "media_resolution_high"} 
                    )
                except Exception as e:
                    raise ValueError(f"Failed to decode data URL to Part: {e}")
        elif part.get("type") == "text":
            return types.Part.from_text(text=part["text"])
        
        raise TypeError(f"Unsupported part type in convert_to_gemini_part: {part.get('type')}")
    
    def _sync_client_chat(model: str, contents: List[types.Part], is_json: bool, thinking_level: str = "low") -> str:
        """Synchronous API call wrapper for the Gemini SDK with thinking_level."""
        if model == "gemini-3-flash-preview":
            thinking_level = "low"

        config_args = {'thinking_config': types.ThinkingConfig(thinking_level=thinking_level)}
        
        if is_json:
            config_args['response_mime_type'] = "application/json"
        
        config = types.GenerateContentConfig(**config_args)
        
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        return response.text
    
    # Synchronous Chat Wrapper
    def gemini_chat(model: str, messages: List[Dict[str, Any]], is_json=False, thinking_level: str = "low") -> str:
        """Synchronous Gemini Chat Wrapper."""
        content_parts = messages[0].get("content", [])
        converted_parts = [convert_to_gemini_part(p) for p in content_parts]
        return _sync_client_chat(model, converted_parts, is_json, thinking_level)
    

except ImportError:
    print("FATAL: google-genai library not found. Please install it.")
    class DummyClient:
        def chat(self, *args, **kwargs): raise NotImplementedError("google-genai required.")
    client = DummyClient()
    _img_to_data_url = lambda x: f"data:image/png;base64,{x}"
    gemini_chat = client.chat
    # Define dummy functions for safety
    def _sync_client_chat(*args, **kwargs): raise NotImplementedError("google-genai required.")


# ================================================================
# Step 1: Interval and Axis Extraction (Synchronous)
# ================================================================
def _img_to_data_url(path: str) -> str:
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# ---------- Parsing the Step 1 output ----------
# helpers_llm_pipeline.py
import re, json
from typing import List, Tuple, Dict, Any, Optional

_RX_JSON = re.compile(r"=== INTERVALS \(JSON\) ===\s*([\s\S]+?)\s*=== INTERVALS \(CSV\) ===", re.I)
_RX_AXIS = re.compile(r"=== AXIS_INFO \(JSON\) ===\s*([\s\S]+?)\s*=== DESCRIPTION ===", re.I)
_RX_DESC = re.compile(r"=== DESCRIPTION ===\s*([\s\S]+?)\Z", re.I)

def parse_step1_output(text: str) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    Extracts and parses Step 1 output sections:
        INTERVALS (JSON)
        AXIS_INFO (JSON)   [optional]
        DESCRIPTION
    Returns (intervals_list, description_str, axis_info_dict).
    This tolerant version allows variable spacing, missing === lines,
    and mixed capitalization.
    """

    # Normalize newlines for safety
    text = text.strip().replace('\r\n', '\n')

    # --- Flexible regex patterns ---
    RX_JSON = re.compile(
        r"=*\s*INTERVALS\s*\(JSON\)\s*=*\s*(\[.*?\])(?=\n\s*(=+|AXIS_INFO|DESCRIPTION|$))",
        re.S | re.I,
    )
    RX_AXIS = re.compile(
        r"=*\s*AXIS[_\s-]*INFO\s*\(JSON\)\s*=*\s*(\{.*?\})(?=\n\s*(=+|DESCRIPTION|$))",
        re.S | re.I,
    )
    RX_DESC = re.compile(
        r"=*\s*DESCRIPTION\s*=*\s*(.*)$",
        re.S | re.I,
    )

    # --- INTERVALS ---
    m_json = RX_JSON.search(text)
    if not m_json:
        raise ValueError("Could not find 'INTERVALS (JSON)' block in Step 1 output.")
    try:
        intervals = json.loads(m_json.group(1).strip())
    except Exception as e:
        raise ValueError(f"Failed to parse INTERVALS JSON: {e}")

    # --- AXIS_INFO (optional) ---
    axis_info: Dict[str, Any] = {}
    m_axis = RX_AXIS.search(text)
    if m_axis:
        try:
            axis_info = json.loads(m_axis.group(1).strip())
        except Exception:
            axis_info = {}

    # --- DESCRIPTION ---
    m_desc = RX_DESC.search(text)
    if not m_desc:
        raise ValueError("Could not find 'DESCRIPTION' block in Step 1 output.")
    description = m_desc.group(1).strip()

    # --- Normalize axis_info structure ---
    def _norm_axis(d):
        if not isinstance(d, dict): return {}
        return {
            "label": d.get("label"),
            "type":  d.get("type"),
            "range": d.get("range") if isinstance(d.get("range"), (list, tuple)) else None,
            "ticks": d.get("ticks") if isinstance(d.get("ticks"), (list, tuple)) else [],
        }

    axis_info = {
        "x_axis": _norm_axis(axis_info.get("x_axis", {})),
        "y_axis": _norm_axis(axis_info.get("y_axis", {})),
    }

    return intervals, description, axis_info

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

            
def interp_tick_to_value(
    tick: float,
    tick_cal: dict,
    axis_cal,
    offset_px: float,
):
    """
    Convert an LLM-predicted tick index into a real axis value.

    tick:        fractional tick index (e.g. 3.1)
    tick_cal:    {"a": float, "b": float, "n_ticks": int}
                 such that pixel_local = a * tick + b
    axis_cal:    AxisCal instance (linear, logK, etc.)
    offset_px:   crop offset in absolute image pixels (main.x0 or main.y0)

    Returns:
        real value on the axis
    """
    # --- tick → local pixel ---
    px_local = tick_cal["a"] * tick + tick_cal["b"]

    # --- local → absolute pixel ---
    px_abs = px_local + offset_px

    # --- absolute pixel → real value ---
    return axis_cal.p2v(px_abs)

def run_step1_interval_brief(chart_path: str, vibe_grid_note: Optional[str] = None, axis_ocr_note: Optional[str] = None, model='gemini-3-pro-preview') -> tuple[list[dict], str, dict]:
    """Calls Gemini Pro for initial chart parsing."""
    print(f"\n--- Running Step 1 with {model} ---")
    img_url = _img_to_data_url(chart_path)
    
    parts = [
        {"type": "text", "text": prompt_agent_1},
        {"type": "image_url", "image_url": {"url": img_url}}
    ]
    if vibe_grid_note: parts.append({"type": "text", "text": f"\n(vibe_grid)\n{vibe_grid_note}"})
    if axis_ocr_note: parts.append({"type": "text", "text": f"\n(axis_ocr)\n{axis_ocr_note}"})

    # Use synchronous chat function
    raw_output = gemini_chat(model=model, messages=[{"role": "user", "content": parts}], is_json=False)                                                                                                                                                                                                                                                                                   
    # Use dummy parser here (replace with your robust one)
    intervals, description, axis_info = parse_step1_output(raw_output) 
    return intervals, description, axis_info

# =====================================================================================
# 2. STEP 2: Using Step 1 output to compute axis calibration and 
# =====================================================================================
def run_step2(
    chart_path: str,
    intervals: List[Dict[str, Any]],
    out_dir: str = "out_interval",
    px_step: int = 15,
    font_scale: float = 0.22,
    thickness: int = 0,
    # optional fallbacks if OCR returns <2 ticks:
    x_limits: Optional[tuple[float, float]] = None,
    y_limits: Optional[tuple[float, float]] = None,
    llm_axis_info: Optional[Dict[str, Any]] = None,
    res: Dict[str, Any] = None,
    x_cal: Optional[AxisCal] = None,
    y_cal: Optional[AxisCal] = None,
) -> List[str]:
    """
    If x/y axis calibration isn't available, compute it.
    Then use it to generate label-removed, grid-enhanced L-shaped crops for each interval.

    The goal is to help the LLM find the right numeric values using each intervals to zoom in.

    Returns:
        List[str]: file paths of saved composite crops.
    Additionally saves calibration info as JSON beside the crops for later numeric mapping.
    """

    # -------------------------------------------------------------------------
    # 1) Prepare output folder
    # -------------------------------------------------------------------------
    crops_dir = os.path.join(out_dir, "crops")
    if not os.path.exists(crops_dir):
        os.makedirs(crops_dir, exist_ok=True)    
    empty_folder(crops_dir)
    print(f"chart_path  {chart_path}")

    # -------------------------------------------------------------------------
    # 2) Build crops with calibration-aware function
    # -------------------------------------------------------------------------
    crops = make_interval_crops(
        image_path=chart_path,
        intervals=intervals,
        out_dir=out_dir,
        llm_axis_info=llm_axis_info,
        x_limits_override=x_limits,
        y_limits_override=y_limits,
        font_scale=font_scale,
        thickness=thickness,
        res=res,
        cal_x=x_cal,
        cal_y=y_cal,
    )

    # -------------------------------------------------------------------------
    # 3) Save calibration data for later use (e.g., Step 2 or benchmarking)
    # -------------------------------------------------------------------------
    try:
        cal_path = os.path.join(out_dir, "interval_calibration.json")
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(crops.calibration, f, indent=2)
        print(f"Calibration info saved → {cal_path}")
    except Exception as e:
        print(f"Warning: could not save calibration info ({e})")

    # -------------------------------------------------------------------------
    # 4) Gather crop file paths (sorted for consistent order)
    # -------------------------------------------------------------------------
    files = sorted([
        os.path.join(crops_dir, f)
        for f in os.listdir(crops_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print(f"✅ Generated {len(files)} crops in {crops_dir}")
    if hasattr(crops, "calibration"):
        print("Calibration summary:",
              {k: {kk: round(vv, 5) if isinstance(vv, (float,int)) else vv
                   for kk,vv in v.items()} if isinstance(v, dict) else v
               for k,v in crops.calibration.items()})

    return files, crops

# ================================================================
# 3. STEP 3: Concurrent Crop Analysis (Threading)
# ================================================================
def analyze_single_image_async(
    model: str,
    img_path: str,
    global_image_path: str, # Keeping parameter for signature compatibility, but unused
    series_descriptions: List[Any] | None = None,
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
- Be carefuyl to not get tricked by grid lines, confidence intervals, or other chart elements. 
- Output at least 5 points per series if visible.
- STRICT JSON OUTPUT ONLY.

OUTPUT FORMAT:
[
  {{
    "id": "<series_id>",
    "points": [
      {{ "x_tick": <float>, "y_tick": <float> }}
    ],
    "points_count": <int>
  }}
]
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
    
def analyze_plot_points_with_llm_concurrent(
    folder_path: str,
    image_path: str,
    meta_data: List[Dict[str, Any]],
    model: str = "gemini-3-pro-preview",
    max_concurrent: int = 10,
    series_descriptions: List[str] = [],
) -> Dict[str, Any]:
    """
    Concurrent batch processing of crops using ThreadPoolExecutor
    with LINEAR tick-space → pixel → real-value conversion.
    """
    from pathlib import Path
    import concurrent.futures
    import time

    # --- Build lookup from filename → calibration info ---
    crop_lookup = {
        Path(entry["image_path"]).name: {
            "bbox": entry["bbox_main_xyxy"],
            "tick_cal": entry["tick_calibration"],
            "axis_cal": entry["axis_calibration"],
        }
        for entry in meta_data
    }

    tasks_args = []
    print(f"🖼️ Found {len(meta_data)} crop(s) to process concurrently with {max_concurrent} threads.")

    for entry in meta_data:
        p = Path(entry["image_path"])
        name = p.name
        if name not in crop_lookup:
            print(f"⚠️ No calibration found for {name}, skipping.")
            continue

        # (model, crop_path, full_image_path, series_descriptions)
        tasks_args.append((model, str(p), image_path, series_descriptions))

    results = []
    t0 = time.time()

    # --- Run LLM inference concurrently ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_name = {
            executor.submit(analyze_single_image_async, *args): args[1]
            for args in tasks_args
        }

        for future in concurrent.futures.as_completed(future_to_name):
            img_path = future_to_name[future]
            try:
                result = future.result()
                results.append(result)
                print(f"✅ Finished {Path(img_path).name} in {time.time() - t0:.2f}s")
            except Exception as exc:
                print(f"🚨 {Path(img_path).name} failed: {exc}")
                results.append({
                    "filename": Path(img_path).name,
                    "error": str(exc),
                })

    # --- Post-process: tick → pixel → real value ---
    for img_entry in results:
        name = img_entry.get("filename")
        if name not in crop_lookup:
            continue
        if "series" not in img_entry:
            continue

        info = crop_lookup[name]
        bbox = info["bbox"]
        tick_cal = info["tick_cal"]
        axis_cal = info["axis_cal"]
        # Reconstruct AxisCal objects
        cal_x = AxisCal(
            a=axis_cal["x"]["a"],
            b=axis_cal["x"]["b"],
            axis="x",
            mode=axis_cal["x"].get("mode", "linear"),
        )
        cal_y = AxisCal(
            a=axis_cal["y"]["a"],
            b=axis_cal["y"]["b"],
            axis="y",
            mode=axis_cal["y"].get("mode", "linear"),
        )

        x0, y0 = bbox[0], bbox[1]
        for s in img_entry.get("series", []):
            for pt in s.get("points", []):
                xt = pt.get("x_tick")
                yt = pt.get("y_tick")
                if xt is not None:
                    pt["x_real"] = interp_tick_to_value(
                        tick=float(xt),
                        tick_cal=tick_cal["x"],
                        axis_cal=cal_x,
                        offset_px=x0,
                    )
                if yt is not None:
                    pt["y_real"] = interp_tick_to_value(
                        tick=float(yt),
                        tick_cal=tick_cal["y"],
                        axis_cal=cal_y,
                        offset_px=y0,
                    )

    return {"images": results}

# ----------------------------------------------------------------------
# 4. STEP 2b: Final Numeric Read (Sync)
# ----------------------------------------------------------------------

import json
import math
from collections import defaultdict

def _safe_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def flatten_concurrent_images_to_series(result_concurrent: dict) -> list:
    """
    Convert {"images":[{filename, series:[{label, points:[{x_real,y_real,...}]}]}]}
    into canonical series list:
      [{"id": str, "points": [{"x": float, "y": float,, ...}, ...]}]
    """
    images = result_concurrent.get("images", []) or []
    by_label = defaultdict(list)

    for img_entry in images:
        fname = img_entry.get("filename") or img_entry.get("image_path") or "unknown_crop"

        for s in img_entry.get("series", []) or []:
            UNASSIGNED_LABEL = "__unassigned__"
            label = s.get("id") or s.get("label")
            label = str(label).strip() if label is not None else ""
            if not label:
                continue

            if not label:
                continue

            for pt in (s.get("points", []) or []):
                xr = _safe_float(pt.get("x_real"))
                yr = _safe_float(pt.get("y_real"))
                if xr is None or yr is None:
                    continue

                by_label[label].append({
                    "x": xr,
                    "y": yr,
                    # keep raw tick info if you want to debug
                    "x_tick": pt.get("x_tick"),
                    "y_tick": pt.get("y_tick"),
                })

    out = []
    for label, pts in by_label.items():
        out.append({"id": label, "points": pts})
    return out


import random

def enforce_strict_x_monotonicity(
    series: list,
    eps: float = 1e-6,
    seed: int | None = 0,
) -> list:
    """
    Enforce strictly increasing x per series.
    - If the FIRST point(s) share the same x, randomly jitter all but one.
    - Otherwise, shift forward minimally relative to the previous confirmed x.
    """
    rng = random.Random(seed)
    out = []

    for s in series:
        pts = list(s.get("points", []))
        if len(pts) <= 1:
            out.append(s)
            continue

        # Stable sort: preserves original order for equal x
        pts_sorted = sorted(enumerate(pts), key=lambda t: (t[1]["x"], t[0]))

        fixed = []
        last_x = None
        i = 0

        while i < len(pts_sorted):
            # Collect run of equal-x points
            j = i
            x0 = float(pts_sorted[i][1]["x"])
            group = []

            while j < len(pts_sorted) and float(pts_sorted[j][1]["x"]) == x0:
                group.append(pts_sorted[j][1])
                j += 1

            if last_x is None:
                # FIRST group → random epsilon jitter
                rng.shuffle(group)
                for k, p in enumerate(group):
                    fixed.append({
                        "x": x0 + k * eps,
                        "y": float(p["y"]),
                    })
                last_x = fixed[-1]["x"]
            else:
                # Subsequent groups → monotonic forward shift
                for p in group:
                    x = max(x0, last_x + eps)
                    fixed.append({
                        "x": x,
                        "y": float(p["y"]),
                    })
                    last_x = x

            i = j

        out.append({
            "id": s["id"],
            "points": fixed
        })

    return out

def concatenate_series(series_list: list) -> list:
    """
    No deduplication, no clustering.
    Just normalize structure.
    """
    out = []
    for s in series_list:
        pts = s.get("points", [])
        if not pts:
            continue
        out.append({
            "id": s.get("id", "series_1"),
            "points": pts
        })
    return out

# ----------------------------------------------------------------------
# 5. MAIN EXECUTION BLOCK (Updated)
# ----------------------------------------------------------------------

def run_full_pipeline_async(image_path: str,
                                  crops: Optional[IntervalCropResult] = None,
                                  model:str='gemini-3-pro-preview',
                                  series_descriptions: List[str]=[]) -> dict:
    """Executes the entire data extraction pipeline using Gemini 3.0 Pro."""
    time_start = time.time()

    # Use the passed image path instead of CHART_PATH
    CHART_PATH = image_path
    MODEL_PRO = model
    MODEL_FLASH = "gemini-3-flash-preview" 
    MODEL_ASYNC = MODEL_PRO

    t0 = time.time()
    description = "No description available."
    if not os.path.exists(CHART_PATH):
        print(f"FATAL: Chart image not found at '{CHART_PATH}'. Please update CHART_PATH.")
        return

    if crops is None:
        # --- 1. STEP 1: Interval Finding & Description (Sync) ---
        try:
            intervals, description, llm_axis_info = run_step1_interval_brief(chart_path=CHART_PATH, model=MODEL_PRO)
            time_step1 = time.time() - time_start
            print(f"✅ Step 1 Success. Time: {time_step1:.1f}s")
        except Exception as e:
            print(f"FATAL: Step 1 failed: {e}")
            return
        print(description)
        
        # --- 2. Step 2: Crop Generation (Dummy) ---
        time_step2_start = time.time()
        files, crops = run_step2(chart_path=CHART_PATH, intervals=intervals, llm_axis_info=llm_axis_info)
        time_step2 = time.time() - time_step2_start
        print(f"✅ Step 2 Success. Time: {time_step2:.1f}s")
    
    # --- 3. STEP 3: CONCURRENT CROP ANALYSIS (Threading) ---
    time_step3_start = time.time()
    try:
        # Call the new synchronous, concurrent function
        result_concurrent = analyze_plot_points_with_llm_concurrent(
            folder_path="out_interval/crops", 
            image_path=image_path,
            meta_data=crops.meta, 
            model=MODEL_ASYNC, 
            max_concurrent=10, # Running up to 10 threads simultaneously
            series_descriptions=series_descriptions
        )
        time_step3 = time.time() - time_step3_start
        print(f"✅ Step 3 Success (Concurrent/Threading). Time: {time_step3:.1f}s")
    except Exception as e:
        print(f"WARNING: Concurrent crop analysis failed, error: {e}")
    
    # ---------------------------
    # MERGE anchors from concurrent output
    # ---------------------------
    flat_series = flatten_concurrent_images_to_series(result_concurrent)

    merged_series = concatenate_series(flat_series)
    merged_series = enforce_strict_x_monotonicity(
        merged_series,
        eps=1e-6,
        seed=0,   # set to None if you truly want nondeterministic
    )

    time_postprocess = time.time()
    print(f"✅ Anchor Merging Success. Time: {time_postprocess - time_step3_start:.1f}")
    print(len(merged_series), "series after merging.")

    completed_series = merged_series
    final_output = {
        "series": [
            {
                "id": s.get("id"),
                "label": s.get("label") or s.get("id"),
                "points": [
                    {"x": float(p["x"]), "y": float(p["y"])}
                    for p in (s.get("points") or [])
                    if _safe_float(p.get("x")) is not None
                    and _safe_float(p.get("y")) is not None
                ],
            }
            for s in completed_series
            if s.get("id") or s.get("label")
        ]
    }

    return final_output