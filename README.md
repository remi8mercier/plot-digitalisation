Python >= 3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# AI Plot Digitizer (Gemini/GPT) — LLM + OCR + Calibration + (Optional) KM Snapping

This repo extracts numeric data from scientific plots by combining:

- **LLM planning** (Gemini / GPT-5/5.2): routing + axis/tick inference + interval proposals
- **OCR tick mining** (EasyOCR): exhaustive tick label detection around axes (incl. rotations)
- **Calibration**: robust pixel↔value mapping (linear/log + optional breaks)
- **Targeted interval crops**: send small “interesting” regions to the LLM instead of the full plot
- **Deterministic extraction tools**: optional Kaplan–Meier (KM) curve snapping + CV helpers
- **Post-processing**: fixes duplicate-x artifacts and preserves step discontinuities

The system is built around **two main entry scripts**:
- `gemini_pipeline.py` (unified + supports KM snapping)
- `gemini_pipeline_km_snapping.py` (unified but **no** KM snapping tool)

---

## How the system is designed

### Core idea: split responsibilities
The LLM is used for **semantic / structural decisions**:
- decide whether the plot is categorical vs continuous vs KM-like
- describe series (ids, colors, line styles, markers)
- propose *intervals* where local detail matters
- infer axis ticks/ranges (as a plan)

Deterministic code is used for **geometry / numeric correctness**:
- detect axes geometry
- mine tick labels with OCR + rotations
- compute pixel↔value calibration
- crop the image consistently in calibrated coordinates
- (optional) snap survival curves to pixels
- enforce continuity and monotonic-x in final output

---

## Entry points

### 1) `gemini_pipeline.py` — Unified extractor (includes KM snapping)
This is the main pipeline with a router that can choose:
- `categorical_tool`
- `crop_generation_tool`
- `km_detection_tool` (Kaplan–Meier snapping)

It runs **two Gemini calls concurrently**:
1) **Calibration planner**: produces axis info (ticks, type, breaks)
2) **Router planner**: selects which tool/pipeline to run

If calibration fails:
- categorical extraction is still allowed
- otherwise it falls back to LLM-only extraction (`gemini_simple.py`)

### 2) `gemini_pipeline_km_snapping.py` — Unified extractor (no KM tool)
Same concurrency pattern (calibration + router), but the router can only choose:
- `categorical_tool`
- `crop_generation_tool`

No KM snapping path exists here.

---

## With vs Without KM snapping

### Without KM snapping (generic crop pipeline)
Used by:
- `gemini_pipeline_km_snapping.py`
- also used inside `gemini_pipeline.py` when router selects `crop_generation_tool`

Flow:
1) detect axes (`_detect_axes`)
2) run OCR (`process_image`) to get candidate ticks
3) calibrate (`AxisCal`) using OCR + LLM axis info
4) generate interval crops (`run_step2` / `make_interval_crops`)
5) analyze each crop asynchronously (`run_full_pipeline_async`)
6) merge crop-level results into final series
7) run post-processing to fix duplicate x

Best for:
- general line plots
- scatter plots
- non-monotonic curves
- multi-line charts where “snapping” isn’t required

---

### With KM snapping (Kaplan–Meier survival curves)
Used only by:
- `gemini_pipeline.py` when router selects `km_detection_tool`

Flow:
1) calibration as usual (axes + OCR + AxisCal)
2) LLM extracts **rough KM guidance points** (value space) per series id  
3) guidance is converted **value→pixel** using `AxisCal`
4) `km_detection_tool` performs deterministic snapping:
   - non-background mask
   - per-series exploration limits
   - joint beam search with smoothness + interaction penalties
   - median filtering
5) final points are converted back **pixel→value**

Key property:
- The LLM never outputs the final snapped curve.  
  It only gives rough guidance; the curve is extracted from pixels.

Best for:
- monotonic step-like survival curves
- plots with heavy overlap and censor markers
- cases where LLM-only digitization is unstable

---

## File-by-file walkthrough (what each file does and how it’s used)

### `gemini_pipeline.py`
Unified orchestrator **with** KM snapping.

Main function:
- `run_unified_plot_extractor(image_path, model=..., saving=...)`

Key responsibilities:
- loads image
- attempts axis detection (`_detect_axes`)
- starts OCR concurrently (`process_image`)
- runs Gemini concurrently:
  - calibration planner (calls `axis_properties_tool`)
  - routing planner (calls one tool: categorical/crop/km)
- executes the selected tool:
  - categorical → `categorical_graph_extraction_tool`
  - crop-based → `_execute_crop_gen_tool` then `run_full_pipeline_async`
  - KM → `_execute_km_tool` then `km_detection_tool`
- post-processes results (`expand_duplicate_x_with_continuity_final`)
- optionally saves overlays + CSVs

---

### `gemini_pipeline_km_snapping.py`
Unified orchestrator **without** KM snapping.

Main function:
- `run_unified_plot_extractor(image_path, model=..., saving=...)`

Differences vs `gemini_pipeline.py`:
- router only supports `categorical_tool` and `crop_generation_tool`
- no `km_detection_tool` path
- if crop tool fails, falls back to `analyze_single_image_async`

---

### “LLM simple files” (direct model calls + saving)
These are intentionally minimal wrappers: call a model with a prompt, parse JSON, save results.

- `gemini_simple.py`
  - contains `extract_plot_data_gemini_streaming(image_path)`
  - used as the *hard fallback* in `gemini_pipeline.py`

(If you also have GPT analogs, they’re the same role: minimal call → JSON → persist.)

How they’re used:
- when the structured pipeline fails (calibration/router/tool execution)
- when you want a quick one-shot digitization without crops/calibration

---

### `ocr.py`
OCR tick miner.

Main entry:
- `process_image(image_path, x_axis_y=..., y_axis_x=..., per_plot_ocr=...)`

What it does:
- searches around the axes region for text
- runs EasyOCR on multiple rotated versions of the region
- rotation is used to increase recall:
  - rotated labels become readable
  - *and* empirically, rotation also helps detect labels that are not rotated (new angle → new detection)

Output:
- a set of candidate tick labels/positions for calibration

Used by:
- both Gemini pipelines (OCR runs concurrently with LLM calibration planning)

---

### `interval_crops.py`
Calibration + crop generation primitives.

Key pieces:
- `AxisCal`: pixel↔value mapping with modes (linear/log)
- `make_interval_crops`: creates interval crops using calibration
- calibration logic that fuses:
  - OCR candidate ticks
  - LLM “true” tick values (axis plan)

Used by:
- `run_step2` / crop pipeline
- axis calibration tool (`axis_properties_tool`) to compute `AxisCal`
- KM snapping, for value→pixel and pixel→value conversions

Important design detail:
- `make_interval_crops` can be called with an **empty interval list** to produce *calibration only*.  
  This is used to separate calibration from interval generation when you want parallelism.

---

### `pipeline_core.py`
The generic numeric extraction engine.

Key entry points used by Gemini pipelines:
- `run_step2(...)`  
  Creates calibrated crops + metadata (uses `make_interval_crops` / calibration objects)
- `run_full_pipeline_async(image_path, crops, model, series_descriptions)`  
  Runs crop analysis concurrently, merges results, returns canonical series output

Used when router selects:
- `crop_generation_tool`

---

### `extraction_tools.py`
Computer vision helpers used across pipelines.

Contains (at least):
- `_detect_axes(img)`  
  returns axis positions/spans used to define OCR search regions + crop geometry
- `extract_non_background_mask(img, ...)`  
  used by KM snapping to isolate curve pixels from background

Used by:
- both pipelines (axis detection)
- KM snapping path (mask extraction + axis constraints)

---

### KM snapping implementation (in `gemini_pipeline.py`)
Key functions:
- `_execute_km_tool(...)`  
  Calls Gemini once to get rough KM guidance per series id, converts it to pixels
- `km_detection_tool(...)`  
  Deterministic snapping: builds candidates around guidance, uses beam search,
  applies smoothness and interaction penalties, emits calibrated series

This is only executed when router picks `km_detection_tool`.

---

### Output post-processing (shared)
- `expand_duplicate_x_with_continuity_final(result, eps_frac=...)`

Why it exists:
- step plots and vertical drops naturally produce duplicate-x points
- some models also emit repeated x for discontinuities

What it does:
- preserves all points
- forces strictly increasing x by adding small epsilon shifts
- resolves duplicates using continuity with previous y
- keeps step discontinuities explicit in the exported data

Used at the end of both pipelines.

---

## Output format

Both pipelines aim to return a canonical structure:

```json
{
  "series": [
    {
      "label": "Series name or category",
      "points": [{"x": 0.1, "y": 0.98}, {"x": 0.2, "y": 0.94}]
    }
  ]
}
```

## Master Thesis

This project is my master thesis. I have attached my report which awarded me a 1.3 at TU Munich (in the german grade system, a 1 is the best grade, and a 1.3 is the second best grade).
The slides for the master thesis defense are also in this repo.