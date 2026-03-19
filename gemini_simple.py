import json
import os
import PIL.Image
from google import genai
from google.genai import types
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()
# Access your constant, for example:
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Configuration ---
# NOTE: Ensure you have the 'google-genai' library installed: pip install google-genai pillow
# and your GEMINI_API_KEY environment variable is set (or pass it to genai.Client).
# This example assumes the API key is set as an environment variable (GOOGLE_API_KEY or GEMINI_API_KEY).
try:
    client = genai.Client(http_options={'api_version': 'v1alpha'})
except Exception as e:
    # Handle the case where the client cannot be initialized (e.g., missing API key)
    print(f"Error initializing Gemini client. Please ensure your API key is configured. Error: {e}")
    # You might want to exit or handle this more robustly
    exit()


# The original prompt remains the same to ensure strict JSON output and rules.
prompt = """
You are an expert system that extracts NUMERICAL data from 2D scientific plots.
You must return ALL visible data series from the given image.

⚠️ Core rules:
- Do NOT guess, extrapolate, or invent data.
- If a value is uncertain, OMIT it.
- Never hallucinate hidden or occluded data.
- Do NOT assume missing structure.
- Work in three internal steps (do NOT show them).

────────────────────────────────────────
INTERNAL LOGIC (DO NOT OUTPUT)
────────────────────────────────────────

1) Identify the plot type:
   - Continuous plot:
     Lines, curves, step-like curves, scatter plots with numeric axes.
   - Categorical plot:
     Bars, grouped bars, box-like summaries, or mixed bar/marker plots where
     the x-axis represents discrete categories rather than a numeric scale.

2) Extract data according to plot type:

   A) Continuous plots
   - Identify all visible data series.
   - For discrete markers: extract all visible points.
   - For continuous curves:
     - Sample at least 20 points if readable.
     - Include local extrema, inflection points, step changes, and uniform samples.
   - Preserve visual ordering and series separation.
   - Do NOT fabricate smoothness or continuity.

   B) Categorical plots
   - Treat EACH CATEGORY as its OWN DATA SERIES.
   - If there are N categories, output exactly N separate series.
   - Each series represents values associated with that category.
   - Each point corresponds to a categorical interval or bin, not a continuous axis.
   - Do NOT interpolate between categories.
   - Do NOT assume linear spacing between categories.

3) Validation:
   - Ensure all extracted points are visually supported.
   - Reject perfectly uniform numeric sequences unless they clearly correspond
     to repeated visible values.
   - Maintain consistency within each series.
   - Keep all points sorted by increasing x-value or category order.

────────────────────────────────────────
OUTPUT FORMAT (STRICT JSON ONLY)
────────────────────────────────────────

{
  "series": [
    {
      "label": "<legend name if visible else 'series_N'>",
      "points": [
        {"x": <float>, "y": <float>, "confidence": <float 0..1>},
        ...
      ]
    },
    ...
  ]
}

📌 Additional constraints:
- Include ALL visible series.
- For continuous curves, use ≥ 20 points if readable.
- For categorical plots, output exactly one series per category.
- Do NOT include explanations, placeholders, or any text outside the JSON.
"""

def extract_plot_data_gemini_streaming(image_file_path: str) -> str:
    """
    Streaming request to Gemini 3 Pro (v1alpha API):
    - High-quality image input
    - No thinking steps
    - Final text-only JSON output collected
    """

    import base64
    import PIL.Image
    import io

    prompt_text = prompt
    # 1. Load image file and base64 encode it
    try:
        with open(image_file_path, "rb") as f:
            img_bytes = f.read()
    except Exception as e:
        print(f"Image load error: {e}")
        return ""

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # 2. Prepare content: text + high-resolution image
    contents = [
        types.Content(
            parts=[
                types.Part(text=prompt_text),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png",
                        data=base64.b64decode(img_b64),
                    ),
                    media_resolution={"level": "media_resolution_high"}  # HQ VISION
                )
            ]
        )
    ]

    config_args = {'thinking_config': types.ThinkingConfig(thinking_level="low")}
    if True:
        config_args['response_mime_type'] = "application/json"
    
    config = types.GenerateContentConfig(**config_args)

    # 4. Create streaming request
    response_stream = client.models.generate_content_stream(
        model="gemini-3-pro-preview",
        contents=contents,
        config=config
    )

    accumulated_json = ""

    # 5. Collect ONLY the actual output (no thoughts available now)
    for chunk in response_stream:
        if not chunk.candidates:
            continue
        parts = chunk.candidates[0].content.parts
        if not parts:
            continue

        for part in parts:
            if part.text:
                #print(part.text, end="", flush=True)
                accumulated_json += part.text

    #print("\n")
    data = json.loads(accumulated_json)
    return data

r_graph1 = "graphs/graph3.webp"
cropped_image_path = "out_interval/crops/ival_04.png"

if __name__ == "__main__":
    # --- Execute the pipeline ---
    IMAGE_FILE_PATH = r_graph1
    if os.path.exists(IMAGE_FILE_PATH):
        raw_output = extract_plot_data_gemini_streaming(IMAGE_FILE_PATH)
        
        print("\n" + "="*50)
        print("FINAL MODEL OUTPUT (Raw JSON String):")
        print("="*50)
        print(raw_output)
        if raw_output:
            # Optional: Validate and pretty-print the final accumulated JSON output
            try:
                if isinstance(raw_output, str):
                    parsed_json = json.loads(raw_output)
                else:
                    parsed_json = raw_output
                #
                print(json.dumps(parsed_json, indent=2))
            except json.JSONDecodeError:
                print("Error: The accumulated text was not valid JSON.")
                print("--- Full Text Received ---")
                print(raw_output)
                
        print("="*50)
    else:
        print(f"\n⚠️ Please set IMAGE_FILE_PATH to a valid file path (e.g., '{IMAGE_FILE_PATH}') and run again.")