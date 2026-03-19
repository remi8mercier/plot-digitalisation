prompt = """
You are an expert system that extracts NUMERICAL data from 2D scientific plots.
You must return ALL visible data series from the given image.

Core rules:
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
        {"x": <float>, "y": <float>,
        ...
      ]
    },
    ...
  ]
}

Additional constraints:
- Include ALL visible series.
- For continuous curves, use ≥ 20 points if readable.
- For categorical plots, output exactly one series per category.
- Do NOT include explanations, placeholders, or any text outside the JSON.
"""


import os
from dotenv import load_dotenv
from openai import OpenAI

import base64

dataset_dir = "data/ExtractedDataMatthias/"
name_g1 = "Data-Naik2016-6MonthPSA-Strat-MetastasisFreeSurvival-as-cLMA-G8dv"
name_g3 = "Data-Naik2016-6MonthPSA-Strat-ProstateCancerSpecificMortality-as-AXhW-IH9H"
name_g5 = "Data-ProstateCancer-BiochemicalRecurrence-TestosteroneNadir-as-kPUM-ja2E"
image_path_g3 = dataset_dir + name_g3 + "/" + name_g3+ ".png"
image_path_base = "graphs/graph_test1.jpeg"
image_path_3 = "graphs/graph1.png"
image_path_g6 = "graphs/servier.png"
name_g7 = "Data-ProstateCancer-OverallSurvival-SerumTestosteroneNadir-20ngPerDl-as-j2Uo-EdA3"
name_g9 = "Data-ProstateCancer-TimeToCastrationResistance-CastrationLevel-as-G8iP-2p8H"
name_g10 = "Data-ProstateCancer-10YearMetastasis-TestosteroneNadir-as-HUFM-k4RT"
name_g11 = "Data-ProstateCancer-OverallSurvival-TestosteroneSuppression-as-gIUH-EPAn"
name_g12 = "Data-ProstateCancer-CastrationLevel-OverallSurvival-LocallyAdvanced-Metastatic-as-HN9j-M4NK"
name_g13 = "Data-ProstateCancer-CastrationLevel-TimeToProgression-LocallyAdvanced-Metastatic-as-aWvR-2wlo"
name_4 = "Data-ProstateCancer-CastrationLevel-TimeToProgression-BiochemicalRecurrence-as-4dag-RoWD"
name_g14 =  "Data-as-YpSF-UNAQ"
name_g15 = "Data-as-CdWI-s6OK"
name_g16 = "/Data-as-cfVO-vMjk"
name_g17 = "Data-as-8975-q7O8"
name_g18 = "Data-as-b98m-TsDT"
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
image_path_g17 = dataset_dir + name_g17 + "/" + name_g17 + ".png"
image_path_g18 = dataset_dir + name_g18 + "/" + name_g18 + ".png"
e_graph1 = "graphs/e_graph1.png"
e_graph2 = "graphs/e_graph2.png"
e_graph3 = "graphs/e_graph3.png"
e_graph4 = "graphs/e_graph4.png"
r_graph1 = "graphs/graph3.webp"
theo_1 = "graphs/theo_1.webp"
theo_2 = "graphs/theo_2.png"


load_dotenv()

# Access your constant, for example:
api_key = os.getenv("OPENAI_API_KEY")
# Send request to gpt-5
def single_step_simple(image_path: str, prompt: str=prompt) -> None:
    """
    This function sends a single request to GPT-5.2 with the given image and prompt, and returns the raw output as a string.
    """
    client = OpenAI(api_key=api_key)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    image_data_url = f"data:image/png;base64,{b64}"

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "You are an assistant that extracts raw numerical data from images of plots, using an accurate description of the image."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]}
        ]
        )

    # Extract JSON string from response
    raw_output = response.choices[0].message.content
    return raw_output

if __name__ == "__main__":
    #test_image_path = "graphs/graph1.png"  # Replace with your test image path
    test_image_path = image_path_g7
    #test_image_path = r_graph1 

    from time import time
    start_time = time()
    print(test_image_path)
    raw_output = single_step_simple(test_image_path)
    end_time = time()
    print(f"Execution time: {end_time - start_time} seconds")
    print(raw_output)