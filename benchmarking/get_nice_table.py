"""
Generate a table of important metrics from a benchmark results file.
"""

import json
import numpy as np
import pandas as pd

import json

def load_results(path):
    """
    Load a file containing back-to-back JSON objects with no separators.
    Each object corresponds to one benchmark case.
    """
    results = []

    with open(path, "r") as f:
        buf = ""
        depth = 0

        for line in f:
            for ch in line:
                if ch == "{":
                    depth += 1
                if depth > 0:
                    buf += ch
                if ch == "}":
                    depth -= 1
                    if depth == 0:
                        # complete JSON object
                        results.append(json.loads(buf))
                        buf = ""

    return results

import numpy as np
def compute_stats(df, case_ids):
    """
    Policy:
    - Failure rate = (# failures) / (# total)
    - Accuracy metrics = MEAN and MEDIAN over successful figures only
    - Metrics are NaN only if there are no successful figures
    """

    sub = df[df["case_id"].isin(case_ids)]

    n_total = len(sub)
    if n_total == 0:
        return {
            "# Figures": 0,
            "Mean NMAE": np.nan,
            "Median NMAE": np.nan,
            "Mean Norm L1": np.nan,
            "Median Norm L1": np.nan,
            "Failure rate": np.nan,
        }

    failures = (~sub["success"]).sum()
    failure_rate = failures / n_total

    ok = sub[sub["success"]]
    
    if len(ok) == 0:
        mean_nmae = np.nan
        median_nmae = np.nan
        mean_l1 = np.nan
        median_l1 = np.nan
    else:
        mean_nmae = np.mean(ok["overall_nmae"])
        median_nmae = np.median(ok["overall_nmae"])
        mean_l1 = np.mean(ok["overall_integral_l1"])
        median_l1 = np.median(ok["overall_integral_l1"])

    return {
        "# Figures": n_total,
        "Mean NMAE": mean_nmae,
        "Median NMAE": median_nmae,
        "Mean Norm L1": mean_l1,
        "Median Norm L1": median_l1,
        "Failure rate": failure_rate,
    }

import pandas as pd

def to_dataframe(objs):
    """
    Convert parsed benchmark JSON objects into a per-figure DataFrame.
    """
    rows = []

    for o in objs:
        r = o["result"]
        m = r.get("metrics", {})

        rows.append({
            "case_id": r["case_id"],
            "success": r["success"],
            "overall_nmae": m.get("overall_nmae"),
            "overall_integral_l1": m.get("overall_integral_l1"),
        })

    return pd.DataFrame(rows)


def build_summary_table(df, normal_ids, km_ids, categorical_ids):
    """Build a summary table with overall stats and stats by chart type."""
    rows = []

    # Overall
    rows.append({
        "Chart type": "Overall",
        **compute_stats(
            df,
            normal_ids + km_ids + categorical_ids
        )
    })

    rows.append({
        "Chart type": "Normal",
        **compute_stats(df, normal_ids)
    })

    rows.append({
        "Chart type": "KM",
        **compute_stats(df, km_ids)
    })

    rows.append({
        "Chart type": "Categorical",
        **compute_stats(df, categorical_ids)
    })

    return pd.DataFrame(rows)

NORMAL_NAMES = ["Data-as-8IYI-AGTa",
                "Data-as-8975-q7O8",
                "Data-as-b98m-TsDT",
                "Data-as-bU4o-xO6R",
                "Data-as-CdWI-s6OK",
                "Data-as-cfVO-vMjk",
                "Data-as-ngkV-QcWm",
                "Data-as-y2nv-TPY1",
                "Data-as-ylJc-Vi0h",
                "Data-as-YpSF-UNAQ",
                ]
KM_NAMES = ["Data-Naik2016-6MonthPSA-Strat-MetastasisFreeSurvival-as-cLMA-G8dv",
            "Data-Naik2016-6MonthPSA-Strat-ProstateCancerSpecificMortality-as-AXhW-IH9H",
            "Data-ProstateCancer-BiochemicalRecurrence-TestosteroneNadir-as-kPUM-ja2E",
            "Data-ProstateCancer-CastrationLevel-OverallSurvival-BiochemicalRecurrence-as-KBNL-4YCX",
            "Data-ProstateCancer-CastrationLevel-OverallSurvival-LocallyAdvanced-Metastatic-as-HN9j-M4NK",
            "Data-ProstateCancer-CastrationLevel-TimeToProgression-BiochemicalRecurrence-as-4dag-RoWD",
            "Data-ProstateCancer-CastrationLevel-TimeToProgression-LocallyAdvanced-Metastatic-as-aWvR-2wlo",
            "Data-ProstateCancer-DiseaseSpecificMortality-FirstLineLocalized-Radiotherapy-Bryant2018-as-PBP9-LpbG",
            "Data-ProstateCancer-DiseaseSpecificSurvival-TestosteroneSuppression-as-pk2y-g1My",
            "Data-ProstateCancer-Metastases-TestosteroneNadir-as-DLEW-Q14P",
            "Data-ProstateCancer-OverallSurvival-SerumTestosteroneNadir-8ngPerDl-as-mu6F-pD1H",
            "Data-ProstateCancer-OverallSurvival-SerumTestosteroneNadir-20ngPerDl-as-j2Uo-EdA3",
            "Data-ProstateCancer-OverallSurvival-SerumTestosteroneNadir-20ngPerDl-SixMonths-as-Byus-Y7YM",
            "Data-ProstateCancer-OverallSurvival-TestosteroneResponseNadir-as-7x4S-GPr7Data-ProstateCancer-OverallSurvival-TestosteroneResponseNadir-as-7x4S-GPr7",
            "Data-ProstateCancer-OverallSurvival-TestosteroneSuppression-as-gIUH-EPAn",
            "Data-ProstateCancer-ProstateCancerSpecificMortality-TestosteroneNadir-as-JAS0-KOng",
            "Data-ProstateCancer-Survival-3MonthPsaResponse-as-FJHE-YyOA",
            "Data-ProstateCancer-Survival-7MonthPsaResponse-as-7II1-HQ1y",
            "Data-ProstateCancer-TimeToBiochemicalFailure-TestosteroneMedian-as-V4ML-TC46",
            "Data-ProstateCancer-TimeToCastrationResistance-CastrationLevel-as-G8iP-2p8H",
            "Data-ProstateCancer-TimeToCastrationResistance-TestosteroneNadir-as-KPBC-NCfw",
            "Data-ProstateCancer-TimeToCauseSpecificSurvival-TestosteroneNadir-as-zQ4P-rWVB"]
CATEGORICAL_NAMES = [
                "Data-as-Atl6-ihFg",
                "Data-ProstateCancer-10YearBiochemicalRecurrence-TestosteroneNadir-as-FRp3-CyQW",
                "Data-ProstateCancer-10YearMetastasis-TestosteroneNadir-as-HUFM-k4RT",
                ]

if __name__ == "__main__":
    INPUT_FILE_PATH = "data/benchmark_results_gemini_only/benchmark_results.jsonl"

    objs = load_results(INPUT_FILE_PATH)
    df = to_dataframe(objs)                  # pandas.DataFrame

    table = build_summary_table(
        df,
        NORMAL_NAMES,
        KM_NAMES,   
        CATEGORICAL_NAMES
    )
    print(INPUT_FILE_PATH)
    print(table.to_string(index=False))
