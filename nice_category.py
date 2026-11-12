from __future__ import annotations

import re
from datetime import date
from typing import Optional

from report import NICE_THRESHOLDS
from canrisk_test_call import call_endpoint

_SEVERITY_ORDER = ["near_population", "moderate", "high"]


def _find_header_and_columns(pedigree_text: str):
    lines = pedigree_text.split("\n")
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("##FamID"))
    cols = lines[header_idx].lstrip("#").split("\t")
    col_index = {name: i for i, name in enumerate(cols)}
    return lines, header_idx, col_index


def with_forced_proband_age(pedigree_text: str, forced_age: int,
                            as_of_year: Optional[int] = None) -> str:

    as_of_year = as_of_year or date.today().year
    lines, header_idx, cols = _find_header_and_columns(pedigree_text)
    age_i, yob_i, target_i = cols["Age"], cols["Yob"], cols["Target"]
    out = list(lines)
    for i in range(header_idx + 1, len(lines)):
        if not lines[i].strip():
            continue
        parts = lines[i].split("\t")
        if len(parts) > target_i and parts[target_i] == "1":
            parts[age_i] = str(forced_age)
            parts[yob_i] = str(as_of_year - forced_age)
            out[i] = "\t".join(parts)
    return "\n".join(out)


def _lifetime_pct(model_result: dict) -> Optional[float]:
    try:
        entry = model_result["pedigree_result"][0]["lifetime_cancer_risk"][0]
        risk_key = next((k for k in entry if "risk" in k), None)
        return entry[risk_key]["percent"] if risk_key else None
    except (KeyError, IndexError, TypeError, StopIteration):
        return None


def _ten_yr_pct(model_result: dict) -> Optional[float]:
    try:
        entry = model_result["pedigree_result"][0]["ten_yr_cancer_risk"][0]
        risk_key = next((k for k in entry if "risk" in k), None)
        return entry[risk_key]["percent"] if risk_key else None
    except (KeyError, IndexError, TypeError, StopIteration):
        return None


def _classify(percent: Optional[float], window: str) -> Optional[str]:
    if percent is None:
        return None
    t = NICE_THRESHOLDS[window]
    if percent < t["near_population_max"]:
        return "near_population"
    if percent < t["moderate_max"]:
        return "moderate"
    return "high"


def compute_nice_category(pedigree_text: str, token: str) -> dict:

    text_20 = with_forced_proband_age(pedigree_text, 20)
    text_40 = with_forced_proband_age(pedigree_text, 40)

    result_20 = call_endpoint("boadicea", token, text_20)
    result_40 = call_endpoint("boadicea", token, text_40)

    pct_20_80 = _lifetime_pct(result_20)
    pct_40_50 = _ten_yr_pct(result_40)

    cat_20_80 = _classify(pct_20_80, "20_80")
    cat_40_50 = _classify(pct_40_50, "40_50")

    categories = [c for c in (cat_20_80, cat_40_50) if c is not None]
    final = None
    if categories:
        final = max(categories, key=_SEVERITY_ORDER.index)

    return {
        "pct_20_80": pct_20_80, "category_20_80": cat_20_80,
        "pct_40_50": pct_40_50, "category_40_50": cat_40_50,
        "nice_category": final,
    }
