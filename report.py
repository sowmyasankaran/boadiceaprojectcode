

import json
from typing import Optional, Any, Dict, List


NICE_THRESHOLDS = {
    "20_80": {"near_population_max": 17, "moderate_max": 30},
    "40_50": {"near_population_max": 3, "moderate_max": 8},
}


def save_raw_responses(results: dict, breast_path: str = "last_boadicea_response.json",
                        ovarian_path: str = "last_ovarian_response.json") -> None:
    for key, path in (("breast", breast_path), ("ovarian", ovarian_path)):
        val = results.get(key)
        if val is not None:
            try:
                with open(path, "w") as f:
                    json.dump(val, f, indent=2, default=str)
            except (TypeError, OSError):
                pass


def _pedigree_result(model_result: dict) -> Optional[dict]:
    try:
        return model_result["pedigree_result"][0]
    except (KeyError, IndexError, TypeError):
        return None


def _risk_key(d: dict) -> Optional[str]:
    return next((k for k in d if isinstance(k, str) and "risk" in k.lower()
                and k != "age"), None)


def _single_risk_entry(value: Any) -> Optional[Dict[str, Any]]:

    if value is None:
        return None
    entry = value[0] if isinstance(value, list) and value else value
    if not isinstance(entry, dict):
        return None
    rk = _risk_key(entry)
    if rk is None:
        return None
    percent = entry.get(rk, {}).get("percent") if isinstance(entry.get(rk), dict) else entry.get(rk)
    if percent is None:
        return None
    return {"age": entry.get("age"), "percent": percent}


def _risk_series(value: Any) -> List[Dict[str, Any]]:

    if not isinstance(value, list):
        return []
    out = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        rk = _risk_key(entry)
        if rk is None:
            continue
        percent = entry.get(rk, {}).get("percent") if isinstance(entry.get(rk), dict) else entry.get(rk)
        if percent is not None:
            out.append({"age": entry.get("age"), "percent": percent})
    return out


def _mutation_probabilities(value: Any) -> Dict[str, float]:

    out = {}
    if isinstance(value, dict):
        for gene, v in value.items():
            if isinstance(v, (int, float)):
                out[gene] = v
            elif isinstance(v, dict) and "percent" in v:
                out[gene] = v["percent"]
    elif isinstance(value, list):
        for entry in value:
            if isinstance(entry, dict) and "gene" in entry:
                pct = entry.get("percent")
                if isinstance(pct, dict):
                    pct = pct.get("percent")
                if pct is not None:
                    out[entry["gene"]] = pct
    return out


def _fmt_pct(p) -> str:
    return f"{p:.1f}%" if isinstance(p, (int, float)) else "not available"


def format_cancer_section(title: str, model_result: dict, cancer_label: str,
                          include_five_and_ten: bool = True) -> str:

    lines = [f"## {title}"]
    pr = _pedigree_result(model_result)
    if pr is None:
        err = model_result.get("error")
        return f"{lines[0]}\n(No result available{': ' + str(err) if err else '.'})"

    lifetime = _single_risk_entry(pr.get("lifetime_cancer_risk"))
    baseline_lifetime = _single_risk_entry(pr.get("baseline_lifetime_cancer_risk"))
    ten_yr = _single_risk_entry(pr.get("ten_yr_cancer_risk"))
    baseline_ten_yr = _single_risk_entry(pr.get("baseline_ten_yr_cancer_risk"))
    series = _risk_series(pr.get("cancer_risks"))

    if include_five_and_ten:
        five_yr = next((e for e in series if e.get("age") is not None), None)
        lines.append(f"- 5-year risk: {_fmt_pct(five_yr['percent']) if five_yr else 'not available in this response'}")
        lines.append(f"- 10-year risk: {_fmt_pct(ten_yr['percent']) if ten_yr else 'not available'}"
                     + (f" (population baseline: {_fmt_pct(baseline_ten_yr['percent'])})" if baseline_ten_yr else ""))

    if lifetime:
        age = lifetime.get("age", 80)
        line = f"- Risk between now and age {age}: {_fmt_pct(lifetime['percent'])}"
        if baseline_lifetime:
            line += f" (population average: {_fmt_pct(baseline_lifetime['percent'])})"
        lines.append(line)
    else:
        lines.append("- Risk to age 80: not available in this response")

    if series:
        lines.append("- Age-specific risk table:")
        for e in series:
            lines.append(f"    age {e.get('age', '?')}: {_fmt_pct(e['percent'])}")

    return "\n".join(lines)


def format_carrier_probabilities(model_result: dict) -> str:
    pr = _pedigree_result(model_result)
    if pr is None:
        return "## Pathogenic Variant Carrier Probabilities\n(No result available.)"
    probs = _mutation_probabilities(pr.get("mutation_probabilties"))
    if not probs:
        return ("## Pathogenic Variant Carrier Probabilities\n"
                "(Not available in this response -- see the saved raw JSON "
                "to check the actual key/shape.)")
    lines = ["## Pathogenic Variant Carrier Probabilities"]
    for gene, pct in probs.items():
        lines.append(f"- {gene}: {_fmt_pct(pct)}")
    return "\n".join(lines)


def format_advisories(results: dict) -> str:

    lines = ["## Advisory Notes"]
    seen = set()
    found = False
    for model in ("breast", "ovarian"):
        for w in (results.get(model) or {}).get("warnings", []) or []:
            if w not in seen:
                seen.add(w)
                lines.append(f"- {w}")
                found = True
    if not found:
        lines.append("- None reported.")
    return "\n".join(lines)


def format_nice_risk_category() -> str:
    return (
        "## Breast Cancer Risk Category (NICE)\n"
        "NOT COMPUTED. NICE's category (near-population / moderate / high) "
        "is based on risk over the FIXED age windows 20-80 and 40-50 -- not "
        "the risk-to-80-from-your-current-age figure above. Per CanRisk's "
        "own documentation, that specific figure is produced by re-running "
        "the model with the target's age artificially set to 20 (and "
        "separately to 40), which this submission does not currently do. "
        "Computing it from the current-age figures instead would risk "
        "showing the wrong category, which matters clinically -- so this is "
        "left unimplemented rather than guessed. Reference thresholds: "
        "20-80 risk <17% near-population, 17-30% moderate, 30%+ high; "
        "40-50 risk <3% near-population, 3-8% moderate, >8% high."
    )


def format_risk_factors_used(model_result: dict) -> str:
    pr = _pedigree_result(model_result)
    if pr is None or not pr.get("risk_factors"):
        return ""
    lines = ["## Risk Factors Used By The Model"]
    for k, v in pr["risk_factors"].items():
        lines.append(f"- {k}: {v}")
    if pr.get("mutation_frequency"):
        lines.append(f"- Mutation frequency population: "
                     f"{list(pr['mutation_frequency'].keys())}")
    return "\n".join(lines)


def build_comprehensive_report(results: dict) -> str:

    save_raw_responses(results)
    sections = [
        format_cancer_section("Breast Cancer Risk", results.get("breast", {}),
                              "breast", include_five_and_ten=True),
        format_nice_risk_category(),
        format_cancer_section("Ovarian Cancer Risk", results.get("ovarian", {}),
                              "ovarian", include_five_and_ten=True),
        format_carrier_probabilities(results.get("breast", {})),
        format_advisories(results),
    ]
    rf = format_risk_factors_used(results.get("breast", {}))
    if rf:
        sections.append(rf)
    return "\n\n".join(sections)