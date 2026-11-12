
import os
from report import (
    build_comprehensive_report, format_cancer_section, format_carrier_probabilities,
    format_advisories, format_nice_risk_category, format_risk_factors_used,
    _single_risk_entry, _risk_series, _mutation_probabilities, save_raw_responses,
)

PASS, FAIL = [], []


def check(label, cond):
    (PASS if cond else FAIL).append(label)
    print(f"{'PASS' if cond else 'FAIL'}: {label}")


print("=" * 68)
print("Low-level extraction helpers")
print("=" * 68)

check("single entry (list form) parses",
      _single_risk_entry([{"age": 80, "breast cancer risk": {"percent": 24.7}}])
      == {"age": 80, "percent": 24.7})
check("single entry (bare dict form) also parses",
      _single_risk_entry({"age": 80, "breast cancer risk": {"percent": 24.7}})
      == {"age": 80, "percent": 24.7})
check("None input -> None, not a crash", _single_risk_entry(None) is None)
check("garbage input -> None, not a crash", _single_risk_entry("nonsense") is None)
check("missing risk key -> None", _single_risk_entry({"age": 80}) is None)

series = _risk_series([
    {"age": 42, "breast cancer risk": {"percent": 1.7}},
    {"age": 80, "breast cancer risk": {"percent": 24.7}},
    {"not a dict": True},
])
check("risk series parses valid entries and skips malformed ones", len(series) == 2)
check("risk series preserves order", series[0]["age"] == 42 and series[1]["age"] == 80)
check("non-list input -> empty list, not a crash", _risk_series(None) == [])

check("mutation_probabilties: flat number dict",
      _mutation_probabilities({"BRCA1": 0.51}) == {"BRCA1": 0.51})
check("mutation_probabilties: nested percent dict",
      _mutation_probabilities({"BRCA1": {"percent": 0.51}}) == {"BRCA1": 0.51})
check("mutation_probabilties: list-of-dict form",
      _mutation_probabilities([{"gene": "BRCA1", "percent": 0.51}]) == {"BRCA1": 0.51})
check("mutation_probabilties: unrecognised shape -> empty, not a crash",
      _mutation_probabilities("nonsense") == {})

print()
print("=" * 68)
print("Section formatting -- realistic mock, graceful degradation")
print("=" * 68)

good_breast = {
    "pedigree_result": [{
        "lifetime_cancer_risk": [{"age": 80, "breast cancer risk": {"percent": 24.7}}],
        "baseline_lifetime_cancer_risk": [{"age": 80, "breast cancer risk": {"percent": 12.3}}],
        "ten_yr_cancer_risk": [{"age": 47, "breast cancer risk": {"percent": 4.3}}],
        "cancer_risks": [{"age": 42, "breast cancer risk": {"percent": 1.7}}],
        "mutation_probabilties": {"BRCA1": {"percent": 0.51}, "BRCA2": {"percent": 0.78}},
        "risk_factors": {"height_cm": 165, "bmi_category": "normal"},
        "mutation_frequency": {"UK": {}},
    }],
    "warnings": ["No year of birth for family member 'father'."],
}

section = format_cancer_section("Breast Cancer Risk", good_breast, "breast")
check("real numbers appear in the formatted section", "24.7%" in section and "12.3%" in section)

carriers = format_carrier_probabilities(good_breast)
check("carrier probabilities render", "BRCA1" in carriers and "0.5%" in carriers)

advisories = format_advisories({"breast": good_breast, "ovarian": {}})
check("advisory warnings surface verbatim", "father" in advisories)

rf = format_risk_factors_used(good_breast)
check("risk factors used section renders", "bmi_category" in rf)

nice = format_nice_risk_category()
check("NICE category is explicitly labelled NOT COMPUTED, not guessed",
      "NOT COMPUTED" in nice)
check("NICE section still surfaces the real public thresholds for later use",
      "17%" in nice and "30%" in nice and "3%" in nice and "8%" in nice)

# Missing/error results must degrade gracefully, never crash or fabricate
broken = {"error": "HTTP 400", "detail": "Pedigree Error"}
section_broken = format_cancer_section("Breast Cancer Risk", broken, "breast")
check("a failed model result produces a clear message, not a crash/fabrication",
      "No result available" in section_broken and "HTTP 400" in section_broken)

empty_section = format_cancer_section("Breast Cancer Risk", {}, "breast")
check("a totally empty result also degrades gracefully", "No result available" in empty_section)

carriers_missing = format_carrier_probabilities({"pedigree_result": [{}]})
check("missing carrier probabilities say so rather than showing nothing/wrong",
      "Not available" in carriers_missing)

print()
print("=" * 68)
print("Full report + raw JSON saving")
print("=" * 68)

full = {"breast": good_breast, "ovarian": {"pedigree_result": [{
    "lifetime_cancer_risk": [{"age": 80, "ovarian cancer risk": {"percent": 1.6}}],
}], "warnings": []}}

report_text = build_comprehensive_report(full)
check("full report includes all expected section headers",
      all(h in report_text for h in [
          "## Breast Cancer Risk", "## Breast Cancer Risk Category (NICE)",
          "## Ovarian Cancer Risk", "## Pathogenic Variant Carrier Probabilities",
          "## Advisory Notes",
      ]))

for f in ("last_boadicea_response.json", "last_ovarian_response.json"):
    if os.path.exists(f):
        os.remove(f)
save_raw_responses(full)
check("raw breast response saved to disk",
      os.path.exists("last_boadicea_response.json"))
check("raw ovarian response saved to disk",
      os.path.exists("last_ovarian_response.json"))
for f in ("last_boadicea_response.json", "last_ovarian_response.json"):
    if os.path.exists(f):
        os.remove(f)

print()
print("=" * 68)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED:", f)
print("=" * 68)