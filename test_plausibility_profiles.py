

from plausibility import (
    check_field_plausibility, check_cross_field, check_country_plausibility,
    summarize_issues,
)

PASS, FAIL = [], []


def check(label, condition):
    (PASS if condition else FAIL).append(label)
    print(f"{'PASS' if condition else 'FAIL'}: {label}")


def field_issue(field, value):
    return check_field_plausibility(field, value)


def cross_issues(height=None, weight=None, age=None):
    bmi = weight / ((height / 100.0) ** 2) if height and weight else None
    return check_cross_field({
        "height_cm": height, "weight_kg": weight, "bmi": bmi, "current_age": age,
    })


print("=" * 70)
print("LAYER 1: individual field plausibility (isolated, single field)")
print("=" * 70)

h, w, a = 165, 65, 35
check("Clean baseline: height individually clean",
      field_issue("height_cm", h) is None)
check("Clean baseline: weight individually clean",
      field_issue("weight_kg", w) is None)
check("Clean baseline: age individually clean",
      field_issue("current_age", a) is None)
check("Clean baseline: no cross-field issues at all",
      len(cross_issues(h, w, a)) == 0)

issue = field_issue("height_cm", 135)
check("Height 135cm -> soft flag (below soft_min 140, above hard_min 50)",
      issue is not None and issue.severity == "soft")

issue = field_issue("weight_kg", 255)
check("Weight 255kg -> soft flag (above soft_max 250, below hard_max 500)",
      issue is not None and issue.severity == "soft")

issue = field_issue("height_cm", 290)
check("Height 290cm -> hard flag (above hard_max 270)",
      issue is not None and issue.severity == "hard")

issue = field_issue("weight_kg", 15)
check("Weight 15kg -> hard flag (below hard_min 20)",
      issue is not None and issue.severity == "hard")

issue = field_issue("current_age", 130)
check("Age 130 -> hard flag (above hard_max 110)",
      issue is not None and issue.severity == "hard")

print()
print("=" * 70)
print("LAYER 2: cross-field consistency (each field individually fine,")
print("combination is not -- or vice versa)")
print("=" * 70)


h, w = 210, 35
check(f"Height {h}cm alone: individually fine",
      field_issue("height_cm", h) is None)
check(f"Weight {w}kg alone: individually fine",
      field_issue("weight_kg", w) is None)
issues = cross_issues(h, w, 30)
bmi_issue = next((i for i in issues if i.field == "bmi" and "together" in i.message), None)
check(f"...but together (BMI {w/((h/100)**2):.1f}) -> HARD cross-field flag",
      bmi_issue is not None and bmi_issue.severity == "hard")

h, w = 200, 48
check(f"Height {h}cm alone: individually fine", field_issue("height_cm", h) is None)
check(f"Weight {w}kg alone: individually fine", field_issue("weight_kg", w) is None)
issues = cross_issues(h, w, 30)
bmi_issue = next((i for i in issues if i.field == "bmi" and "together" in i.message), None)
check(f"...but together (BMI {w/((h/100)**2):.1f}) -> SOFT cross-field flag",
      bmi_issue is not None and bmi_issue.severity == "soft")

h, a = 135, 40
issues = cross_issues(h, 55, a)
height_field_issues = [i for i in issues if i.field == "height_cm"]
range_issue = field_issue("height_cm", h)
check("Height 135cm for an adult (40): cross-field 'short for adult' flag fires",
      len(height_field_issues) == 1)
check("...AND the plain range check also fires independently (2 issues, "
      "same field -- this is the same-field-dedup scenario the agent's "
      "verify loop needs to collapse into ONE question)",
      range_issue is not None and range_issue.severity == "soft")

h, w = 170, 15
issues = cross_issues(h, w, 30)
bmi_issue = next((i for i in issues if i.field == "bmi"), None)
weight_issue = field_issue("weight_kg", w)
check(f"Weight {w}kg (individually hard) at height {h}cm also drags BMI "
      f"({w/((h/100)**2):.1f}) into hard range -- both layers agree",
      weight_issue.severity == "hard" and bmi_issue is not None
      and bmi_issue.severity == "hard")

print()
print("=" * 70)
print("Multi-issue: several simultaneous hard blocks")
print("=" * 70)

h, w, a = 20, 900, 150
summary = summarize_issues([
    field_issue("height_cm", h), field_issue("weight_kg", w),
    field_issue("current_age", a),
] + cross_issues(h, w, a))
check("Absurd profile (20cm/900kg/age150) -> has_blocking True",
      summary["has_blocking"])
check(f"...with {len(summary['hard'])} separate hard issues (height, weight, "
      f"age, and BMI all disagree simultaneously)", len(summary["hard"]) >= 3)

print()
print("=" * 70)
print("LAYER 1 (string field): country plausibility")
print("=" * 70)

check("'UK' (your own default) is NOT flagged",
      check_country_plausibility("UK") is None)
check("'uk' lowercase is NOT flagged (alias matching is case-insensitive)",
      check_country_plausibility("uk") is None)
check("'France' is NOT flagged", check_country_plausibility("France") is None)
issue = check_country_plausibility("Wakanda")
check("'Wakanda' IS flagged (soft)", issue is not None and issue.severity == "soft")
issue = check_country_plausibility("Genovia")
check("'Genovia' IS flagged (soft)", issue is not None and issue.severity == "soft")

print()
print("=" * 70)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:")
    for f in FAIL:
        print(f"  - {f}")
print("=" * 70)
