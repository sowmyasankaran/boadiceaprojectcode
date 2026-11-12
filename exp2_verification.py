

import argparse
import os
from typing import Any, Dict, List, Optional

from plausibility import check_field_plausibility, check_cross_field, check_missing_age
from pedigree_data import load_exp2_cases, load_vignettes, family_ground_truth
from harness import write_results_csv, write_results_json

FIELD_KEYS = ["height_cm", "current_age", "menarche_age", "menopause_age",
              "first_live_birth_age"]


def vignette_to_plausibility_values(vignette: dict) -> Dict[str, Optional[float]]:

    target = next(m for m in vignette["members"] if m.get("is_target"))
    age, yob = target.get("age"), target.get("yob")
    current_age = None if (not age and not yob) else age
    rf = vignette.get("risk_factors", {})
    return {
        "height_cm": rf.get("height"),
        "current_age": current_age,
        "menarche_age": rf.get("menarche"),
        "menopause_age": rf.get("menopause"),
        "first_live_birth_age": rf.get("first_live_birth"),
        "num_children": rf.get("parity"),
    }


def run_plausibility_layer(vignette: dict) -> list:

    values = vignette_to_plausibility_values(vignette)
    issues = []
    for key in ("height_cm", "current_age", "menarche_age",
                "menopause_age", "first_live_birth_age"):
        issue = check_field_plausibility(key, values.get(key))
        if issue:
            issues.append(issue)
    issues.extend(check_cross_field(values))
    missing_age_issue = check_missing_age(values.get("current_age"))
    if missing_age_issue:
        issues.append(missing_age_issue)
    return issues


def score_case(case: dict, with_verification: bool) -> dict:
    expected = case["expected"]
    vignette = case["perturbed_vignette"]

    result = {"case_id": case["case_id"], "kind": case["kind"],
              "with_verification": with_verification,
              "expected_flag": expected["expect_flag"],
              "expected_field": expected.get("field"),
              "expected_severity": expected.get("severity")}

    if not with_verification:
        result["caught"] = False
        result["would_submit"] = True
        return result

    issues = run_plausibility_layer(vignette)
    result["raised_fields"] = [i.field for i in issues]
    result["raised_severities"] = {i.field: i.severity for i in issues}

    if expected["expect_flag"]:
        matched = any(i.field == expected["field"] for i in issues)
        severity_matched = any(
            i.field == expected["field"] and i.severity == expected["severity"]
            for i in issues)
        result["caught"] = matched
        result["severity_matched"] = severity_matched
    else:
        result["caught"] = len(issues) == 0  # false positive if anything fired
        result["false_positive"] = len(issues) > 0

    result["would_submit"] = not any(i.severity == "hard" for i in issues)
    return result


def order_error_conversational_check(vignette: dict) -> dict:

    from pedigree_data import full_ground_truth
    from full_intake import build_structured_intake
    from experiments.oldintake import run_cross_section_checks

    vgt = full_ground_truth(vignette)
    _, ped = build_structured_intake(vgt)

    diagnosed = [ind for ind in ped.individuals.values()
                if not ind.is_proband and ind.diagnoses]
    # find a swappable pair the same way the new check itself would
    pair = None
    for i, a in enumerate(diagnosed):
        for b in diagnosed[i + 1:]:
            a_ref = a.age if a.age is not None else a.age_at_death
            b_ref = b.age if b.age is not None else b.age_at_death
            if a_ref is None or b_ref is None:
                continue
            for da in a.diagnoses:
                for db in b.diagnoses:
                    if (da.cancer_type == db.cancer_type and da.age is not None
                            and db.age is not None and da.age != db.age
                            and db.age <= a_ref and da.age <= b_ref):
                        pair = (a, b, da, db)
                        break
                if pair:
                    break
            if pair:
                break
        if pair:
            break

    if pair is None:
        return {"vignette_id": vignette["vignette_id"], "applicable": False}

    a, b, da, db = pair
    da.age, db.age = db.age, da.age  # perform the swap
    warnings = run_cross_section_checks([], ped)
    caught = any("swapped between them" in w for w in warnings)
    return {
        "vignette_id": vignette["vignette_id"], "applicable": True,
        "swapped": (a.id, b.id), "caught": caught, "warnings": warnings,
    }


def fidelity_error_conversational_check(vignette: dict) -> dict:

    import llm
    from schema import PersonalLifestyleSection
    from verification import verify_section

    gt_height = vignette["risk_factors"].get("height")
    if not gt_height or not llm.available():
        return {"vignette_id": vignette["vignette_id"], "applicable": False}

    section = PersonalLifestyleSection()
    section.set_value("sex", "F")
    section.set_value("country", "UK")
    section.set_value("age", 40)
    section.set_value("weight_kg", 65.0)
    # deliberately record a DIFFERENT height than what the transcript states
    altered_height = round(gt_height * 1.2, 1)
    section.set_value("height_cm", altered_height)
    transcript = [
        {"role": "assistant", "content": "What's your height?"},
        {"role": "user", "content": f"I'm {gt_height}cm tall."},
    ]
    verdict = verify_section(section, transcript, use_llm=True)
    mismatches = verdict["fidelity"].get("mismatches") or []
    caught = any(m["field"] == "height_cm" for m in mismatches)
    return {"vignette_id": vignette["vignette_id"], "applicable": True,
            "caught": caught, "mismatches": mismatches}


def run_experiment2(cases: Optional[List[dict]] = None, out_dir: str = "results",
                    run_order_and_fidelity_checks: bool = True,
                    live_check_sample_size: Optional[int] = None) -> dict:
    cases = cases if cases is not None else load_exp2_cases()
    rows = []
    for case in cases:
        rows.append(score_case(case, with_verification=True))
        rows.append(score_case(case, with_verification=False))

    order_rows, fidelity_rows = [], []
    if run_order_and_fidelity_checks:
        vignettes = list({v["vignette_id"]: v for v in load_vignettes()}.values())
        sample = vignettes if live_check_sample_size is None else vignettes[:live_check_sample_size]
        for v in sample:
            order_rows.append(order_error_conversational_check(v))
            fidelity_rows.append(fidelity_error_conversational_check(v))

    summary = _summarize(rows, order_rows, fidelity_rows)
    write_results_json(os.path.join(out_dir, "exp2_raw.json"), rows)
    write_results_json(os.path.join(out_dir, "exp2_order_error_check.json"), order_rows)
    write_results_json(os.path.join(out_dir, "exp2_fidelity_check.json"), fidelity_rows)
    write_results_json(os.path.join(out_dir, "exp2_summary.json"), summary)
    write_results_csv(os.path.join(out_dir, "exp2_summary.csv"), rows)
    return summary


def _summarize(rows: List[dict], order_rows: List[dict], fidelity_rows: List[dict]) -> dict:
    summary = {}
    kinds = sorted({r["kind"] for r in rows})
    for kind in kinds:
        with_v = [r for r in rows if r["kind"] == kind and r["with_verification"]]
        without_v = [r for r in rows if r["kind"] == kind and not r["with_verification"]]
        n = len(with_v)
        if kind == "clean":
            fp_rate = sum(1 for r in with_v if r.get("false_positive")) / n if n else None
            summary[kind] = {"n": n, "false_positive_rate_with_verification": fp_rate}
        else:
            sensitivity = sum(1 for r in with_v if r["caught"]) / n if n else None
            severity_match_rate = (
                sum(1 for r in with_v if r.get("severity_matched")) / n if n else None)
            silent_submit_rate = (
                sum(1 for r in without_v if r["would_submit"]) / len(without_v)
                if without_v else None)
            summary[kind] = {
                "n": n,
                "sensitivity_with_verification": sensitivity,
                "field_and_severity_match_rate": severity_match_rate,
                "silent_submit_rate_without_verification": silent_submit_rate,
            }

    applicable_order = [r for r in order_rows if r.get("applicable")]
    summary["order_error_check"] = {
        "n_checked": len(applicable_order),
        "caught_rate": (
            sum(1 for r in applicable_order if r["caught"])
            / len(applicable_order) if applicable_order else None),
    }
    applicable_fid = [r for r in fidelity_rows if r.get("applicable")]
    summary["fidelity_error_check"] = {
        "n_checked": len(applicable_fid),
        "caught_rate": (sum(1 for r in applicable_fid if r["caught"])
                        / len(applicable_fid) if applicable_fid else None),
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--skip-live-checks", action="store_true",
                        help="skip the order-error/fidelity-error live LLM checks "
                             "(the 162-case core dataset check needs no LLM either way)")
    parser.add_argument("--live-check-sample-size", type=int, default=None,
                        help="how many of the 27 vignettes to run the live order-error/"
                             "fidelity-error checks on. Default: all 27.")
    args = parser.parse_args()

    summary = run_experiment2(out_dir=args.out,
                              run_order_and_fidelity_checks=not args.skip_live_checks,
                              live_check_sample_size=args.live_check_sample_size)
    print("Experiment 2 summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")