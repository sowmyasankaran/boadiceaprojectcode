

import argparse
import os
from typing import Any, Dict, List

from pedigree_data import load_vignettes, full_ground_truth
from full_intake import build_structured_intake, run_conversational_intake_full
from respondents import (
    BEHAVIOR_TRUTHFUL, BEHAVIOR_WRONG_THEN_CORRECTED, BEHAVIOR_MULTI_FACT,
    BEHAVIOR_AMBIGUOUS, SimulatedPatientAgent,
)
from harness import (per_field_accuracy, all_fields_correct, write_results_csv,
                     write_results_json, _fields_match)


def _all_ground_truth_values(vgt) -> Dict[str, Any]:

    merged: Dict[str, Any] = {}
    for d in (vgt.personal, vgt.alcohol, vgt.womens_health, vgt.children,
              vgt.breast_screening, vgt.medical_history, vgt.prs):
        merged.update(d)
    return merged


def _all_agent_values(sections) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for s in sections:
        merged.update(s.values)
    return merged


def _scored_fields(vgt, condition: str = "structured") -> List[str]:

    from pedigree_data import UNSCORED_FIELDS
    all_fields = [f for f in _all_ground_truth_values(vgt) if f not in UNSCORED_FIELDS]
    if condition in ("structured", "agent_a"):
        return all_fields
    from full_intake import reachable_fields_for_vignette
    reachable = reachable_fields_for_vignette(vgt)
    return [f for f in all_fields if f in reachable]


def _run_condition(vgt, condition: str, seed: int = 0, run_verification: bool = True) -> dict:
    if condition == "structured":
        sections, ped = build_structured_intake(vgt)
        agent_values = _all_agent_values(sections)
        complete = all(s.is_complete() for s in sections)
        turns_used = 0
    else:
        outcome = run_conversational_intake_full(vgt, condition, seed=seed,
                                                  run_verification=run_verification)
        sections = outcome["sections"]
        agent_values = _all_agent_values(sections)
        complete = all(s.is_complete() for s in sections)
        turns_used = sum(t.turns_used for t in outcome["traces"].values())

    fields = _scored_fields(vgt, condition)
    gt_values = _all_ground_truth_values(vgt)
    field_acc = per_field_accuracy(agent_values, gt_values, fields)

    return {
        "condition": condition, "vignette_id": vgt.vignette_id,
        "field_accuracy": field_acc,
        "all_correct": all(field_acc.values()) if field_acc else True,
        "section_complete": complete,
        "turns_used": turns_used,
    }


def _correction_provenance(outcome: dict, behavior_by_field: Dict[str, str],
                           gt_values: Dict[str, Any]) -> Dict[str, int]:

    merged = {"after_gather": {}, "after_verify": {}, "final": {}}
    for trace in outcome["traces"].values():
        snaps = getattr(trace, "stage_values", {}) or {}
        for stage in merged:
            merged[stage].update(snaps.get(stage, {}))
    counts = {"gather": 0, "verify": 0, "teach_back": 0, "never": 0}
    for f, b in behavior_by_field.items():
        if b != BEHAVIOR_WRONG_THEN_CORRECTED or f not in gt_values:
            continue
        gt = gt_values.get(f)
        if _fields_match(f, merged["after_gather"].get(f), gt):
            counts["gather"] += 1
        elif _fields_match(f, merged["after_verify"].get(f), gt):
            counts["verify"] += 1
        elif _fields_match(f, merged["final"].get(f), gt):
            counts["teach_back"] += 1
        else:
            counts["never"] += 1
    return counts


def _run_agent_c_with_logs(vgt, seed: int, run_verification: bool = True,
                            behavior_weights=None) -> dict:

    outcome = run_conversational_intake_full(vgt, "agent_c", seed=seed,
                                              run_verification=run_verification,
                                              behavior_weights=behavior_weights)
    sections = outcome["sections"]

    behavior_by_field: Dict[str, str] = {}
    for log in outcome["behavior_logs"].values():
        for entry in log:
            if entry.get("field"):
                behavior_by_field[entry["field"]] = entry["behavior"]

    ambiguous_flagged: set = set()
    for trace in outcome["traces"].values():
        ambiguous_flagged.update(trace.ambiguous_fields_flagged)

    agent_values = _all_agent_values(sections)
    fields = _scored_fields(vgt, "agent_c")
    gt_values = _all_ground_truth_values(vgt)
    field_acc = per_field_accuracy(agent_values, gt_values, fields)

    per_behavior_correct: Dict[str, List[bool]] = {
        b: [] for b in (BEHAVIOR_TRUTHFUL, BEHAVIOR_WRONG_THEN_CORRECTED,
                        BEHAVIOR_MULTI_FACT, BEHAVIOR_AMBIGUOUS)}
    for f in fields:
        b = behavior_by_field.get(f, BEHAVIOR_TRUTHFUL)
        per_behavior_correct[b].append(field_acc.get(f, False))

    corrected_fields = [f for f, b in behavior_by_field.items()
                        if b == BEHAVIOR_WRONG_THEN_CORRECTED and f in field_acc]
    correction_rate = (
        sum(1 for f in corrected_fields if field_acc.get(f)) / len(corrected_fields)
        if corrected_fields else None)

    ambiguous_fields = [f for f, b in behavior_by_field.items()
                        if b == BEHAVIOR_AMBIGUOUS and f in field_acc]
    flag_rate = (
        sum(1 for f in ambiguous_fields if f in ambiguous_flagged) / len(ambiguous_fields)
        if ambiguous_fields else None)

    unflagged = [f for f in ambiguous_fields if f not in ambiguous_flagged]
    silent_guess_correct_rate = (
        sum(1 for f in unflagged if field_acc.get(f)) / len(unflagged)
        if unflagged else None)

    return {
        "condition": "agent_c", "vignette_id": vgt.vignette_id,
        "field_accuracy": field_acc,
        "all_correct": all(field_acc.values()) if field_acc else True,
        "section_complete": all(s.is_complete() for s in sections),
        "behavior_by_field": behavior_by_field,
        "per_behavior_accuracy": {b: (sum(v) / len(v) if v else None)
                                  for b, v in per_behavior_correct.items()},
        "correction_rate": correction_rate,
        "correction_provenance": _correction_provenance(outcome, behavior_by_field, gt_values),
        "ambiguous_flag_rate": flag_rate,
        "ambiguous_silent_guess_correct_rate": silent_guess_correct_rate,
    }


def run_experiment4(vignettes=None, n_repeats: int = 1, out_dir: str = "results",
                    run_verification: bool = True, behavior_weights=None) -> List[dict]:
    vignettes = vignettes if vignettes is not None else load_vignettes(tag="exp4")
    rows = []
    for vignette_idx, vignette in enumerate(vignettes):
        vgt = full_ground_truth(vignette)
        rows.append(_run_condition(vgt, "structured"))
        rows.append(_run_condition(vgt, "agent_a", run_verification=run_verification))
        rows.append(_run_condition(vgt, "agent_b", run_verification=run_verification))
        for r in range(n_repeats):

            seed = 1000 + vignette_idx * 97 + r
            rows.append(_run_agent_c_with_logs(vgt, seed=seed, run_verification=run_verification,
                                               behavior_weights=behavior_weights))

    summary = _summarize(rows)
    write_results_json(os.path.join(out_dir, "exp4_raw.json"), rows)
    write_results_json(os.path.join(out_dir, "exp4_summary.json"), summary)
    write_results_csv(os.path.join(out_dir, "exp4_summary.csv"),
                      [{"condition": r["condition"], "vignette_id": r["vignette_id"],
                        "all_correct": r["all_correct"], "section_complete": r["section_complete"]}
                       for r in rows])
    return rows


def _summarize(rows: List[dict]) -> dict:
    by_condition: Dict[str, List[dict]] = {}
    for r in rows:
        by_condition.setdefault(r["condition"], []).append(r)

    summary = {}
    for cond, cond_rows in by_condition.items():
        n = len(cond_rows)
        all_correct_rate = sum(1 for r in cond_rows if r["all_correct"]) / n if n else None
        all_flags = [v for r in cond_rows for v in r["field_accuracy"].values()]
        mean_field_acc = sum(all_flags) / len(all_flags) if all_flags else None
        entry = {"n": n, "all_fields_correct_rate": all_correct_rate,
                 "mean_per_field_accuracy": mean_field_acc}
        if cond == "agent_c":
            for b in (BEHAVIOR_TRUTHFUL, BEHAVIOR_WRONG_THEN_CORRECTED,
                      BEHAVIOR_MULTI_FACT, BEHAVIOR_AMBIGUOUS):
                vals = [r["per_behavior_accuracy"].get(b) for r in cond_rows
                        if r["per_behavior_accuracy"].get(b) is not None]
                entry[f"accuracy__{b}"] = sum(vals) / len(vals) if vals else None
            corr = [r["correction_rate"] for r in cond_rows if r["correction_rate"] is not None]
            entry["mean_correction_rate"] = sum(corr) / len(corr) if corr else None
            prov = {"gather": 0, "verify": 0, "teach_back": 0, "never": 0}
            for r in cond_rows:
                for k, v in (r.get("correction_provenance") or {}).items():
                    prov[k] = prov.get(k, 0) + v
            entry["correction_provenance"] = prov
            flag_rates = [r["ambiguous_flag_rate"] for r in cond_rows
                         if r.get("ambiguous_flag_rate") is not None]
            entry["mean_ambiguous_flag_rate"] = (
                sum(flag_rates) / len(flag_rates) if flag_rates else None)
            silent = [r["ambiguous_silent_guess_correct_rate"] for r in cond_rows
                     if r.get("ambiguous_silent_guess_correct_rate") is not None]
            entry["mean_ambiguous_silent_guess_correct_rate"] = (
                sum(silent) / len(silent) if silent else None)
        summary[cond] = entry


    baseline = summary.get("agent_b")
    if baseline:
        for cond, entry in summary.items():
            if cond == "agent_b":
                entry["vs_baseline_mean_field_accuracy_pp"] = 0.0
                entry["vs_baseline_all_correct_rate_pp"] = 0.0
                continue
            if entry.get("mean_per_field_accuracy") is not None and baseline.get("mean_per_field_accuracy") is not None:
                entry["vs_baseline_mean_field_accuracy_pp"] = round(
                    (entry["mean_per_field_accuracy"] - baseline["mean_per_field_accuracy"]) * 100, 2)
            if entry.get("all_fields_correct_rate") is not None and baseline.get("all_fields_correct_rate") is not None:
                entry["vs_baseline_all_correct_rate_pp"] = round(
                    (entry["all_fields_correct_rate"] - baseline["all_fields_correct_rate"]) * 100, 2)
        if "agent_c" in summary:
            for b in (BEHAVIOR_TRUTHFUL, BEHAVIOR_WRONG_THEN_CORRECTED,
                      BEHAVIOR_MULTI_FACT, BEHAVIOR_AMBIGUOUS):
                acc = summary["agent_c"].get(f"accuracy__{b}")
                if acc is not None and baseline.get("mean_per_field_accuracy") is not None:
                    summary["agent_c"][f"vs_baseline_pp__{b}"] = round(
                        (acc - baseline["mean_per_field_accuracy"]) * 100, 2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--all-vignettes", action="store_true",
                        help="use all 27 vignettes instead of just the 6 tagged 'exp4'")
    parser.add_argument("--no-verification", action="store_true",
                        help="run without the verification/fidelity backstop layer "
                             "(the pre-fix behavior) -- for direct comparison against "
                             "earlier runs. Default is WITH verification, matching the "
                             "real deployed pipeline. Also disables the teach-back "
                             "backstop, so this reproduces the pre-fix correction "
                             "numbers.")
    parser.add_argument("--ablate-corrections", action="store_true",
                        help="Agent C only: disable the wrong_then_corrected behavior "
                             "(its weight is dropped; the remaining behaviors "
                             "renormalize automatically). Diagnostic for the context-"
                             "contamination hypothesis -- if Agent C's TRUTHFUL-field "
                             "accuracy (accuracy__truthful) rises toward Agent B's when "
                             "no wrong values are ever stated, a not-yet-corrected wrong "
                             "value was poisoning extraction of other fields. Run into a "
                             "SEPARATE --out dir and compare against the normal run. "
                             "correction_rate is None in this mode (nothing to correct).")
    args = parser.parse_args()

    behavior_weights = None
    if args.ablate_corrections:
        from respondents import DEFAULT_BEHAVIOR_WEIGHTS, BEHAVIOR_WRONG_THEN_CORRECTED
        behavior_weights = {b: w for b, w in DEFAULT_BEHAVIOR_WEIGHTS.items()
                            if b != BEHAVIOR_WRONG_THEN_CORRECTED}

    vignettes = load_vignettes(tag=None if args.all_vignettes else "exp4")
    rows = run_experiment4(vignettes, n_repeats=args.n_repeats, out_dir=args.out,
                           run_verification=not args.no_verification,
                           behavior_weights=behavior_weights)
    print(f"Ran {len(rows)} condition-runs across {len(vignettes)} vignettes. "
          f"Results in {args.out}/exp4_summary.json")