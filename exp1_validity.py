
import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import llm
from pedigree_data import load_vignettes, load_gold_outputs, full_ground_truth
from full_intake import build_structured_intake, run_conversational_intake_full, submit, _flat_sections
from harness import (
    field_level_concordance, bland_altman, write_results_csv, write_results_json,
)
from clinical_sections import (
    WomensHealthSection, ChildrenSection, BreastScreeningSection,
    MedicalHistorySection, AlcoholSection,
)
from genetic_sections import PRSSection
from canrisk_format import build_pedigree_file
from canrisk_test_call import get_auth_token, call_endpoint
from nice_category import compute_nice_category

SKILL_VARIANTS = {
    "strict_tool_determination": (
        "You are a precise clinical intake agent. Only call the extraction "
        "tool with a field if the patient's message unambiguously and "
        "explicitly states it. When in doubt, leave it out and ask again."
    ),
    "broad_prompting": (
        "You are a helpful clinical intake assistant. Use your best "
        "judgment to interpret what the patient means, including reasonable "
        "inference from context."
    ),
}

MINIMAL_INSTRUCTION_SYSTEM = (
    "You have access to the CanRisk/BOADICEA breast and ovarian cancer risk "
    "model via its web-services API, which expects a CanRisk-format "
    "pedigree file: a handful of '##key=value' header lines for the "
    "patient's own risk factors, followed by a '##FamID...' tab-separated "
    "header row and one tab-separated row per family member (including the "
    "patient herself, marked as the target). Given the patient profile and "
    "family history below, construct a complete CanRisk pedigree file "
    "yourself and submit it with the submit_pedigree tool."
)

SUBMIT_PEDIGREE_TOOL = {
    "name": "submit_pedigree",
    "description": "Submit a complete CanRisk-format pedigree file text to run the risk calculation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pedigree_text": {
                "type": "string",
                "description": ("The full CanRisk-format pedigree file text, "
                               "including '##' header lines and tab-separated "
                               "family member rows."),
            }
        },
        "required": ["pedigree_text"],
    },
}


def _network_available() -> bool:
    return bool(os.environ.get("CANRISK_USERNAME") and os.environ.get("CANRISK_PASSWORD"))


_token_cache: Dict[str, str] = {}


def _get_token() -> Optional[str]:
    """Cached auth token for this process -- submit_intake() re-authenticates
    every call by design (kept untouched, out of scope for this experiment
    layer to change), but the EXTRA calls this module makes on its own
    (NICE category, group 5's raw submission) don't need to hammer the auth
    endpoint separately every time."""
    username = os.environ.get("CANRISK_USERNAME")
    password = os.environ.get("CANRISK_PASSWORD")
    if not username or not password:
        return None
    if "token" not in _token_cache:
        _token_cache["token"] = get_auth_token(username, password)
    return _token_cache["token"]


def _lifetime_pct(model_result: dict) -> Optional[float]:
    try:
        entry = model_result["pedigree_result"][0]["lifetime_cancer_risk"][0]
        risk_key = next((k for k in entry if "risk" in k), None)
        return entry[risk_key]["percent"] if risk_key else None
    except (KeyError, IndexError, TypeError, StopIteration):
        return None


def _fields_for_concordance() -> List[str]:
    # bin-level fields the gold/agent outputs are compared on -- personal
    # fields plus the risk-factor fields BOADICEA actually consumes.
    return ["sex", "country", "age", "height_cm", "weight_kg"]


def _build_pedigree_text(sections, ped) -> str:
    def find(cls):
        return next((s for s in sections if isinstance(s, cls)), None)
    personal = sections[0]
    return build_pedigree_file(
        ped, personal=personal, womens=find(WomensHealthSection),
        children=find(ChildrenSection), screening=find(BreastScreeningSection),
        medical=find(MedicalHistorySection), prs=find(PRSSection),
        alcohol=find(AlcoholSection))


def parse_models_arg(models_arg: Optional[str]) -> List[Optional[str]]:
    """--models "openai/gpt-4o,anthropic/claude-sonnet-4.5,google/gemini-2.5-pro"
    -> ["openai/gpt-4o", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-pro"].
    These are OpenRouter's own vendor-prefixed model strings -- real
    cross-vendor comparison now works with the single OPENROUTER_API_KEY,
    no per-vendor key needed (that was the old adapter's limitation).
    None/empty -> [None], meaning "whatever llm.py's OPENROUTER_MODEL/default
    resolves to" -- so this experiment runs exactly as before if you don't
    pass --models at all."""
    if not models_arg:
        return [None]
    out = []
    for entry in models_arg.split(","):
        entry = entry.strip()
        if entry:
            out.append(entry)
    return out or [None]


def _model_label(model: Optional[str]) -> str:
    return model or f"default ({llm.default_model()})"


def _extraction_kwargs(model: Optional[str], extra_instructions: Optional[str] = None) -> dict:
    kwargs = {}
    if model:
        kwargs["model"] = model
    if extra_instructions:
        kwargs["extra_instructions"] = extra_instructions
    return kwargs



def run_group_structured(vgt, submit_live: bool, compute_nice: bool = False) -> dict:
    sections, ped = build_structured_intake(vgt)
    result = {"group": "structured_profile_input", "vignette_id": vgt.vignette_id,
              "model_label": "n/a (no LLM)",
              "field_concordance": 1.0,  # built directly from ground truth
              "section_complete": all(s.is_complete() for s in sections)}
    if submit_live and _network_available():
        api_result = submit(sections, ped)
        result["api_result"] = api_result
        result["breast_lifetime_pct"] = _lifetime_pct(api_result.get("breast", {}))
        if compute_nice:
            token = _get_token()
            if token:
                text = _build_pedigree_text(sections, ped)
                result["nice"] = compute_nice_category(text, token)
    return result


def run_group_conversational(vgt, condition: str, group_name: str, submit_live: bool,
                             seed: int = 0, extraction_kwargs: Optional[dict] = None,
                             model_label: str = "default", compute_nice: bool = False) -> dict:
    outcome = run_conversational_intake_full(vgt, condition, seed=seed,
                                             extraction_kwargs=extraction_kwargs)
    sections, ped = outcome["sections"], outcome["pedigree"]
    personal = sections[0]
    agent_values = {"sex": personal.values.get("sex"), "country": personal.values.get("country"),
                     "age": personal.values.get("age"), "height_cm": personal.values.get("height_cm"),
                     "weight_kg": personal.values.get("weight_kg")}
    fields = _fields_for_concordance()
    result = {
        "group": group_name, "vignette_id": vgt.vignette_id, "model_label": model_label,
        "field_concordance": field_level_concordance(agent_values, vgt.personal, fields),
        "section_complete": all(s.is_complete() for s in sections),
        "cross_section_warnings": outcome["cross_section_warnings"],
        "_agent_values": agent_values,
    }
    if submit_live and _network_available() and result["section_complete"]:
        api_result = submit(sections, ped)
        result["api_result"] = api_result
        result["breast_lifetime_pct"] = _lifetime_pct(api_result.get("breast", {}))
        if compute_nice:
            token = _get_token()
            if token:
                text = _build_pedigree_text(sections, ped)
                result["nice"] = compute_nice_category(text, token)
    return result


def _profile_description(vgt) -> str:
    return (f"Patient: {vgt.personal}. Risk-relevant history: "
           f"{vgt.medical_history}, {vgt.womens_health}. Family history: "
           f"{[(r.relationship, r.age, r.diagnoses, r.gene_tests) for r in vgt.relatives]}. "
           f"Ashkenazi Jewish ancestry: {vgt.ashkenazi}.")


def _run_llm_constructed_submission(vgt, group_name: str, system_prompt: str,
                                    submit_live: bool, model_label: str = "default",
                                    extraction_kwargs: Optional[dict] = None) -> dict:
    """Shared mechanism for any condition where an LLM (with no access to
    our engineered Section/Pedigree schema) constructs its own raw CanRisk
    pedigree text and submits it directly via the generic submit_pedigree
    tool. Used by both group 5 (minimally-instructed) and the off-the-shelf
    baseline (zero product-specific instruction) -- they differ ONLY in
    system_prompt, so whatever gap shows up between them isolates exactly
    the value of even a little domain guidance, separate from the value of
    the full engineered pipeline (which is what comparing either of them
    against the structured/unstructured groups shows)."""
    result = {"group": group_name, "vignette_id": vgt.vignette_id, "model_label": model_label}
    if not llm.available():
        result["error"] = "no LLM provider configured"
        return result

    kwargs = dict(extraction_kwargs or {})
    args = llm.call_with_tool(system_prompt,
                              [{"role": "user", "content": _profile_description(vgt)}],
                              SUBMIT_PEDIGREE_TOOL, **kwargs)
    pedigree_text = args.get("pedigree_text")
    result["llm_constructed_pedigree_text"] = pedigree_text
    if not pedigree_text:
        result["error"] = "LLM did not produce a pedigree_text argument"
        return result

    if submit_live and _network_available():
        token = _get_token()
        if not token:
            result["error"] = "CanRisk authentication failed"
            return result
        try:
            api_result = call_endpoint("boadicea", token, pedigree_text)
        except Exception as e:
            result["error"] = f"submission failed: {e}"
            return result
        result["api_result"] = api_result
        result["breast_lifetime_pct"] = _lifetime_pct(api_result)
        result["submission_error"] = (
            api_result.get("error") or api_result.get("Person Error")
            or api_result.get("Pedigree File Error")
            or next((v for k, v in api_result.items() if "error" in k.lower()), None))
    return result


def run_group_baseline_minimal(vgt, vignette: dict, submit_live: bool,
                               model_label: str = "default",
                               extraction_kwargs: Optional[dict] = None) -> dict:
    """The engineering-effort floor: a minimally-instructed LLM (told
    CanRisk's rough file shape exists) constructs its OWN raw pedigree text
    and submits it. Distinct from the off-the-shelf baseline below, which
    gets NO such hint at all -- see that function's docstring for why the
    two are kept separate rather than merged into one condition."""
    return _run_llm_constructed_submission(
        vgt, "baseline_minimal_instruction", MINIMAL_INSTRUCTION_SYSTEM,
        submit_live, model_label, extraction_kwargs)


OFFSHELF_BASELINE_SYSTEM = (
    "You are a general-purpose AI assistant (the way plain ChatGPT or Claude "
    "would be, with no custom setup). A patient wants to understand their "
    "breast and ovarian cancer risk using the CanRisk/BOADICEA tool "
    "(canrisk.org). You have not been given any CanRisk-specific formatting "
    "guidance or domain engineering -- only a generic function for "
    "submitting whatever data you think the underlying API needs. Use "
    "your own general knowledge and best judgment to help this patient get "
    "a risk result."
)


def run_group_offshelf_baseline(vgt, vignette: dict, submit_live: bool,
                                model_label: str = "default",
                                extraction_kwargs: Optional[dict] = None) -> dict:
    """The doc's own suggested baseline: 'We could also use Claude/Codex as
    a control/baseline that is comparable to our design.' This answers a
    DIFFERENT question than the gold standard does. The gold standard is
    the control -- it holds the backend constant so field concordance /
    absolute difference / NICE match isolate data-capture as the only
    variable, i.e. "are we correct". This baseline instead answers "did our
    engineering effort actually add value, or would a generic, unscaffolded
    agent do just as well with the same task and the same tool access" --
    it is a genuinely off-the-shelf agent, with NO CanRisk-specific hint at
    all (contrast with group 5's baseline_minimal_instruction, which still
    gets told the rough shape of a CanRisk file). Every engineered group's
    table entry gets an explicit delta against this group -- see
    summarize_table."""
    return _run_llm_constructed_submission(
        vgt, "offshelf_agent_baseline", OFFSHELF_BASELINE_SYSTEM,
        submit_live, model_label, extraction_kwargs)


def run_group_llm_only_prediction(vgt, gold: Optional[dict], model_label: str = "default",
                                  extraction_kwargs: Optional[dict] = None) -> dict:
    """No CanRisk call -- the LLM's own clinical judgment. Gold figures are
    included in the result (not fed to the LLM) purely so a human/LLM-judge
    parse step can score against them later without a second lookup."""
    if not llm.available():
        return {"group": "llm_only_prediction", "vignette_id": vgt.vignette_id,
                "model_label": model_label, "error": "no LLM provider configured"}
    system = (
        "You are estimating breast and ovarian cancer lifetime risk from a "
        "patient profile and family history, using your own clinical "
        "judgment (NOT a calculator). Give a lifetime risk percentage "
        "estimate for breast and ovarian cancer, and a NICE risk category "
        "(population/moderate/high), with brief reasoning."
    )
    kwargs = dict(extraction_kwargs or {})
    raw = llm.call_text(system, [{"role": "user", "content": _profile_description(vgt)}], **kwargs)
    gold_breast_lt = _lifetime_pct(gold.get("breast", {})) if gold else None
    return {"group": "llm_only_prediction", "vignette_id": vgt.vignette_id,
            "model_label": model_label, "raw_llm_output": raw,
            "gold_breast_lifetime_pct": gold_breast_lt,
            "note": "ungraded free text by design -- gold figure included for later scoring"}


class BrowserAgentRunner:
    """Group 3 stub -- needs real browser automation against the live
    CanRisk web form. There's no honest text-only substitute for this (see
    module docstring); wire this to a real browser-automation agent."""

    @staticmethod
    def run(vgt):
        raise NotImplementedError(
            "Browser agent group requires live browser automation against "
            "the actual CanRisk web form."
        )


def _icc_one_way(values_by_subject: List[List[float]]) -> Optional[float]:
    """One-way random-effects intraclass correlation coefficient, ICC(1),
    for k repeat measurements on each of n subjects (here: n vignettes, k
    repeat runs each). Standard mean-squares formulation:
        ICC = (MSB - MSW) / (MSB + (k-1)*MSW)
    where MSB is between-subject mean square and MSW is within-subject
    (residual) mean square. Returns None if there's not enough data (need
    at least 2 subjects and at least 2 repeats each) to compute it."""
    rows = [v for v in values_by_subject if len(v) >= 2]
    n = len(rows)
    if n < 2:
        return None
    k = len(rows[0])
    if any(len(v) != k for v in rows):
        k = min(len(v) for v in rows)
        rows = [v[:k] for v in rows]
    if k < 2:
        return None

    grand_mean = sum(x for row in rows for x in row) / (n * k)
    subject_means = [sum(row) / k for row in rows]

    ss_between = k * sum((m - grand_mean) ** 2 for m in subject_means)
    ms_between = ss_between / (n - 1)

    ss_within = sum((x - subject_means[i]) ** 2 for i, row in enumerate(rows) for x in row)
    ms_within = ss_within / (n * (k - 1))

    denom = ms_between + (k - 1) * ms_within
    if denom == 0:
        return None
    return (ms_between - ms_within) / denom


def reliability_metrics(repeat_results: List[dict]) -> dict:
    lifetimes = [r["breast_lifetime_pct"] for r in repeat_results
                if r.get("breast_lifetime_pct") is not None]
    n = len(lifetimes)
    mean = sum(lifetimes) / n if n else None
    sd = ((sum((x - mean) ** 2 for x in lifetimes) / n) ** 0.5) if n and n > 1 else 0.0
    concordances = [r["field_concordance"] for r in repeat_results]

    field_sets = [r.get("_agent_values") for r in repeat_results if r.get("_agent_values")]
    pairwise_agreements = []
    for i in range(len(field_sets)):
        for j in range(i + 1, len(field_sets)):
            keys = set(field_sets[i]) & set(field_sets[j])
            if not keys:
                continue
            agree = sum(1 for k in keys if field_sets[i][k] == field_sets[j][k])
            pairwise_agreements.append(agree / len(keys))

    nice_cats = [r.get("nice", {}).get("nice_category") for r in repeat_results
                if r.get("nice", {}).get("nice_category")]
    category_consistency = (
        (nice_cats.count(max(set(nice_cats), key=nice_cats.count)) / len(nice_cats))
        if nice_cats else None)

    return {
        "n_repeats": len(repeat_results),
        "sd_lifetime_risk": sd,
        "mean_lifetime_risk": mean,
        "mean_field_concordance": sum(concordances) / len(concordances) if concordances else None,
        "repeat_to_repeat_field_agreement": (
            sum(pairwise_agreements) / len(pairwise_agreements) if pairwise_agreements else None),
        "nice_category_consistency": category_consistency,
        "nice_categories_seen": nice_cats or None,
    }


def compute_icc_across_vignettes(depth_summary_raw: Dict[str, List[dict]]) -> Optional[float]:
    """ICC(1) computed across ALL vignettes in the depth subset at once
    (needs >=2 vignettes with >=2 repeats each -- a single-vignette depth
    run can report SD but can't produce a meaningful ICC)."""
    values_by_subject = []
    for vignette_id, repeats in depth_summary_raw.items():
        lifetimes = [r["breast_lifetime_pct"] for r in repeats
                    if r.get("breast_lifetime_pct") is not None]
        if lifetimes:
            values_by_subject.append(lifetimes)
    return _icc_one_way(values_by_subject)



BASELINE_GROUP = "offshelf_agent_baseline"


def summarize_table(rows: List[dict], gold: Dict[str, dict],
                    gold_nice: Optional[Dict[str, dict]] = None) -> Dict[str, dict]:
    by_group: Dict[str, List[dict]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)

    table = {}
    for group, group_rows in by_group.items():
        concordances = [r["field_concordance"] for r in group_rows
                        if r.get("field_concordance") is not None]
        submitted = [r for r in group_rows if r.get("breast_lifetime_pct") is not None]

        success_rate = (
            sum(1 for r in group_rows
                if r.get("breast_lifetime_pct") is not None and not r.get("submission_error"))
            / len(group_rows) if group_rows else None)

        agent_vals, gold_vals = [], []
        nice_matches = []
        for r in submitted:
            g = gold.get(r["vignette_id"])
            gold_lt = _lifetime_pct(g.get("breast", {})) if g else None
            if gold_lt is not None:
                agent_vals.append(r["breast_lifetime_pct"])
                gold_vals.append(gold_lt)
            if gold_nice and r.get("nice", {}).get("nice_category"):
                gold_cat = gold_nice.get(r["vignette_id"], {}).get("nice_category")
                if gold_cat:
                    nice_matches.append(r["nice"]["nice_category"] == gold_cat)

        abs_diffs = [abs(a - g) for a, g in zip(agent_vals, gold_vals)]
        ba = bland_altman(agent_vals, gold_vals) if agent_vals else {
            "mean_diff": None, "sd_diff": None, "loa_lower": None, "loa_upper": None, "n": 0}

        table[group] = {
            "n_profiles": len(group_rows),
            "n_submitted_and_matched_to_gold": len(agent_vals),
            "submission_success_rate": success_rate,
            "field_level_concordance": (
                sum(concordances) / len(concordances) if concordances else None),
            "mean_absolute_difference_lifetime_risk": (
                sum(abs_diffs) / len(abs_diffs) if abs_diffs else None),
            "bland_altman": ba,
            "nice_risk_category_match_rate": (
                sum(nice_matches) / len(nice_matches) if nice_matches else
                ("not computed -- rerun with --nice" if not gold_nice else None)),
        }


    baseline_entry = table.get(BASELINE_GROUP)
    if baseline_entry:
        for group, entry in table.items():
            if group == BASELINE_GROUP:
                entry["vs_offshelf_baseline"] = "reference (this IS the baseline)"
                continue
            comparison = {}
            if entry.get("mean_absolute_difference_lifetime_risk") is not None and \
               baseline_entry.get("mean_absolute_difference_lifetime_risk") is not None:
                comparison["abs_diff_improvement_vs_baseline"] = round(
                    baseline_entry["mean_absolute_difference_lifetime_risk"]
                    - entry["mean_absolute_difference_lifetime_risk"], 3)
            if entry.get("submission_success_rate") is not None and \
               baseline_entry.get("submission_success_rate") is not None:
                comparison["submission_success_rate_delta_pp"] = round(
                    (entry["submission_success_rate"]
                     - baseline_entry["submission_success_rate"]) * 100, 2)
            if comparison:
                entry["vs_offshelf_baseline"] = comparison
    return table



def run_experiment1(vignettes: Optional[List[dict]] = None, submit_live: bool = False,
                    depth_repeats: int = 0, depth_subset_size: int = 3,
                    out_dir: str = "results", models: Optional[str] = None,
                    compute_nice: bool = False) -> dict:
    vignettes = vignettes if vignettes is not None else load_vignettes(tag="exp1")
    gold = load_gold_outputs()
    model_list = parse_models_arg(models)

    gold_nice: Dict[str, dict] = {}
    if compute_nice and submit_live and _network_available():
        token = _get_token()
        if token:
            for vignette in vignettes:
                gold_nice[vignette["vignette_id"]] = compute_nice_category(
                    vignette["canrisk_file"], token)

    breadth_rows: List[dict] = []
    for vignette in vignettes:
        vgt = full_ground_truth(vignette)
        for model in model_list:
            label = _model_label(model)

            llm.set_active_model(model)
            extraction_kwargs = _extraction_kwargs(model)

            breadth_rows.append(run_group_structured(vgt, submit_live, compute_nice))
            breadth_rows.append(run_group_conversational(
                vgt, "agent_a", "unstructured_profile_input", submit_live,
                extraction_kwargs=extraction_kwargs, model_label=label,
                compute_nice=compute_nice))
            for name, instructions in SKILL_VARIANTS.items():
                variant_kwargs = _extraction_kwargs(model, instructions)
                breadth_rows.append(run_group_conversational(
                    vgt, "agent_b", f"backend_api_{name}", submit_live,
                    extraction_kwargs=variant_kwargs, model_label=label,
                    compute_nice=compute_nice))
            breadth_rows.append(run_group_baseline_minimal(
                vgt, vignette, submit_live, model_label=label,
                extraction_kwargs=extraction_kwargs))
            breadth_rows.append(run_group_offshelf_baseline(
                vgt, vignette, submit_live, model_label=label,
                extraction_kwargs=extraction_kwargs))
            breadth_rows.append(run_group_llm_only_prediction(
                vgt, gold.get(vignette["vignette_id"]), model_label=label,
                extraction_kwargs=extraction_kwargs))
            # browser_agent group intentionally omitted -- call
            # BrowserAgentRunner.run(vgt) once real browser automation exists.

    depth_summary = {}
    depth_raw: Dict[str, List[dict]] = {}
    if depth_repeats > 0 and submit_live and _network_available():
        for model in model_list:
            label = _model_label(model)
            llm.set_active_model(model)  # whole-agent sweep (see breadth loop)
            extraction_kwargs = _extraction_kwargs(model)
            for vignette in vignettes[:depth_subset_size]:
                vgt = full_ground_truth(vignette)
                repeats = []
                for i in range(depth_repeats):
                    r = run_group_conversational(
                        vgt, "agent_a", "unstructured_profile_input", submit_live=True,
                        seed=1000 + i, extraction_kwargs=extraction_kwargs,
                        model_label=label, compute_nice=compute_nice)
                    repeats.append(r)
                key = f"{vignette['vignette_id']}__{label}"
                depth_raw[key] = repeats
                depth_summary[key] = reliability_metrics(repeats)

        icc = compute_icc_across_vignettes(depth_raw)
        depth_summary["_overall_icc_lifetime_risk"] = icc

    llm.set_active_model(None)  # clear the sweep's process-wide model
    table = summarize_table(breadth_rows, gold, gold_nice if compute_nice else None)
    write_results_json(os.path.join(out_dir, "exp1_table_summary.json"), table)
    write_results_json(os.path.join(out_dir, "exp1_breadth_raw.json"), breadth_rows)
    write_results_json(os.path.join(out_dir, "exp1_depth_reliability.json"), depth_summary)
    write_results_json(os.path.join(out_dir, "exp1_gold_nice.json"), gold_nice)
    write_results_csv(os.path.join(out_dir, "exp1_breadth_summary.csv"),
                      [{"group": r["group"], "vignette_id": r["vignette_id"],
                        "model_label": r.get("model_label"),
                        "field_concordance": r.get("field_concordance"),
                        "section_complete": r.get("section_complete")} for r in breadth_rows])
    return {"breadth": breadth_rows, "depth": depth_summary, "table": table}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true",
                        help="make live CanRisk API calls (requires CANRISK_USERNAME/PASSWORD)")
    parser.add_argument("--depth-repeats", type=int, default=0)
    parser.add_argument("--depth-subset-size", type=int, default=3)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--models", type=str, default=None,
                        help="comma-separated OpenRouter model strings, e.g. "
                             "'openai/gpt-4o,anthropic/claude-sonnet-4.5,"
                             "google/gemini-2.5-pro'. Omit to use "
                             "OPENROUTER_MODEL (or llm.py's built-in default).")
    parser.add_argument("--nice", action="store_true",
                        help="also compute NICE risk category (DOUBLES live BOADICEA "
                             "calls per submission -- age reset to 20 and 40)")
    args = parser.parse_args()

    vignettes = load_vignettes(tag="exp1")
    if not args.submit:
        print("[exp1] --submit not passed: structured group reports 1.0 field "
              "concordance by construction; conversational groups compare "
              "extracted values to ground truth but skip live CanRisk calls.")
    results = run_experiment1(vignettes, submit_live=args.submit,
                              depth_repeats=args.depth_repeats,
                              depth_subset_size=args.depth_subset_size, out_dir=args.out,
                              models=args.models, compute_nice=args.nice)
    print(f"Ran {len(results['breadth'])} breadth condition-runs across {len(vignettes)} vignettes.")
    print("\nTable summary (also written to exp1_table_summary.json):")
    for group, metrics in results["table"].items():
        print(f"\n  {group}:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")