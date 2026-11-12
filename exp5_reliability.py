

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from pedigree_data import load_vignettes, full_ground_truth, load_gold_outputs, UNSCORED_FIELDS
from full_intake import (
    run_conversational_intake_full, build_structured_intake,
    reachable_fields_for_vignette, submit as submit_full,
)
from harness import per_field_accuracy, bland_altman, nice_category_match_rate
import exp5_metrics

DEFAULT_MODELS: List[Tuple[str, str]] = [
    ("anthropic", "claude-sonnet-4-5"),
    ("openai", "gpt-5"),
]

CONDITIONS: List[Tuple[str, str, Optional[str]]] = [
    ("unstructured", "agent_a", None),
]


def nice_category(breast_lifetime_pct: Optional[float]) -> Optional[str]:
    if breast_lifetime_pct is None:
        return None
    r = breast_lifetime_pct
    return "near-population" if r < 17 else ("moderate" if r < 30 else "high")



def _merge_gt(vgt) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for d in (vgt.personal, vgt.alcohol, vgt.womens_health, vgt.children,
              vgt.breast_screening, vgt.medical_history, vgt.prs):
        merged.update(d or {})
    return merged


def _merge_agent(sections) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for s in sections:
        merged.update(getattr(s, "values", {}) or {})
    return merged


def _scored_fields(vgt, condition: str) -> List[str]:
    all_fields = [f for f in _merge_gt(vgt) if f not in UNSCORED_FIELDS]
    if condition in ("structured", "agent_a"):
        return all_fields
    reachable = reachable_fields_for_vignette(vgt)
    return [f for f in all_fields if f in reachable]


def _lifetime_pct(results: dict, which: str = "breast") -> Optional[float]:

    if not results or "error" in results:
        return None
    try:
        lt = results.get(which, {})["pedigree_result"][0]["lifetime_cancer_risk"][0]
    except (KeyError, IndexError, TypeError):
        return None
    risk_key = next((k for k in lt if "risk" in k), None)
    try:
        return float(lt[risk_key]["percent"])
    except (KeyError, TypeError, ValueError):
        return None


def _extraction_kwargs(provider: str, model: str, temperature: Optional[float]) -> dict:
    kw: Dict[str, Any] = {"model": model, "provider_override": provider}
    if temperature is not None:
        kw["temperature"] = temperature
    return kw



def _run_conversational(vgt, condition: str, extra_instructions: Optional[str],
                        provider: str, model: str, temperature: Optional[float],
                        seed: int, submit: bool) -> Dict[str, Any]:
    ek = _extraction_kwargs(provider, model, temperature)
    if extra_instructions:
        ek["extra_instructions"] = extra_instructions

    t0 = time.time()
    error = None
    agent_values: Dict[str, Any] = {}
    breast = ovarian = None
    try:
        outcome = run_conversational_intake_full(
            vgt, condition, seed=seed, run_verification=True, extraction_kwargs=ek)
        sections = outcome["sections"]
        agent_values = _merge_agent(sections)
        if submit:
            results = submit_full(sections, outcome["pedigree"])
            breast = _lifetime_pct(results, "breast")
            ovarian = _lifetime_pct(results, "ovarian")
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    return {
        "fields": agent_values,
        "breast_lifetime_pct": breast,
        "ovarian_lifetime_pct": ovarian,
        "nice_category": nice_category(breast),
        "latency_sec": round(time.time() - t0, 2),
        "cost_usd": None,  # requires llm.py to surface token usage; see design doc
        "error": error,
    }


def _run_structured(vgt, submit: bool) -> Dict[str, Any]:
    """Model-independent floor: structured entry, no extraction. Run once."""
    t0 = time.time()
    error = None
    agent_values: Dict[str, Any] = {}
    breast = None
    try:
        sections, ped = build_structured_intake(vgt)
        agent_values = _merge_agent(sections)
        if submit:
            results = submit_full(sections, ped)
            breast = _lifetime_pct(results, "breast")
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    return {"fields": agent_values, "breast_lifetime_pct": breast,
            "nice_category": nice_category(breast),
            "latency_sec": round(time.time() - t0, 2), "error": error}


def run_breadth(models, conditions, vignettes, *, temperature, seed, submit, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gold = load_gold_outputs() if submit else {}
    rows: List[dict] = []

    # structured floor -- once, model-independent
    for v in vignettes:
        vgt = full_ground_truth(v)
        r = _run_structured(vgt, submit)
        fields = _scored_fields(vgt, "structured")
        acc = per_field_accuracy(r["fields"], _merge_gt(vgt), fields)
        rows.append({"phase": "breadth", "model": "n/a", "provider": "n/a",
                     "condition": "structured", "vignette_id": vgt.vignette_id,
                     "field_accuracy_mean": _mean(list(acc.values())),
                     "all_correct": all(acc.values()) if acc else True,
                     "breast_lifetime_pct": r["breast_lifetime_pct"],
                     "nice_category": r["nice_category"], "latency_sec": r["latency_sec"],
                     "error": r["error"]})

    for provider, model in models:
        for label, condition, extra in conditions:
            for vi, v in enumerate(vignettes):
                vgt = full_ground_truth(v)
                seed_i = seed + vi
                r = _run_conversational(vgt, condition, extra, provider, model,
                                        temperature, seed_i, submit)
                fields = _scored_fields(vgt, condition)
                acc = per_field_accuracy(r["fields"], _merge_gt(vgt), fields)
                rows.append({
                    "phase": "breadth", "model": model, "provider": provider,
                    "condition": label, "vignette_id": vgt.vignette_id,
                    "field_accuracy_mean": _mean(list(acc.values())),
                    "all_correct": all(acc.values()) if acc else True,
                    "breast_lifetime_pct": r["breast_lifetime_pct"],
                    "nice_category": r["nice_category"], "latency_sec": r["latency_sec"],
                    "cost_usd": r["cost_usd"], "error": r["error"]})
                print(f"[breadth] {provider}:{model} {label} {vgt.vignette_id} "
                      f"acc={rows[-1]['field_accuracy_mean']}", flush=True)

    summary = _summarize_breadth(rows, gold)
    _dump(out_dir, "exp5_breadth_raw.json", rows)
    _dump(out_dir, "exp5_breadth_summary.json", summary)
    return rows, summary


def _summarize_breadth(rows, gold) -> dict:
    by: Dict[Tuple[str, str], List[dict]] = {}
    for r in rows:
        by.setdefault((r["model"], r["condition"]), []).append(r)
    out = {}
    for (model, cond), rs in by.items():
        accs = [r["field_accuracy_mean"] for r in rs if r["field_accuracy_mean"] is not None]
        # Bland-Altman + NICE match vs gold (only where a submitted risk exists)
        agent_pct, gold_pct, a_cat, g_cat = [], [], [], []
        for r in rs:
            g = gold.get(r["vignette_id"]) if gold else None
            if g is None:
                continue
            gp = _gold_breast_pct(g)
            if r.get("breast_lifetime_pct") is not None and gp is not None:
                agent_pct.append(r["breast_lifetime_pct"]); gold_pct.append(gp)
            gc = g.get("nice_category") or nice_category(gp)
            if r.get("nice_category") and gc:
                a_cat.append(r["nice_category"].lower()); g_cat.append(str(gc).lower())
        lat = [r["latency_sec"] for r in rs if r.get("latency_sec") is not None]
        out[f"{model} | {cond}"] = {
            "n": len(rs),
            "field_accuracy_mean": _mean(accs),
            "all_correct_rate": _mean([1.0 if r["all_correct"] else 0.0 for r in rs]),
            "bland_altman_breast": bland_altman(agent_pct, gold_pct) if agent_pct else None,
            "nice_category_match_rate": nice_category_match_rate(a_cat, g_cat) if a_cat else None,
            "mean_latency_sec": _mean(lat),
            "errors": sum(1 for r in rs if r.get("error")),
        }
    return out


def run_depth(models, condition_spec, subset, *, repeats, temperature, seed, submit, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    label, condition, extra = condition_spec
    raw: List[dict] = []
    summary: Dict[str, dict] = {}

    for provider, model in models:
        categories_per_profile: List[List[Optional[str]]] = []
        risk_per_profile: List[List[Optional[float]]] = []
        fields_per_profile: List[List[Dict[str, Any]]] = []

        for vi, v in enumerate(subset):
            vgt = full_ground_truth(v)
            cats, risks, fields = [], [], []
            for rep in range(repeats):
                seed_rep = (seed + vi * 1000 + rep) if seed is not None else rep
                r = _run_conversational(vgt, condition, extra, provider, model,
                                        temperature, seed_rep, submit)
                cats.append(r["nice_category"])
                risks.append(r["breast_lifetime_pct"])
                fields.append(r["fields"])
                raw.append({"phase": "depth", "model": model, "provider": provider,
                            "condition": label, "vignette_id": vgt.vignette_id,
                            "repeat": rep, "breast_lifetime_pct": r["breast_lifetime_pct"],
                            "nice_category": r["nice_category"], "latency_sec": r["latency_sec"],
                            "error": r["error"]})
                print(f"[depth] {provider}:{model} {vgt.vignette_id} rep {rep+1}/{repeats}",
                      flush=True)
            categories_per_profile.append(cats)
            risk_per_profile.append(risks)
            fields_per_profile.append(fields)

        summary[f"{model}"] = exp5_metrics.summarize_model_depth(
            categories_per_profile, risk_per_profile, fields_per_profile)

    _dump(out_dir, "exp5_depth_raw.json", raw)
    _dump(out_dir, "exp5_depth_summary.json", summary)
    return raw, summary



def _gold_breast_pct(g: dict) -> Optional[float]:
    """ADAPTER: adjust to gold_outputs.json's key names (same as the browser-agent
    harness). Returns the gold breast lifetime risk percentage."""
    for k in ("breast_lifetime_risk_pct", "breast_lifetime", "breast"):
        if k in g and g[k] is not None:
            try:
                return float(g[k])
            except (TypeError, ValueError):
                return None
    return None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _dump(out_dir, name, obj):
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _select_models(spec: Optional[str]) -> List[Tuple[str, str]]:
    if not spec:
        return DEFAULT_MODELS
    out = []
    for item in spec.split(","):
        provider, _, model = item.partition(":")
        if provider and model:
            out.append((provider.strip(), model.strip()))
    return out or DEFAULT_MODELS


def main():
    ap = argparse.ArgumentParser(description="Experiment 5 model-choice + reliability sweep.")
    ap.add_argument("--phase", choices=["breadth", "depth"], required=True)
    ap.add_argument("--models", default=None,
                    help="Comma list of provider:model (e.g. "
                         "'anthropic:claude-sonnet-4-5,openai:gpt-5'). Defaults to DEFAULT_MODELS.")
    ap.add_argument("--conditions", default=None,
                    help="Comma list of condition labels from CONDITIONS to include "
                         "(breadth only). Defaults to all of CONDITIONS.")
    ap.add_argument("--condition", default="unstructured",
                    help="Single condition label from CONDITIONS to use (depth only).")
    ap.add_argument("--out", default="results/exp5")
    ap.add_argument("--tag", default="exp1", help="Vignette tag (breadth profile set).")
    ap.add_argument("--all-vignettes", action="store_true")
    ap.add_argument("--subset-tag", default="exp5_depth",
                    help="Tag selecting the smaller depth subset (depth only).")
    ap.add_argument("--repeats", type=int, default=5, help="Repeats per profile (depth).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Fixed temperature for the sweep. Use >0 for a meaningful "
                         "depth reliability read (at 0 you measure determinism).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-submit", action="store_true",
                    help="Skip CanRisk submission -> field accuracy / field agreement "
                         "only; no Bland-Altman / NICE / SD / ICC (those need a risk result).")
    args = ap.parse_args()

    models = _select_models(args.models)
    submit = not args.no_submit

    if args.phase == "breadth":
        conds = CONDITIONS
        if args.conditions:
            keep = {c.strip() for c in args.conditions.split(",")}
            conds = [c for c in CONDITIONS if c[0] in keep]
        vignettes = load_vignettes(tag=None if args.all_vignettes else args.tag)
        _, summary = run_breadth(models, conds, vignettes, temperature=args.temperature,
                                 seed=(args.seed or 0), submit=submit, out_dir=args.out)
    else:
        spec = next((c for c in CONDITIONS if c[0] == args.condition), CONDITIONS[0])
        subset = load_vignettes(tag=args.subset_tag)
        _, summary = run_depth(models, spec, subset, repeats=args.repeats,
                               temperature=args.temperature, seed=args.seed,
                               submit=submit, out_dir=args.out)

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nResults in {args.out}/")


if __name__ == "__main__":
    main()
