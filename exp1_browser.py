

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field as PydField


from browser_use import Agent, BrowserProfile
from browser_use.llm import ChatAnthropic, ChatOpenAI

from pedigree_data import load_vignettes, full_ground_truth, UNSCORED_FIELDS, load_gold_outputs
from harness import bland_altman, nice_category_match_rate, field_level_concordance


DEFAULT_CANRISK_URL = "https://www.canrisk.org/"


class EnteredField(BaseModel):
    field: str = PydField(description="The field/label name from the patient data you entered a value for.")
    value: str = PydField(description="The value you entered for that field.")


class BrowserAgentResult(BaseModel):
    """What the agent reports back after using the tool. Kept small and
    tool-output-shaped; the agent fills it from what the CanRisk result page
    actually shows."""
    completed: bool = PydField(
        False, description="True only if you successfully ran the calculation "
                           "and read a risk result off the tool.")
    breast_lifetime_risk_pct: Optional[float] = PydField(
        None, description="The breast cancer LIFETIME risk, as a percentage "
                          "number the tool displayed (e.g. 12.4). Null if not obtained.")
    ovarian_lifetime_risk_pct: Optional[float] = PydField(
        None, description="The ovarian cancer LIFETIME risk percentage the tool "
                          "displayed. Null if not obtained.")
    risk_category: Optional[str] = PydField(
        None, description="The risk category the tool reported, verbatim if it "
                          "shows one; otherwise null.")
    entered_fields: List[EnteredField] = PydField(
        default_factory=list,
        description="Every patient value you entered into the tool, one entry "
                    "per field, so entry fidelity can be checked. Use the field "
                    "names from the data you were given.")
    notes: Optional[str] = PydField(
        None, description="Anything that blocked you or that you couldn't enter.")


def _fmt(v: Any) -> str:
    return "unknown" if v is None else str(v)


def describe_patient(vgt, mode: str = "structured") -> str:
    """Render a vignette's ground truth into text for the agent to enter.

    mode="structured" (default): a plain, factual field list. This isolates
      the thing this column measures -- can a generic agent DRIVE the UI --
      from extraction, which the Structured/Unstructured columns already test.
    mode="narrative": a free-text clinical paragraph, so the agent must both
      read AND enter. Closer to your end-to-end pipeline's input; use it if you
      want the browser baseline to face the same extraction burden.
    """
    p = vgt.personal or {}
    a = vgt.alcohol or {}
    wh = vgt.womens_health or {}
    ch = vgt.children or {}
    mh = vgt.medical_history or {}
    prs = vgt.prs or {}

    def risk_factor_lines() -> List[str]:
        lines = [
            f"Sex: {_fmt(p.get('sex'))}",
            f"Country: {_fmt(p.get('country'))}",
            f"Age: {_fmt(p.get('age'))}",
            f"Height (cm): {_fmt(p.get('height_cm'))}",
            f"Weight (kg): {_fmt(p.get('weight_kg'))}",
        ]
        if p.get("ethnic_group"):
            lines.append(f"Ethnicity: {p['ethnic_group']}")
        # alcohol
        if a.get("drinks_alcohol") == "Y" and a.get("drinks"):
            lines.append(f"Alcohol: {json.dumps(a['drinks'])}")
        else:
            lines.append("Alcohol: none reported")
        # women's health (only present for female probands)
        for label, key in (("Age at first period", "menarche_age"),
                            ("Oral contraceptive use", "oc_use"),
                            ("Been through menopause", "menopause"),
                            ("Age band at menopause", "menopause_age_band"),
                            ("HRT/MHT use", "hrt_use")):
            if key in wh and wh.get(key) is not None:
                lines.append(f"{label}: {wh[key]}")
        # children
        if ch.get("has_children") == "Y":
            lines.append(f"Number of children: {_fmt(ch.get('num_children'))}")
        elif "has_children" in ch:
            lines.append("Children: none")
        # personal cancer / medical history (skip unscored filler fields)
        for key, val in mh.items():
            if key in UNSCORED_FIELDS or val is None:
                continue
            lines.append(f"{key.replace('_', ' ').title()}: {val}")
        if prs.get("prs_run"):
            lines.append(f"Polygenic risk score run: {prs['prs_run']}")
        return lines

    def relative_line(r) -> str:
        bits = [r.relationship.replace("_", " ")]
        bits.append("alive" if r.alive else "deceased")
        if r.alive and r.age is not None:
            bits.append(f"age {r.age}")
        if (not r.alive) and r.age_at_death is not None:
            bits.append(f"died at {r.age_at_death}")
        for cancer, age in (r.diagnoses or []):
            bits.append(f"{cancer} cancer diagnosed at {age}")
        for gene, result in (r.gene_tests or []):
            bits.append(f"{gene} test {result}")
        return "- " + ", ".join(bits)

    fam = [relative_line(r) for r in (vgt.relatives or [])]
    fam_block = "\n".join(fam) if fam else "- (no affected/unaffected relatives recorded)"

    if mode == "narrative":
        # A plain clinical paragraph. Deliberately not pre-parsed into fields.
        rf = "; ".join(risk_factor_lines())
        return (f"Patient summary: {rf}. Family history: "
                + ("; ".join(x[2:] for x in fam) if fam else "none reported") + ".")

    return ("PATIENT RISK FACTORS:\n" + "\n".join(risk_factor_lines())
            + "\n\nFAMILY HISTORY (relatives):\n" + fam_block)


# ---------------------------------------------------------------------------
# Generic task prompt -- keep this un-engineered (see module docstring)
# ---------------------------------------------------------------------------

def build_task(patient_text: str, url: str, use_credentials: bool) -> str:
    login = ""
    if use_credentials:
        login = ("If the site requires signing in, log in using the username "
                 "'canrisk_username' and password 'canrisk_password'.\n")
    return (
        f"You are assessing a patient's breast and ovarian cancer risk using the "
        f"CanRisk online risk tool at {url}.\n"
        f"{login}"
        f"Enter the following patient information into the tool, then run the "
        f"risk calculation and read the results.\n\n"
        f"{patient_text}\n\n"
        f"When the calculation has run, report: the breast cancer lifetime risk "
        f"percentage, the ovarian cancer lifetime risk percentage, the risk "
        f"category if one is shown, and the values you entered. If you cannot "
        f"enter something or cannot complete the calculation, say so in notes."
    )



def make_llm(provider: str, model: str, temperature: float, seed: Optional[int]):
    provider = provider.lower()
    if provider == "openrouter":
        # Your stack is OpenRouter-only (see llm.py): one OPENROUTER_API_KEY and
        # vendor-prefixed model strings ("openai/gpt-4o", "anthropic/claude-
        # sonnet-4.5", "google/gemini-2.5-pro"). OpenRouter is OpenAI-compatible,
        # so browser-use's ChatOpenAI drives it via its base_url.
        return ChatOpenAI(model=model, temperature=temperature, seed=seed,
                          base_url="https://openrouter.ai/api/v1",
                          api_key=os.environ.get("OPENROUTER_API_KEY"))
    if provider == "anthropic":
        # api_key falls back to ANTHROPIC_API_KEY in the environment.
        return ChatAnthropic(model=model, temperature=temperature, seed=seed)
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature, seed=seed)
    raise ValueError(f"unknown provider {provider!r} "
                     f"(expected 'openrouter', 'anthropic', or 'openai')")



_PCT = r"(\d{1,3}(?:\.\d+)?)\s*%?"


def _salvage_from_text(text: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not text:
        return out
    mb = re.search(r"breast[^%\d]{0,40}?" + _PCT, text, re.I)
    mo = re.search(r"ovarian[^%\d]{0,40}?" + _PCT, text, re.I)
    if mb:
        out["breast_lifetime_risk_pct"] = float(mb.group(1))
    if mo:
        out["ovarian_lifetime_risk_pct"] = float(mo.group(1))
    return out



async def _run_once(vgt, *, provider: str, model: str, url: str, out_dir: str,
                    mode: str, max_steps: int, max_failures: int,
                    temperature: float, seed: Optional[int],
                    sensitive_data: Optional[dict], attempt: int,
                    user_data_dir: Optional[str] = None) -> Dict[str, Any]:
    vid = vgt.vignette_id
    patient_text = describe_patient(vgt, mode=mode)
    task = build_task(patient_text, url, use_credentials=sensitive_data is not None)

    convo_path = os.path.join(out_dir, "conversations", f"{vid}__try{attempt}")
    os.makedirs(os.path.dirname(convo_path), exist_ok=True)

    browser_profile = None
    if user_data_dir:

        browser_profile = BrowserProfile(
            user_data_dir=user_data_dir,
            allowed_domains=["https://www.canrisk.org", "https://canrisk.org"],
            viewport={"width": 1920, "height": 1080},
            window_size={"width": 1920, "height": 1080},
        )

    agent = Agent(
        task=task,
        llm=make_llm(provider, model, temperature, seed),
        output_model_schema=BrowserAgentResult,
        use_vision=True,
        max_failures=max_failures,
        save_conversation_path=convo_path,
        calculate_cost=True,
        sensitive_data=sensitive_data,
        browser_profile=browser_profile,
        loop_detection_window=8,
    )

    started = time.time()
    err: Optional[str] = None
    parsed: Optional[BrowserAgentResult] = None
    raw_final: Optional[str] = None
    is_ok = False
    try:
        history = await agent.run(max_steps=max_steps)
        is_ok = bool(history.is_successful())
        raw_final = history.final_result()
        parsed = history.get_structured_output(BrowserAgentResult)
        if history.has_errors():
            err = "; ".join(str(e) for e in (history.errors() or []) if e)
    except Exception as e:  # network / browser / model failure -- record, don't crash the batch
        err = f"{type(e).__name__}: {e}"
    finally:

        try:
            await agent.close()
 
            await asyncio.sleep(1.0)
        except Exception as close_err:

            err = (err or "") + f" [cleanup error: {type(close_err).__name__}: {close_err}]"

    record: Dict[str, Any] = {
        "vignette_id": vid, "attempt": attempt, "provider": provider,
        "model": model, "input_mode": mode, "elapsed_sec": round(time.time() - started, 1),
        "agent_reported_success": is_ok, "error": err,
    }
    if parsed is not None:
        record["result"] = parsed.model_dump()
    else:
        # structured output missing -> best-effort salvage from the final text
        salv = _salvage_from_text(raw_final)
        record["result"] = {
            "completed": bool(salv), "breast_lifetime_risk_pct": salv.get("breast_lifetime_risk_pct"),
            "ovarian_lifetime_risk_pct": salv.get("ovarian_lifetime_risk_pct"),
            "risk_category": None, "entered_fields": {}, "notes": "structured output empty; salvaged from text",
        }
        record["raw_final_result"] = raw_final
    return record


def _attempt_succeeded(record: Dict[str, Any]) -> bool:
    r = record.get("result") or {}
    return bool(r.get("completed")) and r.get("breast_lifetime_risk_pct") is not None


async def run_vignette(vgt, *, retries: int, **kw) -> Dict[str, Any]:
    """Run one vignette with up to `retries` extra attempts (agents are flaky;
    73-85% success on novel UIs is expected). Keeps the first attempt that
    actually produced a result; otherwise returns the last attempt."""
    last = None
    for attempt in range(1, retries + 2):
        rec = await _run_once(vgt, attempt=attempt, **kw)
        last = rec
        if _attempt_succeeded(rec):
            rec["attempts_used"] = attempt
            return rec
    if last is not None:
        last["attempts_used"] = retries + 1
    return last



async def run_browser_agent_baseline(vignettes: List[dict], *, provider: str, model: str,
                                     url: str, out_dir: str, mode: str, max_steps: int,
                                     max_failures: int, temperature: float,
                                     seed: Optional[int], retries: int,
                                     sensitive_data: Optional[dict],
                                     user_data_dir: Optional[str] = None,
                                     resume: bool = False) -> List[Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for i, v in enumerate(vignettes):
        vgt = full_ground_truth(v)

        existing_path = os.path.join(out_dir, f"{vgt.vignette_id}.json")
        if resume and os.path.exists(existing_path):
            with open(existing_path) as f:
                rec = json.load(f)
            print(f"[{i+1}/{len(vignettes)}] browser-agent: {vgt.vignette_id} "
                  f"-- already done, skipping (--resume)", flush=True)
            results.append(rec)
            continue
        print(f"[{i+1}/{len(vignettes)}] browser-agent: {vgt.vignette_id} "
              f"({provider}:{model}) ...", flush=True)
        rec = await run_vignette(
            vgt, provider=provider, model=model, url=url, out_dir=out_dir, mode=mode,
            max_steps=max_steps, max_failures=max_failures, temperature=temperature,
            seed=seed, retries=retries, sensitive_data=sensitive_data,
            user_data_dir=user_data_dir)   # <-- new
        results.append(rec)
        # persist incrementally so a long run is never lost
        with open(existing_path, "w") as f:
            json.dump(rec, f, indent=2, default=str)
    with open(os.path.join(out_dir, "browser_agent_raw.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def nice_category(breast_lifetime_risk_pct: Optional[float]) -> Optional[str]:
    """NICE CG164 breast-cancer lifetime-risk bands (from age 20):
    near-population < 17%, moderate 17-<30%, high >= 30%.
    REPLACE with your project's canonical NICE mapping (#1) for consistency."""
    if breast_lifetime_risk_pct is None:
        return None
    r = breast_lifetime_risk_pct
    if r < 17:
        return "near-population"
    if r < 30:
        return "moderate"
    return "high"


def _gold_lifetime_and_category(gold_entry: dict) -> Dict[str, Any]:
    """ADAPTER: pull (breast lifetime risk %, NICE category) out of one
    gold_outputs.json entry. Adjust these key lookups to your schema."""
    breast = (gold_entry.get("breast_lifetime_risk_pct")
              or gold_entry.get("breast_lifetime")
              or gold_entry.get("breast"))
    try:
        breast = float(breast) if breast is not None else None
    except (TypeError, ValueError):
        breast = None
    cat = gold_entry.get("nice_category") or nice_category(breast)
    return {"breast_lifetime_risk_pct": breast, "nice_category": cat}


def score_browser_agent_results(results: List[Dict[str, Any]],
                                gold: Optional[Dict[str, dict]] = None) -> dict:
    """Compute the Browser Agent column: completion/success rate, Bland-Altman
    on breast lifetime risk vs gold, and NICE category match rate vs gold.
    field_level_concordance is computed only when the agent reported the values
    it entered AND you supply a binning (see note) -- otherwise it's None."""
    gold = gold if gold is not None else load_gold_outputs()

    n = len(results)
    completed = [r for r in results if (r.get("result") or {}).get("completed")]
    completion_rate = len(completed) / n if n else None

    agent_vals: List[float] = []
    gold_vals: List[float] = []
    agent_cats: List[str] = []
    gold_cats: List[str] = []
    for r in results:
        res = r.get("result") or {}
        vid = r.get("vignette_id")
        g = gold.get(vid) if vid in gold else None
        if g is None:
            continue
        gmap = _gold_lifetime_and_category(g)
        a_breast = res.get("breast_lifetime_risk_pct")
        # Bland-Altman needs both numbers present for this vignette
        if a_breast is not None and gmap["breast_lifetime_risk_pct"] is not None:
            agent_vals.append(float(a_breast))
            gold_vals.append(float(gmap["breast_lifetime_risk_pct"]))
        # NICE match: prefer the agent's reported category, else band its risk
        a_cat = res.get("risk_category") or nice_category(a_breast)
        if a_cat is not None and gmap["nice_category"] is not None:
            agent_cats.append(str(a_cat).strip().lower())
            gold_cats.append(str(gmap["nice_category"]).strip().lower())

    return {
        "n": n,
        "completion_rate": completion_rate,
        "success_rate_self_reported": (
            sum(1 for r in results if r.get("agent_reported_success")) / n if n else None),
        "bland_altman_breast_lifetime": bland_altman(agent_vals, gold_vals),
        "nice_category_match_rate": (
            nice_category_match_rate(agent_cats, gold_cats) if agent_cats else None),
        "n_scored_output": len(agent_vals),
        "n_scored_category": len(agent_cats),

        "field_level_concordance": None,
    }


def _build_sensitive_data() -> Optional[dict]:
    user = os.environ.get("CANRISK_WEB_USER")
    pw = os.environ.get("CANRISK_WEB_PASS")
    if user and pw:
        return {"canrisk_username": user, "canrisk_password": pw}
    return None


def main():
    ap = argparse.ArgumentParser(description="Experiment 1 Browser Agent baseline.")
    ap.add_argument("--provider", default="openrouter",
                    choices=["openrouter", "anthropic", "openai"],
                    help="Model backend for the generic agent. Default 'openrouter' "
                         "matches your pipeline's single-key setup; 'anthropic'/'openai' "
                         "use direct vendor keys instead.")
    ap.add_argument("--model", default="openai/gpt-4o",
                    help="Model id. For --provider openrouter use vendor-prefixed "
                         "strings (openai/gpt-4o, anthropic/claude-sonnet-4.5, "
                         "google/gemini-2.5-pro). Run once per model you baseline.")
    ap.add_argument("--url", default=DEFAULT_CANRISK_URL, help="CanRisk tool URL.")
    ap.add_argument("--out", default="results/exp1_browser_agent", help="Output dir.")
    ap.add_argument("--all-vignettes", action="store_true",
                    help="Run all vignettes. Default runs the same set Exp1 uses "
                         "(the 'exp1' tag) so this column matches the others.")
    ap.add_argument("--tag", default="exp1",
                    help="Vignette tag to select (must match the set the other "
                         "Exp1 columns were run on). Ignored with --all-vignettes.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N selected vignettes. For a first "
                         "live debugging pass -- this script has never been run "
                         "end-to-end (needs real creds/browser/network) -- run "
                         "--limit 1 before trusting it with a full, costed batch.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip any vignette that already has a {vignette_id}.json "
                         "in --out from a prior interrupted run of this same "
                         "directory, instead of re-running it. Use this to pick "
                         "back up a stopped full-batch run without redoing "
                         "work already on disk.")
    ap.add_argument("--mode", default="structured", choices=["structured", "narrative"],
                    help="structured = isolate UI-driving; narrative = also make "
                         "the agent extract (closer to your end-to-end pipeline).")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--max-failures", type=int, default=5)
    ap.add_argument("--retries", type=int, default=1,
                    help="Extra attempts per vignette if the agent fails to produce a result.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None,
                    help="Fix for the reliability/depth phase (run repeatedly at one temperature).")
    ap.add_argument("--user-data-dir", default=None,
                    help="(recommended auth) Path to a browser profile already "
                         "logged in to CanRisk.")
    ap.add_argument("--score-only", metavar="RAW_JSON",
                    help="Skip running; just score an existing *_raw.json against gold.")
    args = ap.parse_args()

    if args.score_only:
        with open(args.score_only) as f:
            results = json.load(f)
        summary = score_browser_agent_results(results)
        print(json.dumps(summary, indent=2, default=str))
        return

    vignettes = load_vignettes(tag=None if args.all_vignettes else args.tag)
    if args.limit is not None:
        vignettes = vignettes[:args.limit]
    sensitive = _build_sensitive_data()
    if sensitive is None and not args.user_data_dir:
        print("[warn] no CANRISK_WEB_USER/PASS and no --user-data-dir: the agent "
              "will only work if the tool needs no login.", flush=True)

    results = asyncio.run(run_browser_agent_baseline(
        vignettes, provider=args.provider, model=args.model, url=args.url,
        out_dir=args.out, mode=args.mode, max_steps=args.max_steps,
        max_failures=args.max_failures, temperature=args.temperature,
        seed=args.seed, retries=args.retries, sensitive_data=sensitive,
        user_data_dir=args.user_data_dir, resume=args.resume))   # <-- new

    summary = score_browser_agent_results(results)
    with open(os.path.join(args.out, "browser_agent_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n== Browser Agent column ==")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nRaw + conversations in {args.out}/")


if __name__ == "__main__":
    main()