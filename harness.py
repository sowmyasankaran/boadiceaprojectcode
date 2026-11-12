from __future__ import annotations

import json
import os
import time
import csv
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Protocol

import llm
from agent import (
    extract_turn, compose_question, apply_extraction, resolve_field,
    is_affirmative,
)
from verification import verify_section


class Respondent(Protocol):


    def next_message(self, transcript: List[dict], section) -> Optional[str]:
        ...

    def resolve_plausibility(self, field: str, message: str) -> str:
        ...

    def confirm_or_correct(self, field: str, message: str) -> str:
        ...

    def teach_back_response(self, summary: str, section=None) -> str:
        ...

    def pending_disclosure(self) -> bool:
        ...


@dataclass
class GroundTruthRespondentBase:

    profile: Dict[str, Any]

    def resolve_plausibility(self, field: str, message: str) -> str:
        return str(self.profile.get(field, ""))

    def confirm_or_correct(self, field: str, message: str) -> str:
        return "yes"

    def pending_disclosure(self) -> bool:
        return False

    def teach_back_response(self, summary: str, section=None) -> str:
        if section is None:
            return "yes"
        discrepancies = []
        for name, recorded in section.values.items():
            if name not in self.profile:
                continue
            truth = self.profile.get(name)
            if truth is None:
                continue
            if not _fields_match(name, recorded, truth):
                discrepancies.append((name, truth))
        if not discrepancies:
            return "yes"
        if not llm.available():
            return "; ".join(f"actually, {n} should be {v}" for n, v in discrepancies)
        system = (
            "You are role-playing a patient who was just read back a summary "
            "of their intake and noticed one or more values are wrong. State "
            "the correction(s) naturally, giving the correct value(s), and "
            "nothing beyond the corrections listed."
        )
        user = (f"Summary you were read:\n{summary}\n\n"
                f"Corrections to state (field -> correct value): "
                f"{json.dumps(dict(discrepancies), default=str)}\n"
                "Reply with ONLY the patient's spoken correction.")
        return llm.call_text(system, [{"role": "user", "content": user}],
                             model=llm.reference_model())


@dataclass
class TraceEvent:
    kind: str
    content: str
    field: Optional[str] = None


@dataclass
class RunTrace:
    events: List[TraceEvent] = dc_field(default_factory=list)
    turns_used: int = 0
    corrections_applied: int = 0
    teach_back_corrections: int = 0
    deferred_recovered: int = 0
    hit_max_turns: bool = False
    stage_values: Dict[str, Dict[str, Any]] = dc_field(default_factory=dict)
    ambiguous_fields_flagged: set = dc_field(default_factory=set)

    def log(self, kind: str, content: str, field: Optional[str] = None):
        self.events.append(TraceEvent(kind, content, field))


def gather_noninteractive(section, transcript: List[dict], respondent: Respondent,
                           trace: RunTrace, max_turns: int = 25,
                           extraction_kwargs: Optional[dict] = None) -> None:
    extraction_kwargs = extraction_kwargs or {}

    def _pending() -> bool:
        fn = getattr(respondent, "pending_disclosure", None)
        return bool(fn()) if callable(fn) else False

    while not section.is_complete() or _pending():
        if trace.turns_used >= max_turns:
            trace.hit_max_turns = True
            return
        user_msg = respondent.next_message(transcript, section)
        if user_msg is None:
            return  # respondent has nothing further to volunteer
        trace.turns_used += 1
        transcript.append({"role": "user", "content": user_msg})
        trace.log("user", user_msg)
        extracted = extract_turn(section, transcript, **extraction_kwargs)
        ambiguous = apply_extraction(section, extracted)
        if ambiguous:
            trace.ambiguous_fields_flagged.update(ambiguous)
        if not section.is_complete() or ambiguous:
            agent_turn = compose_question(section, transcript)
            transcript.append({"role": "assistant", "content": agent_turn})
            trace.log("assistant", agent_turn)


def verify_noninteractive(section, transcript: List[dict], respondent: Respondent,
                           trace: RunTrace,
                           confirmed_plausibility: Optional[set] = None,
                           resolved_fidelity: Optional[set] = None,
                           max_passes: int = 15) -> tuple:

    confirmed_plausibility = set() if confirmed_plausibility is None else confirmed_plausibility
    resolved_fidelity = set() if resolved_fidelity is None else resolved_fidelity
    verdict_history = []

    for _ in range(max_passes):
        verdict = verify_section(section, transcript)
        verdict_history.append(verdict)

        if verdict["blocking_plausibility"]:
            corrected_fields = set()
            for i in verdict["blocking_plausibility"]:
                trace.log("plausibility_block", i.message, i.field)
                user_msg = respondent.resolve_plausibility(i.field, i.message)
                transcript.append({"role": "user", "content":
                                   f"For {i.field}, the correct value is: {user_msg}"})
                extracted = resolve_field(section, i.field, user_msg)
                apply_extraction(section, extracted)
                corrected_fields.add(i.field)
                trace.corrections_applied += 1
            for f in corrected_fields:
                confirmed_plausibility.discard(f)
                resolved_fidelity.discard(f)
            continue

        fidelity_mismatches = [m for m in (verdict["fidelity"].get("mismatches") or [])
                               if m["field"] not in resolved_fidelity]
        unresolved_soft = [i for i in verdict["confirm_plausibility"]
                           if i.field not in confirmed_plausibility]

        if not fidelity_mismatches and not unresolved_soft:
            return confirmed_plausibility, resolved_fidelity, verdict_history

        touched_this_pass = set()

        for m in fidelity_mismatches:
            fld = m["field"]
            if fld in resolved_fidelity or fld in touched_this_pass:
                continue
            trace.log("fidelity_mismatch",
                      f"recorded={m['recorded_value']} said={m['what_patient_said']}", fld)
            resp = respondent.confirm_or_correct(
                fld, f"I recorded {fld} as '{m['recorded_value']}', but it sounded "
                     f"like you said '{m['what_patient_said']}'.")
            if resp.strip() == "1" or resp.strip().lower() in ("yes", "keep"):
                pass
            else:
                user_reply = m["what_patient_said"] if resp.strip() == "2" else resp
                transcript.append({"role": "user", "content":
                                   f"Clarification for {fld}: {user_reply}"})
                extracted = resolve_field(section, fld, user_reply)
                apply_extraction(section, extracted)
                trace.corrections_applied += 1
            resolved_fidelity.add(fld)
            confirmed_plausibility.discard(fld)
            if fld in ("height_cm", "weight_kg"):
                confirmed_plausibility.discard("bmi")
                resolved_fidelity.discard("bmi")
            touched_this_pass.add(fld)

        for i in unresolved_soft:
            if i.field in confirmed_plausibility or i.field in touched_this_pass:
                continue
            trace.log("plausibility_soft", i.message, i.field)
            resp = respondent.confirm_or_correct(i.field, i.message)
            if is_affirmative(resp, context=f"Flagged value: {i.message}"):
                confirmed_plausibility.add(i.field)
            else:
                transcript.append({"role": "user", "content":
                                   f"Correction for {i.field}: {resp}"})
                extracted = resolve_field(section, i.field, resp)
                apply_extraction(section, extracted)
                trace.corrections_applied += 1
                resolved_fidelity.discard(i.field)
                if i.field in ("height_cm", "weight_kg"):
                    confirmed_plausibility.discard("bmi")
                    resolved_fidelity.discard("bmi")
            touched_this_pass.add(i.field)

    return confirmed_plausibility, resolved_fidelity, verdict_history


def run_conversational_intake(section, respondent: Respondent, opening: str,
                               max_turns: int = 25, run_verification: bool = True,
                               extraction_kwargs: Optional[dict] = None,
                               shared_transcript: Optional[List[dict]] = None,
                               ) -> tuple:
    if shared_transcript is not None:
        transcript = shared_transcript
        transcript.append({"role": "assistant", "content": opening})
    else:
        transcript: List[dict] = [{"role": "assistant", "content": opening}]
    trace = RunTrace()

    gather_noninteractive(section, transcript, respondent, trace, max_turns=max_turns,
                          extraction_kwargs=extraction_kwargs)
    trace.stage_values["after_gather"] = dict(section.values)

    if run_verification and section.is_complete():
        confirmed, resolved = set(), set()
        for i in range(5):
            confirmed, resolved, _ = verify_noninteractive(
                section, transcript, respondent, trace, confirmed, resolved)
            if i == 0:
                trace.stage_values["after_verify"] = dict(section.values)
            summary = section.teach_back()
            trace.log("teach_back", summary)
            resp = respondent.teach_back_response(summary, section)
            if is_affirmative(resp, context=summary):
                break

            transcript.append({"role": "user", "content": resp})
            trace.log("user", resp)
            extracted = extract_turn(section, transcript, **(extraction_kwargs or {}))
            apply_extraction(section, extracted)
            trace.teach_back_corrections += 1
            trace.corrections_applied += 1

    trace.stage_values.setdefault("after_verify", dict(section.values))
    trace.stage_values["final"] = dict(section.values)

    return section, transcript, trace


def submit_personal_section(section) -> dict:

    from agent import submit_section
    return submit_section(section)

def field_level_concordance(agent_values: Dict[str, Any], gold_values: Dict[str, Any],
                             fields: List[str]) -> float:
    if not fields:
        return 1.0
    matches = sum(1 for f in fields if agent_values.get(f) == gold_values.get(f))
    return matches / len(fields)


def bland_altman(agent_vals: List[float], gold_vals: List[float]) -> dict:

    diffs = [a - g for a, g in zip(agent_vals, gold_vals)]
    n = len(diffs)
    if n == 0:
        return {"mean_diff": None, "sd_diff": None, "loa_lower": None, "loa_upper": None}
    mean_diff = sum(diffs) / n
    sd_diff = (sum((d - mean_diff) ** 2 for d in diffs) / n) ** 0.5 if n > 1 else 0.0
    return {
        "mean_diff": mean_diff, "sd_diff": sd_diff,
        "loa_lower": mean_diff - 1.96 * sd_diff,
        "loa_upper": mean_diff + 1.96 * sd_diff,
        "n": n,
    }


def nice_category_match_rate(agent_categories: List[str], gold_categories: List[str]) -> float:
    if not agent_categories:
        return 0.0
    matches = sum(1 for a, g in zip(agent_categories, gold_categories) if a == g)
    return matches / len(agent_categories)


def _fields_match(field: str, agent_value: Any, gold_value: Any) -> bool:

    if field == "drinks":
        from conversions import alcohol_table_to_grams_per_day
        try:
            agent_gpd = alcohol_table_to_grams_per_day(agent_value or [])
        except (ValueError, KeyError, TypeError):
            return False
        try:
            gold_gpd = alcohol_table_to_grams_per_day(gold_value or [])
        except (ValueError, KeyError, TypeError):
            gold_gpd = 0.0
        tolerance = max(2.0, 0.10 * gold_gpd)
        return abs(agent_gpd - gold_gpd) <= tolerance
    return agent_value == gold_value


def per_field_accuracy(agent_values: Dict[str, Any], gold_values: Dict[str, Any],
                        fields: List[str]) -> Dict[str, bool]:
    return {f: _fields_match(f, agent_values.get(f), gold_values.get(f)) for f in fields}


def all_fields_correct(agent_values: Dict[str, Any], gold_values: Dict[str, Any],
                        fields: List[str]) -> bool:
    return all(_fields_match(f, agent_values.get(f), gold_values.get(f)) for f in fields)



def write_results_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_results_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)