
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional

import llm
from harness import GroundTruthRespondentBase


def _missing_target_fields(section) -> List[str]:
    missing = section.missing_required()
    if not missing:
        return []
    group = next((g for g in getattr(section, "grouped_fields", [])
                 if g and g[0] == missing[0] and all(f in missing for f in g)),
                 None)
    return group if group else [missing[0]]


SYNTHETIC_FIELD_ALIASES: Dict[str, List[str]] = {
    "date_of_birth_or_age": ["age", "dob_iso"],           # PersonalLifestyleSection
    "age_at_first_period": ["menarche_age", "menarche_unknown"],  # WomensHealthSection
    "drink_details": ["drinks"],                           # AlcoholSection
    "children_details": ["children"],                      # ChildrenSection
}


def _resolve_payload(profile: Dict[str, Any], target_fields: List[str]) -> Dict[str, Any]:

    payload: Dict[str, Any] = {}
    for target in target_fields:
        if target in profile:
            payload[target] = profile.get(target)
            continue
        for alias in SYNTHETIC_FIELD_ALIASES.get(target, []):
            if alias in profile and profile.get(alias) is not None:
                payload[alias] = profile.get(alias)
                break

    return payload


def _phrase_truthful_answer(profile: Dict[str, Any], target_fields: List[str],
                             agent_question: str) -> str:
    values = _resolve_payload(profile, target_fields)
    if not llm.available():
        return "; ".join(f"{k}: {v}" for k, v in values.items())
    system = (
        "You are role-playing a patient answering a clinician's intake "
        "question, in natural conversational language (not a form). Only "
        "state the values given below -- do not add, omit, or alter any of "
        "them, and do not volunteer any other information."
    )
    user = (f"Clinician's question: {agent_question!r}\n"
            f"Values you must convey: {json.dumps(values, default=str)}\n"
            "Reply with ONLY the patient's spoken answer, nothing else.")
    return llm.call_text(system, [{"role": "user", "content": user}],
                         model=llm.reference_model())  # simulator: pinned model


@dataclass
class OneShotRespondent(GroundTruthRespondentBase):
    narrative: Optional[str] = None
    _sent: bool = False

    def next_message(self, transcript, section) -> Optional[str]:
        if not self._sent:
            self._sent = True
            if self.narrative:
                return self.narrative
            if not llm.available():
                return "; ".join(f"{k}: {v}" for k, v in self.profile.items())
            system = (
                "Write a short, natural free-text clinical narrative a patient "
                "might type in one go, conveying exactly and only the following "
                "profile fields -- no more, no less, no invented details."
            )
            user = json.dumps(self.profile, default=str)
            return llm.call_text(system, [{"role": "user", "content": user}],
                                 model=llm.reference_model())  # simulator: pinned

        targets = _missing_target_fields(section)
        if not targets:
            return None
        last_question = next(
            (m["content"] for m in reversed(transcript) if m["role"] == "assistant"), "")
        return _phrase_truthful_answer(self.profile, targets, last_question)


@dataclass
class ScriptedTruthfulRespondent(GroundTruthRespondentBase):

    def next_message(self, transcript, section) -> Optional[str]:
        targets = _missing_target_fields(section)
        if not targets:
            return None
        last_question = next(
            (m["content"] for m in reversed(transcript) if m["role"] == "assistant"), "")
        return _phrase_truthful_answer(self.profile, targets, last_question)


class StructuredInputRespondent:
    uses_extraction = False

    @staticmethod
    def apply_structured(section, profile: Dict[str, Any]) -> None:
        for name, value in profile.items():
            section.set_value(name, value)

BEHAVIOR_TRUTHFUL = "truthful"
BEHAVIOR_WRONG_THEN_CORRECTED = "wrong_then_corrected"
BEHAVIOR_MULTI_FACT = "multi_fact"
BEHAVIOR_AMBIGUOUS = "ambiguous"

ALL_BEHAVIORS = [BEHAVIOR_TRUTHFUL, BEHAVIOR_WRONG_THEN_CORRECTED,
                  BEHAVIOR_MULTI_FACT, BEHAVIOR_AMBIGUOUS]

DEFAULT_BEHAVIOR_WEIGHTS = {
    BEHAVIOR_TRUTHFUL: 0.5,
    BEHAVIOR_WRONG_THEN_CORRECTED: 0.2,
    BEHAVIOR_MULTI_FACT: 0.15,
    BEHAVIOR_AMBIGUOUS: 0.15,
}


def _wrong_value_for(field: str, correct_value: Any) -> Any:
    if isinstance(correct_value, (int, float)):
        return correct_value * 1.15 + 1
    return f"NOT_{correct_value}"


@dataclass
class SimulatedPatientAgent(GroundTruthRespondentBase):
    behavior_weights: Dict[str, float] = dc_field(
        default_factory=lambda: dict(DEFAULT_BEHAVIOR_WEIGHTS))
    rng: random.Random = dc_field(default_factory=random.Random)

    behavior_log: List[Dict[str, Any]] = dc_field(default_factory=list)
    _pending_correction: Optional[Dict[str, Any]] = None

    def _sample_behavior(self) -> str:
        choices = list(self.behavior_weights.keys())
        weights = list(self.behavior_weights.values())
        return self.rng.choices(choices, weights=weights, k=1)[0]

    def next_message(self, transcript, section) -> Optional[str]:
        targets = _missing_target_fields(section)
        if not targets and self._pending_correction is None:
            return None

        last_question = next(
            (m["content"] for m in reversed(transcript) if m["role"] == "assistant"), "")

        resolved_payload = _resolve_payload(self.profile, targets)
        resolved_fields = list(resolved_payload.keys()) or targets

        behavior = self._sample_behavior() if targets else BEHAVIOR_TRUTHFUL
        payload: Dict[str, Any] = dict(resolved_payload)
        note_correction = None

        if behavior == BEHAVIOR_WRONG_THEN_CORRECTED and resolved_fields:
            fld = resolved_fields[0]
            wrong = _wrong_value_for(fld, self.profile.get(fld))
            payload[fld] = wrong
            self._pending_correction = {"field": fld, "correct": self.profile.get(fld)}
            self.behavior_log.append({"field": fld, "behavior": behavior,
                                       "wrong_value_given": wrong})
        elif behavior == BEHAVIOR_MULTI_FACT and resolved_fields:
            still_unset = [f for f in section.required_fields()
                           if f not in section.values and f not in targets]
            extra = still_unset[:1]
            for f in extra:
                payload[f] = self.profile.get(f)
            self.behavior_log.append({"field": resolved_fields[0], "behavior": behavior,
                                       "bundled_extra": extra})
        elif behavior == BEHAVIOR_AMBIGUOUS and resolved_fields:
            fld = resolved_fields[0]
            self.behavior_log.append({"field": fld, "behavior": behavior})
        else:
            behavior = BEHAVIOR_TRUTHFUL
            self.behavior_log.append({"field": resolved_fields[0] if resolved_fields else None,
                                       "behavior": behavior})

        if self._pending_correction is not None and behavior != BEHAVIOR_WRONG_THEN_CORRECTED:
            note_correction = self._pending_correction
            self._pending_correction = None

        return self._phrase(last_question, payload, behavior, note_correction)

    def _phrase(self, agent_question: str, payload: Dict[str, Any], behavior: str,
                note_correction: Optional[Dict[str, Any]]) -> str:
        if not llm.available():
            base = "; ".join(f"{k}: {v}" for k, v in payload.items())
            if note_correction:
                base += (f" (also, correction: {note_correction['field']} is "
                         f"actually {note_correction['correct']})")
            return base

        instructions = {
            BEHAVIOR_TRUTHFUL: "Answer truthfully and directly, nothing extra.",
            BEHAVIOR_WRONG_THEN_CORRECTED: (
                "State the given value as your answer (it is intentionally "
                "the WRONG value for this simulation -- state it as fact, "
                "don't hedge or flag it as wrong)."),
            BEHAVIOR_MULTI_FACT: (
                "Answer the main question, and unprompted, also mention the "
                "extra field(s) given, in the same turn, the way a real "
                "patient might volunteer related information."),
            BEHAVIOR_AMBIGUOUS: (
                "Give an imprecise, range-bound, or hedged version of the "
                "value instead of a precise one (e.g. 'around 40', "
                "'somewhere in my 40s', 'a few years ago') -- never state "
                "the exact value."),
        }[behavior]

        system = (
            "You are role-playing a patient answering a clinician intake "
            "question, in natural conversational language. " + instructions
        )
        user_parts = [f"Clinician's question: {agent_question!r}",
                      f"Value(s) to convey: {json.dumps(payload, default=str)}"]
        if note_correction:
            user_parts.append(
                f"Additionally, in this SAME turn, correct an earlier answer: "
                f"{note_correction['field']} is actually "
                f"{note_correction['correct']} (you previously misstated it).")
        user_parts.append("Reply with ONLY the patient's spoken answer.")
        return llm.call_text(system, [{"role": "user", "content": "\n".join(user_parts)}],
                             model=llm.reference_model())  # simulator: pinned model

    def confirm_or_correct(self, field: str, message: str) -> str:
        return str(self.profile.get(field, "yes"))

    def pending_disclosure(self) -> bool:
        return self._pending_correction is not None