
import json
import re
from typing import List, Dict, Any, Optional

import llm
from conversions import lbs_to_kg, feet_inches_to_cm
from plausibility import (
    check_field_plausibility, check_cross_field, summarize_issues, PlausibilityIssue,
    check_missing_age,
)

def run_plausibility(section) -> dict:
    """Runs per-field + cross-field plausibility on a completed section.
    Returns {hard: [...], soft: [...], has_blocking: bool}."""
    issues: List[PlausibilityIssue] = []

    # per-field numeric checks
    for f in section.fields:
        if f.plausibility_key:
            issue = check_field_plausibility(f.plausibility_key, section.values.get(f.name))
            if issue:
                issues.append(issue)

    # cross-field checks (section exposes the relevant values)
    if hasattr(section, "plausibility_values"):
        pv = section.plausibility_values()
        issues.extend(check_cross_field(pv))
        if "current_age" in pv:
            missing_age_issue = check_missing_age(pv.get("current_age"))
            if missing_age_issue:
                issues.append(missing_age_issue)

    return summarize_issues(issues)


FIDELITY_TOOL = {
    "name": "report_fidelity_check",
    "description": (
        "Report whether each extracted value faithfully matches what the "
        "patient actually said in the conversation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mismatches": {
                "type": "array",
                "description": "Fields where the recorded value does NOT match the transcript",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "recorded_value": {"type": "string"},
                        "what_patient_said": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["field", "recorded_value", "what_patient_said"],
                },
            },
            "unconfirmed": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Fields whose recorded value can't be clearly traced to "
                    "anything the patient said (possible hallucination/default)."
                ),
            },
            "all_faithful": {
                "type": "boolean",
                "description": "True if every recorded value matches the transcript.",
            },
        },
        "required": ["all_faithful"],
    },
}


def run_fidelity(section, transcript: List[dict]) -> dict:
    """
    Re-reads the full transcript and checks it against the recorded values.
    transcript: list of {"role": "user"|"assistant", "content": str}.
    Returns the tool input dict (mismatches / unconfirmed / all_faithful).
    """
    convo_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in transcript)
    recorded = json.dumps(section.values, default=str, indent=2)

    system = (
        "You are an independent verification step in a medical intake system. "
        "Below is (a) the full patient conversation and (b) the structured "
        "values another component extracted from it. Your job is ONLY to check "
        "faithfulness: does each recorded value match what the patient actually "
        "said? Flag mismatches and any value you can't trace to the transcript. "
        "SUPERSEDING CORRECTIONS: if the patient stated a value and then later "
        "corrected or restated it (e.g. 'actually it's 71', 'I misspoke, make "
        "that 12'), the LATEST stated value is the truth -- the recorded value "
        "should match that latest value, and a recorded value that matches only "
        "an EARLIER, since-corrected statement IS a mismatch (report the "
        "corrected value as what the patient said), even though that earlier "
        "value does still literally appear in the transcript. "
        "IMPORTANT: a correct unit conversion is NOT a mismatch -- if the "
        "patient said '140 pounds' and the recorded value is '63.5' kg, that is "
        "faithful (140 lb = 63.5 kg); only flag it if the converted value is "
        "wrong or the underlying quantity differs. Recording a yes/no answer "
        "using its code is NOT a mismatch either: 'no' recorded as 'N', 'yes' "
        "as 'Y', and \"don't know\" as 'U' are all faithful. Likewise, categorizing a "
        "free-text description into the closest matching enum/category value "
        "is NOT a mismatch as long as it's a reasonable interpretation -- e.g. "
        "patient said 'beer' and the recorded drink type is 'bottle_beer' or "
        "'pint', or patient said 'spirits' and the recorded type is "
        "'spirit_shot'; only flag it if the category is actually wrong for "
        "what was said (e.g. patient said 'wine' but the recorded type is "
        "'spirit_shot'). Do not judge medical plausibility -- only fidelity "
        "to the conversation."
    )
    user = f"CONVERSATION:\n{convo_text}\n\nRECORDED VALUES:\n{recorded}"

    result = llm.call_with_tool(system, [{"role": "user", "content": user}], FIDELITY_TOOL)
    result = result or {"all_faithful": True, "mismatches": [], "unconfirmed": []}
    mismatches = _dedupe_mismatches(result.get("mismatches") or [])
    mismatches = _drop_false_positive_code_mismatches(mismatches)
    result["mismatches"] = _drop_false_positive_conversion_mismatches(mismatches)
    if not result["mismatches"] and not result.get("unconfirmed"):
        result["all_faithful"] = True
    return result


def _dedupe_mismatches(mismatches: List[dict]) -> List[dict]:
    seen = set()
    deduped = []
    for m in mismatches:
        key = (str(m.get("recorded_value", "")).strip().lower(),
               str(m.get("what_patient_said", "")).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)
    return deduped

_LB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?)\b", re.IGNORECASE)
_KG_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kgs?|kilograms?|kilogrammes?)\b", re.IGNORECASE)
_CM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:cm|centimeters?|centimetres?)\b", re.IGNORECASE)
_FT_IN_RE = re.compile(
    r"(\d+)\s*(?:'|feet|foot|ft)\.?\s*(\d+(?:\.\d+)?)?\s*(?:\"|''|in\.?|inch(?:es)?)?\b",
    re.IGNORECASE,
)


def _regex_parse_weight_kg(text: str) -> Optional[float]:
    m = _LB_RE.search(text)
    if m:
        return lbs_to_kg(float(m.group(1)))
    m = _KG_RE.search(text)
    if m:
        return float(m.group(1))
    return None


def _regex_parse_height_cm(text: str) -> Optional[float]:
    m = _FT_IN_RE.search(text)
    if m:
        feet = float(m.group(1))
        inches = float(m.group(2)) if m.group(2) else 0.0
        return feet_inches_to_cm(feet, inches)
    m = _CM_RE.search(text)
    if m:
        return float(m.group(1))
    return None


_REGEX_FIELD_PARSERS = {
    "weight_kg": _regex_parse_weight_kg,
    "height_cm": _regex_parse_height_cm,
}



QUANTITY_PARSE_TOOL = {
    "name": "report_stated_quantity",
    "description": (
        "Extract the literal number(s) and unit(s) stated in a phrase "
        "describing a weight or height -- e.g. '560 lbs', '5 foot 2 inches', "
        "'157.48 cm'. Report EXACTLY what was said. Do NOT convert units or "
        "do any arithmetic -- that is handled separately."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "found": {
                "type": "boolean",
                "description": "True if the phrase contains a parseable number+unit.",
            },
            "primary_value": {
                "type": "number",
                "description": ("The main number stated -- e.g. 5 in '5 feet 2 "
                                "inches', or 560 in '560 lbs'."),
            },
            "primary_unit": {
                "type": "string",
                "enum": ["kg", "lb", "cm", "m", "ft", "in"],
                "description": "Unit of primary_value, as literally stated.",
            },
            "secondary_value": {
                "type": "number",
                "description": ("For a compound feet+inches phrase only -- "
                                "e.g. 2 in '5 feet 2 inches'. Omit otherwise."),
            },
        },
        "required": ["found"],
    },
}


def _llm_parse_quantity(text: str) -> Optional[dict]:
    if not llm.available():
        return None
    system = (
        "Extract only the literal number(s) and unit(s) stated in this "
        "phrase, using the tool. Do not convert units. Do not guess a unit "
        "if none is stated -- set found=false instead."
    )
    result = llm.call_with_tool(system, [{"role": "user", "content": text}],
                                QUANTITY_PARSE_TOOL)
    return result if result and result.get("found") else None


def _expected_from_parsed_quantity(field: str, parsed: dict) -> Optional[float]:
    if not parsed:
        return None
    value, unit = parsed.get("primary_value"), parsed.get("primary_unit")
    if value is None or unit is None:
        return None
    value = float(value)

    if field == "weight_kg":
        if unit == "kg":
            return value
        if unit == "lb":
            return lbs_to_kg(value)
        return None

    if field == "height_cm":
        if unit == "cm":
            return value
        if unit == "m":
            return round(value * 100, 1)
        if unit == "ft":
            inches = parsed.get("secondary_value") or 0.0
            return feet_inches_to_cm(value, float(inches))
        if unit == "in":
            return feet_inches_to_cm(0, value)
        return None

    return None


def _is_correct_conversion(field: str, recorded_value, what_patient_said: str,
                            tolerance: float = 0.6) -> bool:
    regex_parser = _REGEX_FIELD_PARSERS.get(field)
    expected = regex_parser(what_patient_said) if regex_parser else None

    if expected is None and field in ("weight_kg", "height_cm"):
        parsed = _llm_parse_quantity(what_patient_said)
        expected = _expected_from_parsed_quantity(field, parsed)

    if expected is None:
        return False
    try:
        recorded = float(recorded_value)
    except (TypeError, ValueError):
        return False
    return abs(recorded - expected) <= tolerance


def _drop_false_positive_conversion_mismatches(mismatches: List[dict]) -> List[dict]:
    return [m for m in mismatches
            if not _is_correct_conversion(m.get("field", ""), m.get("recorded_value"),
                                          m.get("what_patient_said", ""))]



def verify_section(section, transcript: List[dict], use_llm: bool = True) -> dict:
    plausibility = run_plausibility(section)

    if use_llm and llm.available():
        fidelity = run_fidelity(section, transcript)
    else:
        fidelity = {"all_faithful": None, "mismatches": [], "unconfirmed": [],
                    "skipped": "LLM fidelity check skipped"}

    fidelity_ok = fidelity.get("all_faithful") is True or (
        not fidelity.get("mismatches") and not fidelity.get("unconfirmed")
    )

    return {
        "can_submit": (not plausibility["has_blocking"]) and fidelity_ok,
        "blocking_plausibility": plausibility["hard"],
        "confirm_plausibility": plausibility["soft"],
        "fidelity": fidelity,
    }


_CODE_SYNONYMS = {
    "Y": {"y", "yes", "yeah", "yep", "yup", "true", "correct", "affirmative",
          "i have", "i do", "positive"},
    "N": {"n", "no", "nope", "nah", "false", "never", "none", "i haven't",
          "i have not", "i don't", "i do not", "negative"},
    "U": {"u", "unknown", "not sure", "unsure", "don't know", "do not know",
          "dont know", "no idea", "not known", "maybe"},
}


def _normalize(text: str) -> str:
    return str(text).strip().strip(".!,").lower()


def _is_equivalent_code(recorded_value, what_patient_said: str) -> bool:
    rec = _normalize(recorded_value).upper()
    said = _normalize(what_patient_said)
    if rec not in _CODE_SYNONYMS:
        return False
    if said == rec.lower():
        return True
    return said in _CODE_SYNONYMS[rec]


def _drop_false_positive_code_mismatches(mismatches: List[dict]) -> List[dict]:
    return [m for m in mismatches
            if not _is_equivalent_code(m.get("recorded_value", ""),
                                       m.get("what_patient_said", ""))]