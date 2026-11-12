"""
The conversational agent

  1. GATHER    Agent chats naturally toward the section's goals. Each user
               turn is parsed (schema-constrained) and merges into the section. checklist decides when all are present
  2. VERIFY    Once complete: fidelity (LLM re-reads transcript) + plausibility
               (deterministic). Hard issues block; soft issues are confirmed.
  3. SUBMIT    Completed + verified section -> CanRisk pedigree -> real API.
  4. REPORT    API JSON -> plain-language summary in the conversation.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import date

import llm
import plausibility
from schema import PersonalLifestyleSection
from verification import verify_section
from canrisk_test_call import get_auth_token, call_endpoint
from run_agent_pedigree import build_pedigree_from_profile
from conversions import PersonalLifestyleProfile
from agent_io import ask_user


def _api_messages(transcript: List[dict]) -> List[dict]:
    msgs = [{"role": m["role"], "content": m["content"]} for m in transcript]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


def extract_turn(section, transcript: List[dict], extra_instructions: str = None,
                 model: str = None, temperature: float = None,
                 provider_override: str = None) -> dict:
    tool = section.extraction_tool_schema()
    recorded = section.confirmed_values()
    latest_user_msg = next(
        (m["content"] for m in reversed(transcript) if m["role"] == "user"), ""
    )
    system = (
        "You are the extraction step of a clinician-facing cancer-risk intake "
        "agent. " + section.goals_text() + "\n\n"
        f"Fields already recorded (context only -- do NOT re-derive these): "
        f"{json.dumps(recorded, default=str)}\n\n"
        "Extract ONLY fields that the patient's MOST RECENT message (quoted "
        "below) actually specifies or changes. Use earlier turns in the "
        "conversation solely to interpret that latest message -- resolving "
        "units, pronouns, or references like 'that's right' -- never to "
        "re-derive a field that already has a recorded value above unless "
        "the most recent message explicitly restates or updates it. If the "
        "most recent message doesn't touch a field, leave it out of your "
        "extraction entirely; do not re-guess it from older parts of the "
        "conversation.\n\n"
        f"Most recent patient message: {latest_user_msg!r}\n\n"
        "AMBIGUITY -- READ CAREFULLY, this is commonly missed: if the "
        "patient's answer for a field contains ANY hedge -- words like "
        "'around', 'about', 'roughly', 'approximately', 'maybe', 'ish', "
        "'I think', 'somewhere between', or a stated range like '40-45' -- "
        "you MUST put that field's name in ambiguous_fields, even if you "
        "could technically infer a specific number from it. Do NOT quietly "
        "pick a point value from a hedge and treat the field as settled -- "
        "flagging it is what lets the agent ask a clarifying follow-up, "
        "which is the correct behavior here, not a fallback. Only treat a "
        "field as settled when the patient stated a precise value directly. "
        "Convert units (feet/inches->cm, lb->kg). For date of birth: only "
        "fill dob_iso if the patient gave an actual date; if they gave only "
        "an age, fill the age field and NEVER invent a birth date."
    )
    if extra_instructions:
        system += "\n\n" + extra_instructions
    kwargs = {}
    if model is not None:
        kwargs["model"] = model
    if temperature is not None:
        kwargs["temperature"] = temperature
    if provider_override is not None:
        kwargs["provider_override"] = provider_override
    primary = llm.call_with_tool(system, _api_messages(transcript), tool, **kwargs)

    corrections = scan_for_corrections(section, transcript, extra_instructions,
                                       model, temperature, provider_override)
    if corrections:
        primary = dict(primary)
        primary.update(corrections)  # dedicated scan wins on conflict
    return primary


def _correction_scan_tool(section) -> dict:
    known_fields = [f.name for f in section.fields]
    return {
        "name": "record_corrections",
        "description": (
            "Find corrections to fields that ALREADY have a recorded value. "
            "Do not extract information for fields that are still missing -- "
            "that is handled by a different step. Your only job: did the "
            "patient's most recent message state a DIFFERENT value for "
            "something already on record, anywhere in the message, even in "
            "passing while answering something else?"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "corrections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string", "enum": known_fields},
                            "corrected_value": {
                                "description": ("The new, corrected value, in the "
                                               "same format/units the field normally uses.")
                            },
                        },
                        "required": ["field", "corrected_value"],
                    },
                    "description": ("Every already-recorded field the patient just "
                                   "restated with a value DIFFERENT from what's "
                                   "on record. Empty array if there's no correction "
                                   "in this message."),
                }
            },
            "required": ["corrections"],
        },
    }


def scan_for_corrections(section, transcript: List[dict], extra_instructions: str = None,
                         model: str = None, temperature: float = None,
                         provider_override: str = None) -> dict:
    recorded = {k: v for k, v in section.values.items()}
    if not recorded:
        return {}
    latest_user_msg = next(
        (m["content"] for m in reversed(transcript) if m["role"] == "user"), "")
    if not latest_user_msg:
        return {}

    tool = _correction_scan_tool(section)
    system = (
        "You are the correction-detection step of a clinician-facing intake "
        "agent. Your ONLY job is to check whether the patient's most recent "
        "message restates a DIFFERENT value for something already on "
        "record -- for example 'oh, I said 65kg earlier, it's actually "
        "68kg', or a correction mentioned in passing while the patient is "
        "really answering a different question. This is exactly the case "
        "that's easy to miss when focused on the main question, so look "
        "carefully at the whole message, not just its main topic.\n\n"
        f"Fields already recorded: {json.dumps(recorded, default=str)}\n\n"
        f"Most recent patient message: {latest_user_msg!r}\n\n"
        "If the message doesn't correct anything, return an empty array. "
        "Never flag a field just because it was mentioned again -- only if "
        "the STATED value is actually different from what's recorded above."
    )
    if extra_instructions:
        system += "\n\n" + extra_instructions
    kwargs = {}
    if model is not None:
        kwargs["model"] = model
    if temperature is not None:
        kwargs["temperature"] = temperature
    if provider_override is not None:
        kwargs["provider_override"] = provider_override
    result = llm.call_with_tool(system, _api_messages(transcript), tool, **kwargs)
    out = {}
    for c in (result.get("corrections") or []):
        field, value = c.get("field"), c.get("corrected_value")
        if field and field in recorded and value is not None:
            out[field] = value
    return out


def _ensure_question(reply: str, fallback_question: str) -> str:
    if "?" in reply:
        return reply
    reply = reply.strip()
    return (reply + " " if reply else "") + fallback_question


def next_question(section, transcript: List[dict]) -> str:
    """Generate the agent's next natural-language turn, steered toward whatever
    required fields are still missing."""
    missing = section.missing_required()
    system = (
        "You are a warm, clear clinician-facing intake assistant collecting "
        "information to run a breast/ovarian cancer risk model. " +
        section.goals_text() + "\n\n"
        f"Still needed: {missing}. Ask about these naturally -- you may cover "
        "more than one in a turn if it flows, and acknowledge what the patient "
        "just told you. Keep it conversational, not a form. Ask only about "
        "information, never give medical advice or risk interpretation yet.\n\n"
        "Your reply MUST end with an actual, direct question about one of the "
        "still-needed items above. An acknowledgment on its own ('Great, "
        "thanks for sharing that.') is NOT a complete reply, even if it "
        "sounds polite and conversational -- the patient needs to know "
        "specifically what to answer next."
    )
    reply = llm.call_text(system, _api_messages(transcript))
    descs = []
    for name in missing:
        field = next((f for f in section.fields if f.name == name), None)
        descs.append(field.description if field else name.replace("_", " "))
    fallback = "Could you tell me: " + "; ".join(descs) + "?" if descs else \
               "Could you tell me a bit more?"
    return _ensure_question(reply, fallback)


def compose_ambiguous_followup(section, transcript: List[dict],
                                ambiguous_fields: List[str]) -> str:
    descs = []
    for name in ambiguous_fields:
        field = next((f for f in section.fields if f.name == name), None)
        descs.append(f"{name} -- {field.description}" if field else name)
    system = (
        "You are a warm, clear clinician-facing intake assistant collecting "
        "information to run a breast/ovarian cancer risk model. The patient "
        "just gave an imprecise or hedged answer, and a PRECISE value is "
        "still needed, for exactly this: " + "; ".join(descs) + "\n\n"
        "Ask a short, specific follow-up requesting an exact value for "
        "EXACTLY this -- do not ask about anything else, and do not just "
        "acknowledge their answer without asking again."
    )
    reply = llm.call_text(system, _api_messages(transcript))
    fallback = "Could you give me a precise value for: " + "; ".join(descs) + "?"
    return _ensure_question(reply, fallback)


def compose_question(section, transcript: List[dict]) -> str:
    if not getattr(section, "ask_one_at_a_time", False):
        return next_question(section, transcript)

    missing = section.missing_required()
    if not missing:
        return next_question(section, transcript)

    group = next((g for g in getattr(section, "grouped_fields", [])
                 if g and g[0] == missing[0] and all(f in missing for f in g)),
                 None)
    targets = group if group else [missing[0]]

    descs = []
    for target in targets:
        field = next((f for f in section.fields if f.name == target), None)
        if field is not None:
            desc = field.description
            if field.enum:
                desc += " Options: " + ", ".join(field.enum) + "."
        else:
            synthetic = getattr(section, "synthetic_field_descriptions", {})
            desc = synthetic.get(target, target.replace("_", " "))
        descs.append(f"{target} -- {desc}")

    if len(targets) > 1:
        ask_clause = ("Ask about EXACTLY these things together, in one "
                      "natural question: " + "; ".join(descs))
    else:
        ask_clause = f"Ask about EXACTLY ONE thing: {descs[0]}"

    system = (
        "You are a warm, clear clinician-facing intake assistant collecting "
        "information to run a breast/ovarian cancer risk model.\n\n"
        f"{ask_clause}\n\n"
        "Do NOT ask about any other field beyond what's listed above, and do "
        "not stack unrelated questions into one turn. Acknowledge briefly "
        "what the patient just told you if relevant, then ask. Keep it "
        "conversational and short. Never give medical advice or risk "
        "interpretation.\n\n"
        "Your reply MUST end with an actual, direct question about the item "
        "above. An acknowledgment on its own ('Great, thanks for sharing "
        "that.') is NOT a complete reply, even if it sounds polite and "
        "conversational -- the patient needs to know specifically what to "
        "answer next."
    )
    reply = llm.call_text(system, _api_messages(transcript))
    fallback = "Could you tell me: " + "; ".join(descs) + "?"
    return _ensure_question(reply, fallback)


def resolve_field(section, field: str, user_reply: str) -> dict:
    tool = section.extraction_tool_schema()
    unit_hint = {
        "height_cm": "centimeters (cm)",
        "weight_kg": "kilograms (kg)",
    }.get(field, "its native unit")
    system = (
        "You are resolving a single ambiguous field in a medical intake form. "
        f"The field in question is '{field}'. The patient was just asked to "
        f"clarify this value and replied: '{user_reply}'. "
        "Extract ONLY this field's value using the tool, converting units as "
        "needed (feet/inches -> cm, lb -> kg). If the reply states an explicit "
        f"unit, use it. If the reply is a bare number with no unit, treat it as "
        f"already being in {unit_hint} -- do not reinterpret it against any "
        "earlier statement in the conversation. Do not extract any other field."
    )
    return llm.call_with_tool(system, [{"role": "user", "content": user_reply}], tool)


YES_NO_TOOL = {
    "name": "report_yes_no",
    "description": (
        "Classify whether the user's reply is a clean affirmative confirmation "
        "of the stated content (agreeing it's correct/complete, nothing to "
        "add or change) versus anything else -- a correction, a disagreement, "
        "or NEW information not already reflected in what they're confirming. "
        "Volunteering additional information ('I have X to report', 'also...') "
        "is NOT affirmative even when phrased cooperatively or positively -- "
        "it means the stated content is incomplete."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_affirmative": {
                "type": "boolean",
                "description": ("True only if the reply confirms the stated "
                                "content is correct AS-IS, with nothing to "
                                "add, change, or correct."),
            },
        },
        "required": ["is_affirmative"],
    },
}

_FAST_YES = {"yes", "y", "yep", "yeah", "yea", "yup", "correct", "right",
             "indeed", "confirmed", "sure", "ok", "okay", "that's right",
             "thats right", "that is correct", "sounds right", "looks right",
             "lgtm", "affirmative", "correct!"}
_FAST_NO = {"no", "n", "nope", "nah", "wrong", "incorrect", "not right"}


def is_affirmative(resp: str, context: str = "") -> bool:
    cleaned = resp.strip().lower().rstrip("!.")
    if cleaned in _FAST_YES:
        return True
    if cleaned in _FAST_NO:
        return False
    if llm.available():
        system = "Classify the user's reply to a yes/no confirmation question."
        if context:
            system += f"\n\nWhat the user was asked to confirm:\n{context}"
        result = llm.call_with_tool(system, [{"role": "user", "content": resp}], YES_NO_TOOL)
        return bool(result.get("is_affirmative", False))
    return False  # offline, ambiguous reply: don't assume agreement


def gather_until_complete(section, transcript: List[dict]) -> None:
    while not section.is_complete():
        user_msg = ask_user("You: ")
        transcript.append({"role": "user", "content": user_msg})
        extracted = extract_turn(section, transcript)
        ambiguous = apply_extraction(section, extracted)
        if ambiguous:
            agent_turn = compose_ambiguous_followup(section, transcript, ambiguous)
            print(f"Agent: {agent_turn}")
            transcript.append({"role": "assistant", "content": agent_turn})
        elif not section.is_complete():
            agent_turn = compose_question(section, transcript)
            print(f"Agent: {agent_turn}")
            transcript.append({"role": "assistant", "content": agent_turn})


def verify_until_clean(section, transcript: List[dict],
                        confirmed_plausibility: Optional[set] = None,
                        resolved_fidelity: Optional[set] = None) -> tuple:
    print("\nAgent: Thanks -- let me just double-check everything before I run this.")
    confirmed_plausibility = set() if confirmed_plausibility is None else confirmed_plausibility
    resolved_fidelity = set() if resolved_fidelity is None else resolved_fidelity

    while True:
        verdict = verify_section(section, transcript)

        if verdict["blocking_plausibility"]:
            print("Agent: Some values can't be right, so I need to fix them first:")
            corrected_fields = set()
            for i in verdict["blocking_plausibility"]:
                print(f"  - {i.message}")
                user_msg = ask_user(f"  Correct value for {i.field}: ")
                transcript.append({"role": "user", "content":
                                   f"For {i.field}, the correct value is: {user_msg}"})
                extracted = resolve_field(section, i.field, user_msg)
                apply_extraction(section, extracted)
                corrected_fields.add(i.field)
            for f in corrected_fields:
                confirmed_plausibility.discard(f)
                resolved_fidelity.discard(f)
            continue  # re-verify from scratch

        fidelity_mismatches = [m for m in (verdict["fidelity"].get("mismatches") or [])
                               if m["field"] not in resolved_fidelity]
        unresolved_soft = [i for i in verdict["confirm_plausibility"]
                           if i.field not in confirmed_plausibility]

        if not fidelity_mismatches and not unresolved_soft:
            break  # nothing left to address -- clean

        touched_this_pass = set()

        for m in fidelity_mismatches:
            field = m["field"]
            if field in resolved_fidelity or field in touched_this_pass:
                continue  # already handled earlier in this same pass
            print(f"Agent: I recorded {field} as '{m['recorded_value']}', but it "
                  f"sounded like you said '{m['what_patient_said']}'. Which is right?\n"
                  f"  Reply '1' to keep {m['recorded_value']}, '2' to use "
                  f"{m['what_patient_said']}, or type the correct value.")
            resp = ask_user("You: ").strip()
            if resp == "1":
                pass  # keep the currently-recorded value as-is
            else:
                user_reply = m["what_patient_said"] if resp == "2" else resp
                transcript.append({"role": "user", "content":
                                   f"Clarification for {field}: {user_reply}"})
                extracted = resolve_field(section, field, user_reply)
                apply_extraction(section, extracted)
            resolved_fidelity.add(field)
            confirmed_plausibility.discard(field)
            if field == "height_cm" or field == "weight_kg":
                confirmed_plausibility.discard("bmi")  # derived from both -- don't
                resolved_fidelity.discard("bmi")        # let a stale bmi confirm hide a new mismatch
            touched_this_pass.add(field)

        for i in unresolved_soft:
            if i.field in confirmed_plausibility or i.field in touched_this_pass:
                continue  # already handled earlier in this same pass
            print(f"Agent: {i.message}")
            resp = ask_user("  Is that correct? (yes / or type the correct value): ").strip()
            if is_affirmative(resp, context=(
                    f"Flagged value: {i.message}\nQuestion asked: 'Is that correct?'")):
                confirmed_plausibility.add(i.field)
            else:
                transcript.append({"role": "user", "content":
                                   f"Correction for {i.field}: {resp}"})
                extracted = resolve_field(section, i.field, resp)
                apply_extraction(section, extracted)
                resolved_fidelity.discard(i.field)  # value changed; let fidelity
                                                      # look at it fresh too
                if i.field == "height_cm" or i.field == "weight_kg":
                    confirmed_plausibility.discard("bmi")
                    resolved_fidelity.discard("bmi")
            touched_this_pass.add(i.field)

        # loop back and re-verify with whatever changed

    return confirmed_plausibility, resolved_fidelity


def correct_at_teach_back(section, transcript: List[dict]) -> set:
    print("Agent: No problem -- what would you like to correct?")
    all_touched: set = set()
    while True:
        correction = ask_user("You: ")
        if is_affirmative(correction, context=(
                "The user was asked what they'd like to correct about their "
                "intake summary. A reply like 'nevermind' or 'it's fine' "
                "means no correction is needed; anything describing an actual "
                "change or new information is NOT affirmative.")):
            return all_touched  # changed their mind -- nothing further to correct

        transcript.append({"role": "user", "content": correction})
        extracted = extract_turn(section, transcript)
        fields_touched = [k for k in extracted if k != "ambiguous_fields"]
        ambiguous = apply_extraction(section, extracted)
        all_touched.update(fields_touched)
        if any(f in ("height_cm", "weight_kg") for f in fields_touched):
            all_touched.add("bmi")  # derived from both -- invalidate too

        if not fields_touched and not ambiguous:
            print("Agent: I didn't catch a specific change there -- which "
                  "field would you like to update, and what should it be?")
            continue

        if ambiguous:
            followup = compose_ambiguous_followup(section, transcript, ambiguous)
            print(f"Agent: {followup}")
            transcript.append({"role": "assistant", "content": followup})
            continue  # still ambiguous -- keep gathering before moving on

        return all_touched  # got a concrete correction -- caller will re-verify it


def apply_extraction(section, extracted: dict) -> List[str]:
    ambiguous = extracted.pop("ambiguous_fields", [])
    before_unconfirmed = set(section.unconfirmed)
    touched_this_call = set()
    for name, value in extracted.items():
        if section.set_value(name, value):
            touched_this_call.add(name)
    for f in before_unconfirmed - touched_this_call:
        section.mark_confirmed(f)
    return ambiguous



def submit_section(section) -> dict:
    from pedigree import Pedigree
    from canrisk_format import build_pedigree_file, validate_pedigree_file

    username = os.environ.get("CANRISK_USERNAME")
    password = os.environ.get("CANRISK_PASSWORD")
    if not username or not password:
        return {"error": "Set CANRISK_USERNAME / CANRISK_PASSWORD to submit."}

    ped = Pedigree()
    ped.proband.sex = section.values.get("sex")
    ay = section.age_yob()
    if ay:
        ped.proband.age, ped.proband.year_of_birth = ay

    pedigree_text = build_pedigree_file(ped, personal=section)
    problems = validate_pedigree_file(pedigree_text)
    if problems:
        return {"error": "Pedigree failed local validation:\n  - "
                          + "\n  - ".join(problems)}

    out = section.to_canrisk()
    try:
        token = get_auth_token(username, password)
    except Exception as e:
        return {"error": f"CanRisk authentication failed: {e}"}
    breast = call_endpoint("boadicea", token, pedigree_text)
    ovarian = call_endpoint("ovarian", token, pedigree_text)
    return {"breast": breast, "ovarian": ovarian,
            "unresolved": out.get("unresolved"), "pedigree": pedigree_text}


def _legacy_submit_section_with_fabricated_family(section) -> dict:

    username = os.environ.get("CANRISK_USERNAME")
    password = os.environ.get("CANRISK_PASSWORD")
    if not username or not password:
        return {"error": "Set CANRISK_USERNAME / CANRISK_PASSWORD to submit."}

    out = section.to_canrisk()
    p = out["proband_overrides"]
    age, yob = section.age_yob()
    # if a real birth date was given, use it; otherwise carry the year of birth
    # via Jan-1 purely as an internal integer-carrier for the pedigree builder
    # (BOADICEA bins on year; this is never presented as the patient's birthday)
    dob = (date.fromisoformat(section.values["dob_iso"])
           if section.values.get("dob_iso") else date(yob, 1, 1))
    profile = PersonalLifestyleProfile(
        sex=p["sex"], country=section.values.get("country", "UK"),
        dob=dob,
        height_cm=section.values["height_cm"], weight_kg=section.values["weight_kg"],
        drinks=section.values.get("drinks") or [],
    )
    pedigree = build_pedigree_from_profile(profile)

    token = get_auth_token(username, password)
    breast = call_endpoint("boadicea", token, pedigree)
    ovarian = call_endpoint("ovarian", token, pedigree)
    return {"breast": breast, "ovarian": ovarian, "unresolved": out["unresolved"]}


def summarize_results(results: dict, use_llm: bool = True) -> str:

    if "error" in results:
        return results["error"]

    def lifetime(model_result):
        try:
            return model_result["pedigree_result"][0]["lifetime_cancer_risk"][0]
        except (KeyError, IndexError, TypeError):
            return None

    breast_lt = lifetime(results.get("breast", {}))
    ovarian_lt = lifetime(results.get("ovarian", {}))
    facts = {"breast_lifetime": breast_lt, "ovarian_lifetime": ovarian_lt,
             "unresolved": results.get("unresolved")}

    def risk_key(d):
        return next((k for k in d if "risk" in k), None)

    if not (use_llm and llm.available()):
        parts = ["Here are your results (informational, for review with a clinician):"]
        if breast_lt:
            parts.append(f"- Breast cancer risk to {breast_lt['age']}: {breast_lt[risk_key(breast_lt)]['percent']}%")
        if ovarian_lt:
            parts.append(f"- Ovarian cancer risk to {ovarian_lt['age']}: {ovarian_lt[risk_key(ovarian_lt)]['percent']}%")
        return "\n".join(parts)

    system = (
        "You are delivering cancer-risk model results to a patient in plain, "
        "calm language. Use ONLY the numbers provided -- do not invent or "
        "reinterpret. Explain what a lifetime risk figure means simply, note "
        "the result is informational and should be discussed with their "
        "clinician, and do NOT give management advice or reassurance/alarm "
        "beyond the numbers. Keep it short."
    )
    return llm.call_text(system, [{"role": "user", "content": json.dumps(facts, default=str)}])


def run(section=None):
    if not llm.available():
        print("No LLM provider. Set OPENAI_API_KEY (or ANTHROPIC_API_KEY), "
              "optionally LLM_PROVIDER=openai|anthropic.")
        return
    print(f"[using LLM provider: {llm.provider()}]")
    if plausibility.pycountry is None:
        print("[WARNING: pycountry is not installed -- country plausibility "
              "checking is DISABLED for this run. Fake countries will NOT be "
              "flagged. Run: pip install pycountry --break-system-packages]")
    section = section or PersonalLifestyleSection()
    transcript: List[dict] = []

    opening = "Hi -- let's start: what's your sex, and which country do you live in?"
    print(f"Agent: Hi -- I'll ask a few questions to set up a breast and ovarian "
          f"cancer risk assessment. {opening}")
    transcript.append({"role": "assistant", "content": opening})

    gather_until_complete(section, transcript)

    confirmed_plausibility, resolved_fidelity = set(), set()
    while True:
        confirmed_plausibility, resolved_fidelity = verify_until_clean(
            section, transcript, confirmed_plausibility, resolved_fidelity)

        summary = section.teach_back()
        print("\n" + summary)
        print("\nAgent: Does all of that look correct? (yes to run the assessment)")
        resp = ask_user("You: ")
        if is_affirmative(resp, context=(
                f"{summary}\n\nQuestion asked: 'Does all of that look correct?' "
                "If the reply mentions anything not already reflected above "
                "(e.g. reporting alcohol use when it says 'none reported'), "
                "that is a correction, not a confirmation.")):
            break

        touched = correct_at_teach_back(section, transcript)
        for f in touched:
            confirmed_plausibility.discard(f)
            resolved_fidelity.discard(f)
        if not section.is_complete():
            print("Agent: Got it -- I still need a bit more info before I can re-check that.")
            gather_until_complete(section, transcript)


    print("\nAgent: Running your assessment...")
    results = submit_section(section)
    print("\n" + summarize_results(results))

    if "breast" in results or "ovarian" in results:
        from report import build_comprehensive_report
        print("\n" + "=" * 60)
        print(build_comprehensive_report(results))
        print("=" * 60)
        print("[full raw API responses saved to last_boadicea_response.json "
              "and last_ovarian_response.json]")


if __name__ == "__main__":
    run()