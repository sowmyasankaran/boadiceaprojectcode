

import json
from typing import Any, List, Optional

import llm
from agent import is_affirmative
from agent_io import ask_user

HOLISTIC_AUDIT_TOOL = {
    "name": "report_holistic_audit",
    "description": (
        "Report any issue that only becomes visible from reading the ENTIRE "
        "intake conversation as one continuous story, rather than any single "
        "section in isolation. A list of issues already caught by earlier, "
        "deterministic cross-section checks will be given to you -- do NOT "
        "re-report anything on that list, or anything confined to a single "
        "section's own internal consistency (already checked elsewhere). "
        "Only report something new that requires reading across the whole "
        "conversation to notice."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "cross_section_issues": {
                "type": "array",
                "description": (
                    "Narrative contradictions between what was said in "
                    "different parts of the conversation (including family "
                    "history) that are NOT already covered by the "
                    "already-caught list."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Plain description of the contradiction, "
                                           "quoting or paraphrasing both sides.",
                        },
                        "severity": {
                            "type": "string", "enum": ["blocking", "soft"],
                            "description": (
                                "'blocking' only if the record is internally "
                                "impossible (e.g. two different birth years "
                                "given for the same patient). 'soft' for "
                                "anything merely surprising or worth a human "
                                "glance."
                            ),
                        },
                    },
                    "required": ["description", "severity"],
                },
            },
            "story_coherence_issues": {
                "type": "array",
                "description": (
                    "Cases where every individual value is independently "
                    "plausible and faithful to its own section, but the "
                    "combination describes a clinically incoherent patient "
                    "history -- e.g. a bilateral mastectomy on record "
                    "alongside an active native-breast screening schedule."
                ),
                "items": {
                    "type": "object",
                    "properties": {"description": {"type": "string"}},
                    "required": ["description"],
                },
            },
            "all_clear": {
                "type": "boolean",
                "description": "True only if both arrays above are empty.",
            },
        },
        "required": ["all_clear"],
    },
}


def _transcript_text(transcript: List[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in transcript)


def _all_recorded_values(sections: List[Any]) -> dict:

    return {s.name: dict(s.values) for s in sections if hasattr(s, "values")}


def run_holistic_audit(sections: List[Any], transcript: List[dict],
                       ped: Optional[Any] = None,
                       existing_warnings: Optional[List[str]] = None,
                       use_llm: bool = True) -> dict:
    if not (use_llm and llm.available()):
        return {"can_submit": True, "blocking": [], "soft": [],
                "raw": {"skipped": "holistic audit skipped (no LLM configured)"}}

    convo_text = _transcript_text(transcript)
    recorded = _all_recorded_values(sections)
    pedigree_summary = ped.teach_back() if ped is not None and hasattr(ped, "teach_back") else None

    system = (
        "You are the FINAL review step in a medical intake system, run once "
        "after (a) every individual section has already been separately "
        "verified for its own fidelity and plausibility, and (b) a set of "
        "deterministic cross-section rules has already run and surfaced its "
        "own findings to the patient (given below as ALREADY-CAUGHT ISSUES). "
        "Your job is narrow: read the whole conversation below as one "
        "continuous patient story and find anything that STILL doesn't add "
        "up, that neither of those earlier layers would catch -- a "
        "narrative contradiction that isn't about a single numeric field, or "
        "a combination of individually-fine values that together describe a "
        "clinically incoherent patient. Do NOT re-report anything already on "
        "the ALREADY-CAUGHT list, and do not flag something confined to a "
        "single section's own consistency (that's checked elsewhere).\n\n"
        "KNOWN, ACCEPTED SYSTEM LIMITATION -- do not flag this as an issue: "
        "population/mutation-frequency model parameters currently only "
        "support UK. EVERY country of residence, including the US or "
        "anywhere else, is deliberately mapped to UK population parameters "
        "by the current implementation -- this is a known scope limitation, "
        "not a data-entry error, and the patient has no way to correct it. "
        "A country of residence other than UK alongside 'UK' population "
        "parameters is EXPECTED and must not be reported as a "
        "cross_section_issue or story_coherence_issue."
    )
    user_parts = [f"FULL INTAKE CONVERSATION, IN ORDER:\n{convo_text}"]
    if pedigree_summary:
        user_parts.append(f"FAMILY HISTORY, AS STRUCTURED:\n{pedigree_summary}")
    user_parts.append(f"RECORDED VALUES BY SECTION:\n{json.dumps(recorded, default=str, indent=2)}")
    user_parts.append("ALREADY-CAUGHT ISSUES (do not re-report these):\n" +
                      ("\n".join(f"- {w}" for w in existing_warnings) if existing_warnings
                       else "(none)"))
    user = "\n\n".join(user_parts)

    result = llm.call_with_tool(system, [{"role": "user", "content": user}],
                                HOLISTIC_AUDIT_TOOL)
    result = result or {"all_clear": True, "cross_section_issues": [],
                        "story_coherence_issues": []}

    cross = result.get("cross_section_issues") or []
    story = result.get("story_coherence_issues") or []
    blocking = [i for i in cross if i.get("severity") == "blocking"]
    soft = [i for i in cross if i.get("severity") != "blocking"] + story

    return {"can_submit": not blocking, "blocking": blocking, "soft": soft,
            "raw": result}


def resolve_holistic_audit(sections: List[Any], transcript: List[dict],
                           ped: Optional[Any] = None,
                           existing_warnings: Optional[List[str]] = None) -> bool:

    verdict = run_holistic_audit(sections, transcript, ped=ped,
                                 existing_warnings=existing_warnings)

    if verdict["blocking"]:
        print("\nAgent: Something in the full picture doesn't add up, and I "
              "can't proceed until it's resolved:")
        for i in verdict["blocking"]:
            print(f"  - {i['description']}")
        print("Agent: Please revisit the relevant section(s) to fix this "
              "before I can run the assessment.")
        return False

    if verdict["soft"]:
        print("\nAgent: One more pass across everything you've told me -- a "
              "couple of things are worth a second look (these don't block "
              "the assessment):")
        for i in verdict["soft"]:
            print(f"  - {i['description']}")
        print("\nAgent: Do you want to correct any of those before I run "
              "this? (yes to continue anyway)")
        resp = ask_user("You: ")
 
        if not is_affirmative(resp, context=(
                "The patient was shown holistic-audit findings and asked "
                "whether to continue anyway. There is currently NO way for "
                "the patient to actually apply a correction at this point "
                "in the conversation, so treat this as affirmative (continue) "
                "UNLESS the reply states a SPECIFIC NEW value that "
                "contradicts something already recorded -- a genuine "
                "correction attempt. Restating, confirming, or explaining "
                "already-recorded information (e.g. 'I am from the USA' when "
                "USA is already on record) is NOT a correction attempt -- "
                "treat that as affirmative too.")):
            print("Agent: Understood -- revisit the relevant section to fix "
                  "those, then we can try again.")
            return False

    return True