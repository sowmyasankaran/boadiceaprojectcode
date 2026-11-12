
import os
import json
from typing import List, Dict, Any, Optional

import requests

import llm
import plausibility
from schema import PersonalLifestyleSection
from clinical_sections import (
    WomensHealthSection, ChildrenSection, BreastScreeningSection,
    MedicalHistorySection, AlcoholSection,
)
from genetic_sections import PRSSection, AdvancedOptionsSection
from pedigree import (
    Pedigree, Individual, Diagnosis, GeneTest, Pathology, seed_structure,
    RELATIONSHIPS, CANCER_TYPES, GENE_PANEL,
)
from canrisk_format import build_pedigree_file, validate_pedigree_file
from report import build_comprehensive_report
from canrisk_test_call import get_auth_token, call_vcf2prs
from genetic_sections import BREAST_PRS_REFERENCE_FILES
from agent import (
    gather_until_complete, verify_until_clean, correct_at_teach_back,
    is_affirmative, apply_extraction, summarize_results, _api_messages,
    extract_turn, compose_question,
)
from audit import resolve_holistic_audit
from agent_io import ask_user




def default_sections(proband_sex: Optional[str] = None) -> List:

    sections = [PersonalLifestyleSection(), AlcoholSection()]
    if proband_sex != "M":
        sections.append(WomensHealthSection())
        sections.append(ChildrenSection())
        sections.append(BreastScreeningSection())
    sections.append(MedicalHistorySection())
    sections.append(PRSSection())
    return sections



FAMILY_TOOL = {
    "name": "record_family_history",
    "description": (
        "Record relatives the patient described, one entry per relative. "
        "Only record relatives the patient ACTUALLY mentioned. Never invent "
        "relatives, ages, or diagnoses. If the patient says a relative had "
        "cancer but doesn't give an age, record the diagnosis with age omitted "
        "rather than guessing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "relatives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "relationship": {
                            "type": "string", "enum": RELATIONSHIPS,
                            "description": "Relationship to the patient.",
                        },
                        "existing_id": {
                            "type": "string",
                            "description": (
                                "If this record adds detail to a relative "
                                "ALREADY recorded (they are listed with their "
                                "ID in the system prompt), put their ID here "
                                "so the details merge onto that same person. "
                                "Omit only when this is genuinely a different, "
                                "new person."
                            ),
                        },
                        "temp_id": {
                            "type": "string",
                            "description": (
                                "Optional short label YOU invent for this "
                                "relative (e.g. 'sib1'), used ONLY so another "
                                "relative described in this SAME response can "
                                "reference them via linked_via_id before they "
                                "have a real ID yet -- e.g. if you're "
                                "recording a brother and his daughter (a "
                                "niece) in the same response, give the "
                                "brother a temp_id and put that same value in "
                                "the niece's linked_via_id."
                            ),
                        },
                        "linked_via_id": {
                            "type": "string",
                            "description": (
                                "REQUIRED whenever relationship is 'niece' or "
                                "'nephew' and more than one sibling has been "
                                "recorded (across this response and prior "
                                "ones), or relationship is 'cousin' and more "
                                "than one aunt/uncle has been recorded: "
                                "either the existing ID (from the system "
                                "prompt, for a sibling/aunt-uncle described "
                                "in an EARLIER turn) or the temp_id (for one "
                                "described in THIS SAME response) of the "
                                "sibling (niece/nephew) or aunt/uncle "
                                "(cousin) who is THIS person's parent. If not "
                                "stated and there's more than one candidate, "
                                "ask the patient whose child this is rather "
                                "than guessing. Not needed if only one "
                                "sibling/aunt-or-uncle exists to be the "
                                "parent."
                            ),
                        },
                        "shared_parent_side": {
                            "type": "string", "enum": ["mother", "father"],
                            "description": (
                                "REQUIRED whenever relationship is "
                                "'half_sister' or 'half_brother' and BOTH the "
                                "patient's mother and father have been "
                                "recorded: which parent this half-sibling "
                                "shares with the patient. Not needed if only "
                                "one of the patient's parents is in the "
                                "pedigree at all -- that's the only possible "
                                "answer."
                            ),
                        },
                        "name": {"type": "string",
                                 "description": "Name/label if given (e.g. 'Aunt Mary')."},
                        "sex": {"type": "string", "enum": ["M", "F"]},
                        "alive": {"type": "boolean"},
                        "age": {"type": "integer",
                                "description": "Current age if alive."},
                        "age_at_death": {"type": "integer"},
                        "year_of_birth": {"type": "integer"},
                        "diagnoses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "cancer_type": {"type": "string", "enum": CANCER_TYPES},
                                    "age": {"type": "integer",
                                            "description": "Age at diagnosis; omit if not stated."},
                                },
                                "required": ["cancer_type"],
                            },
                        },
                        "gene_tests": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "gene": {"type": "string", "enum": GENE_PANEL},
                                    "result": {"type": "string",
                                               "enum": ["P", "N", "U", "HOM", "HET"],
                                               "description": ("P=pathogenic, N=negative, "
                                                   "U=unknown. HOXB13 is the exception: "
                                                   "use HOM/HET/N for it, never P.")},
                                },
                                "required": ["gene", "result"],
                            },
                        },
                        "pathology": {
                            "type": "object",
                            "description": (
                                "Breast tumor receptor/marker status, ONLY if "
                                "the patient states it (e.g. 'ER positive', "
                                "'triple negative', 'HER2 negative'). Each "
                                "field is 'P' (positive), 'N' (negative), or "
                                "omitted if not stated -- never guess."
                            ),
                            "properties": {
                                "er": {"type": "string", "enum": ["P", "N"]},
                                "pr": {"type": "string", "enum": ["P", "N"]},
                                "her2": {"type": "string", "enum": ["P", "N"]},
                                "ck14": {"type": "string", "enum": ["P", "N"]},
                                "ck56": {"type": "string", "enum": ["P", "N"]},
                            },
                        },
                    },
                    "required": ["relationship"],
                },
            },
            "ashkenazi": {
                "type": "boolean",
                "description": "True only if the patient states Ashkenazi Jewish ancestry.",
            },
            "no_more_relatives": {
                "type": "boolean",
                "description": "True if the patient indicated they have nothing more to add.",
            },
        },
    },
}


def extract_family_turn(ped: Pedigree, transcript: List[dict], extra_instructions: str = None,
                        model: str = None, temperature: float = None,
                        provider_override: str = None) -> dict:

    already = [f"[id={i.id}] {i.rel_to_proband}"
               + (f" ({i.name})" if i.name else "")
               + (f", age {i.age}" if i.age is not None else "")
               + (f", cancers: {[d.cancer_type for d in i.diagnoses]}" if i.diagnoses else "")
               + (" -- PLACEHOLDER, not yet described by the patient: if this "
                  "message describes this relative, use existing_id="
                  f"{i.id} to fill in their real details -- do NOT create a "
                  "new relative for them" if i.is_placeholder else "")
               for i in ped.individuals.values() if not i.is_proband]
    latest = next((m["content"] for m in reversed(transcript)
                   if m["role"] == "user"), "")
    system = (
        "You are the family-history extraction step of a cancer-risk intake "
        "agent. Extract relatives the patient describes into structured "
        "records.\n\n"
        f"Relatives ALREADY recorded: {already or 'none yet'}\n"
        "If the patient adds detail about one of these (an age, a diagnosis, "
        "whether they're alive), record it with that relative's existing_id so "
        "it merges onto the SAME person -- do not create a second copy of "
        "them. This applies EVEN IF that entry currently has no details at "
        "all (marked PLACEHOLDER above) -- a placeholder with nothing filled "
        "in is still the correct existing_id to use once the patient actually "
        "describes that relative; it is not a reason to create a new one.\n\n"
        "Extract ONLY from the patient's MOST RECENT message, quoted below. "
        "Use earlier turns solely to interpret it (e.g. resolving 'she' or "
        "'her mother'). If the most recent message adds no new relative or "
        "detail, return an empty relatives list.\n\n"
        f"Most recent patient message: {latest!r}\n\n"
        "Never invent a relative, an age, or a diagnosis. If an age wasn't "
        "stated, omit it rather than estimating. 'her mother' said of the "
        "patient's mother means the patient's maternal grandmother.\n\n"
        "For a niece, nephew, cousin, half-sister, or half-brother: use "
        "linked_via_id (and temp_id if their parent is being introduced in "
        "this same message) to say WHOSE child/half-sibling they are, "
        "whenever that's stated or determinable -- otherwise they can't be "
        "placed in the family tree at all."
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
    return llm.call_with_tool(system, _api_messages(transcript), FAMILY_TOOL, **kwargs) or {}


def apply_family_extraction(ped: Pedigree, extracted: dict) -> List[str]:

    added = []
    if extracted.get("ashkenazi") is not None:
        ped.ashkenazi = extracted["ashkenazi"]

    records = extracted.get("relatives") or []
    temp_id_map: Dict[str, str] = {}
    created: List[tuple] = []  # (record, Individual)

    # pass 1: create/merge everyone, deferring linked_via_id resolution
    for r in records:
        rel = r.get("relationship")
        if not rel:
            continue
        attrs = {k: r[k] for k in ("name", "sex", "alive", "age",
                                    "age_at_death", "year_of_birth",
                                    "shared_parent_side")
                 if r.get(k) is not None}

        ind = None
        eid = r.get("existing_id")
        if eid and eid in ped.individuals:
            ind = ped.individuals[eid]
            for k, v in attrs.items():
                if v is not None:
                    setattr(ind, k, v)
        else:
            ind = ped.upsert_relative(rel, **attrs)

        if r.get("temp_id"):
            temp_id_map[r["temp_id"]] = ind.id

        for d in r.get("diagnoses") or []:
            if d.get("cancer_type") in CANCER_TYPES:
                # don't double-record the same cancer on the same person
                if not any(x.cancer_type == d["cancer_type"] for x in ind.diagnoses):
                    ind.diagnoses.append(Diagnosis(d["cancer_type"], d.get("age")))
                else:
                    for x in ind.diagnoses:
                        if x.cancer_type == d["cancer_type"] and x.age is None:
                            x.age = d.get("age")
        for g in r.get("gene_tests") or []:
            if g.get("gene") in GENE_PANEL and g.get("result") in ("P", "N", "U", "HOM", "HET"):
                if not any(x.gene == g["gene"] for x in ind.gene_tests):
                    ind.gene_tests.append(GeneTest(g["gene"], g["result"]))
        path = r.get("pathology") or {}
        if path:
            if ind.pathology is None:
                ind.pathology = Pathology()
            for marker in ("er", "pr", "her2", "ck14", "ck56"):
                val = path.get(marker)
                if val in ("P", "N"):
                    setattr(ind.pathology, marker, val)

        created.append((r, ind))
        added.append(ind.name or rel.replace("_", " "))

    # pass 2: resolve linked_via_id now that every id in this batch exists
    for r, ind in created:
        raw = r.get("linked_via_id")
        if not raw:
            continue
        resolved = temp_id_map.get(raw) or (raw if raw in ped.individuals else None)
        if resolved:
            ind.linked_via_id = resolved

    return added


def gather_family_history(ped: Pedigree, transcript: List[dict]) -> None:

    print("\nAgent: Now I'd like to hear about cancer in your family. Just tell "
          "me in your own words -- who in your family has had breast, ovarian, "
          "or pancreatic cancer, and roughly what age they were diagnosed if "
          "you know. Include anyone: parents, siblings, grandparents, aunts, "
          "uncles. If nobody has, that's fine -- just say so.")

    while True:
        user_msg = ask_user("You: ")
        transcript.append({"role": "user", "content": user_msg})
        extracted = extract_family_turn(ped, transcript)
        added = apply_family_extraction(ped, extracted)

        if added:
            print(f"Agent: Got it -- noted {', '.join(added)}.")

        done = extracted.get("no_more_relatives") or is_affirmative(
            user_msg, context=("The patient was asked about cancer in their "
                               "family. Is this reply saying they have nothing "
                               "(more) to report?"))

        gaps = ped.missing_detail()
        if done and not gaps:
            break
        if done and gaps:
            for attempt in range(3):
                current_gaps = gaps[:3]
                print("Agent: Before I move on, a couple of quick gaps:")
                for g in current_gaps:
                    print(f"  - {g}")
                reply = ask_user("You: ")
                transcript.append({"role": "user", "content": reply})
                gap_instructions = (
                    "This reply is answering EXACTLY this list of gaps about "
                    "relatives already on record -- make sure each one you "
                    "can answer from the reply gets recorded using that "
                    "relative's existing_id (see the 'Relatives ALREADY "
                    "recorded' list above) so it merges onto them, not a new "
                    "person:\n" + "\n".join(f"- {g}" for g in current_gaps)
                )
                apply_family_extraction(
                    ped, extract_family_turn(ped, transcript,
                                             extra_instructions=gap_instructions))
                gaps = ped.missing_detail()
                if not gaps:
                    break
            if gaps:
                print("Agent: That's alright -- I'll estimate what's needed "
                      "and let you know before running the assessment.")
            break
        if not done:
            print("Agent: Anything else -- any other relatives, or anyone I've "
                  "missed? (or say 'that's everyone')")


    diagnosed = [ind for ind in ped.individuals.values()
                if not ind.is_proband and ind.has_any_cancer()]
    if diagnosed:
        names = ", ".join(ind.name or ind.rel_to_proband.replace("_", " ")
                          for ind in diagnosed)
        print(f"\nAgent: For {names} -- do you know if genetic testing (like "
              f"BRCA1/BRCA2) was ever done, and what it showed? And for any "
              f"breast cancer, do you know anything about the tumor's "
              f"hormone-receptor status (ER/PR/HER2), like whether it was "
              f"called 'triple negative' or 'HER2 positive'? Totally fine to "
              f"skip this if you don't know.")
        reply = ask_user("You: ")
        transcript.append({"role": "user", "content": reply})
        added2 = apply_family_extraction(ped, extract_family_turn(ped, transcript))
        if added2:
            print(f"Agent: Got it, noted for {', '.join(added2)}.")


    if ped.ashkenazi is None:
        print("\nAgent: One more thing -- do you have any Ashkenazi Jewish "
              "ancestry? (this affects the genetics, so it's worth asking)")
        reply = ask_user("You: ")
        transcript.append({"role": "user", "content": reply})
        ped.ashkenazi = is_affirmative(reply, context=(
            "The patient was asked whether they have any Ashkenazi Jewish "
            "ancestry. Answer true only for yes."))

    unplaceable = ped.build_structure()
    if unplaceable:
        print("Agent: (Noting for the record -- these need manual placement in "
              "the pedigree: " + "; ".join(unplaceable) + ")")


def verify_family_history(ped: Pedigree, transcript: List[dict]) -> None:
    while True:
        summary = ped.teach_back()
        print("\n" + summary)
        print("\nAgent: Does that family history look right?")
        resp = ask_user("You: ")
        if is_affirmative(resp, context=(
                f"{summary}\n\nQuestion asked: 'Does that family history look "
                "right?' If the reply adds or changes any relative or "
                "diagnosis, that's a correction, not a confirmation.")):
            break
        transcript.append({"role": "user", "content": resp})
        added = apply_family_extraction(ped, extract_family_turn(ped, transcript))
        if added:
            print(f"Agent: Updated -- {', '.join(added)}.")
        else:
            print("Agent: I didn't catch a specific change there -- what "
                  "should I correct?")
        ped.build_structure()




def run_cross_section_checks(sections: List, ped: Optional[Pedigree]) -> List[str]:

    warnings: List[str] = []

    def find(cls):
        return next((s for s in sections if isinstance(s, cls)), None)

    personal = find(PersonalLifestyleSection)
    children = find(ChildrenSection)
    womens = find(WomensHealthSection)
    medical = find(MedicalHistorySection)

    age = yob = None
    if personal:
        ay = personal.age_yob()
        if ay:
            age, yob = ay


    if children and yob is not None:
        afb = children.age_at_first_birth(yob)
        if afb is not None:
            from plausibility import check_field_plausibility
            issue = check_field_plausibility("first_live_birth_age", afb)
            if issue:
                warnings.append(issue.message)
            if age is not None and afb > age:
                warnings.append(
                    f"Age at first birth ({afb}) is greater than current age "
                    f"({age}), which is impossible -- check the child's birth "
                    f"year or your own.")

    if womens and children and yob is not None:
        menarche = womens.values.get("menarche_age")
        afb = children.age_at_first_birth(yob)
        if menarche is not None and afb is not None and afb <= menarche:
            warnings.append(
                f"Age at first birth ({afb}) isn't after age at first period "
                f"({menarche}), which is biologically unexpected.")

    if medical and age is not None:
        for label, key in [("breast cancer", "breast_cancer_age"),
                            ("ovarian cancer", "ovarian_cancer_age"),
                            ("pancreatic cancer", "pancreatic_cancer_age"),
                            ("contralateral breast cancer",
                             "contralateral_breast_cancer_age")]:
            dx_age = medical.values.get(key)
            if dx_age is not None and dx_age > age:
                warnings.append(
                    f"Age at {label} diagnosis ({dx_age}) is greater than your "
                    f"current age ({age}).")

    if medical:
        if medical.values.get("contralateral_breast_cancer") == "Y" and \
           medical.values.get("breast_cancer") != "Y":
            warnings.append(
                "A second (contralateral) breast cancer was recorded but no "
                "first breast cancer -- one of these is likely wrong.")
        b1 = medical.values.get("breast_cancer_age")
        b2 = medical.values.get("contralateral_breast_cancer_age")
        if b1 is not None and b2 is not None and b2 < b1:
            warnings.append(
                f"The contralateral breast cancer ({b2}) is recorded as earlier "
                f"than the first breast cancer ({b1}).")

    if womens and medical:
        if medical.values.get("oophorectomy") == "Y" and \
           womens.values.get("menopause") == "N":
            warnings.append(
                "Both ovaries removed was recorded, but periods are recorded as "
                "not stopped -- after a bilateral oophorectomy, periods stop. "
                "Worth double-checking.")


    if ped:
        for ind in ped.individuals.values():
            ref_age = ind.age if ind.age is not None else ind.age_at_death
            if ref_age is None:
                continue
            for d in ind.diagnoses:
                if d.age is not None and d.age > ref_age:
                    who = ind.name or ind.rel_to_proband.replace("_", " ")
                    warnings.append(
                        f"{who}: {d.cancer_type.replace('_',' ')} diagnosis age "
                        f"({d.age}) is greater than their age ({ref_age}).")


    if ped:
        diagnosed = [ind for ind in ped.individuals.values()
                    if not ind.is_proband and ind.diagnoses]
        flagged_pairs = set()
        for i, a in enumerate(diagnosed):
            for b in diagnosed[i + 1:]:
                a_ref = a.age if a.age is not None else a.age_at_death
                b_ref = b.age if b.age is not None else b.age_at_death
                if a_ref is None or b_ref is None:
                    continue
                for da in a.diagnoses:
                    for db in b.diagnoses:
                        if da.cancer_type != db.cancer_type:
                            continue
                        if da.age is None or db.age is None:
                            continue
                        if da.age == db.age:
                            continue  # identical ages aren't a "swap" concern
                        if db.age <= a_ref and da.age <= b_ref:
                            pair_key = (a.id, b.id, da.cancer_type)
                            if pair_key in flagged_pairs:
                                continue
                            flagged_pairs.add(pair_key)
                            who_a = a.name or a.rel_to_proband.replace("_", " ")
                            who_b = b.name or b.rel_to_proband.replace("_", " ")
                            warnings.append(
                                f"{who_a}'s {da.cancer_type.replace('_',' ')} "
                                f"diagnosis age ({da.age}) and {who_b}'s "
                                f"({db.age}) would both still be plausible if "
                                f"swapped between them -- worth double-checking "
                                f"these are recorded against the right person.")

    return warnings


def run_full_intake():
    if not llm.available():
        print("No LLM provider. Set OPENAI_API_KEY (or ANTHROPIC_API_KEY), "
              "optionally LLM_PROVIDER=openai|anthropic.")
        return
    print(f"[using LLM provider: {llm.provider()}]")
    if plausibility.pycountry is None:
        print("[WARNING: pycountry is not installed -- country plausibility "
              "checking is DISABLED for this run. Fake countries will NOT be "
              "flagged. Run: pip install pycountry --break-system-packages]")

    transcript: List[dict] = []
    print("Agent: Hi -- I'll ask about your health, your family, and a few "
          "lifestyle details to set up a breast and ovarian cancer risk "
          "assessment. We'll go a section at a time, and I'll read each part "
          "back to you before moving on.")


    personal = PersonalLifestyleSection()
    print("\n--- Personal details & lifestyle ---")
    _run_section(personal, transcript)

    sections = default_sections(personal.values.get("sex"))
    sections[0] = personal  # already done

    for section in sections[1:]:
        label = section.name.replace("_", " ").title()
        print(f"\n--- {label} ---")
        _run_section(section, transcript)

    print("\n--- Family History ---")
    ped = Pedigree()
    ped.proband.sex = personal.values.get("sex")
    ay = personal.age_yob()
    if ay:
        ped.proband.age, ped.proband.year_of_birth = ay
    gather_family_history(ped, transcript)
    verify_family_history(ped, transcript)

    advanced = AdvancedOptionsSection.with_country_default(
        personal.values.get("country"))
    sections.append(advanced)

    warnings = run_cross_section_checks(sections, ped)
    if warnings:
        print("\nAgent: A few things don't quite line up across what you've "
              "told me. None of these block the assessment, but they're worth "
              "a look:")
        for w in warnings:
            print(f"  - {w}")
        print("\nAgent: Do you want to correct any of those before I run this? "
              "(yes to continue anyway)")

        if not is_affirmative(ask_user("You: "), context=(
                "The patient was shown cross-section inconsistencies and "
                "asked whether to continue anyway. There is currently NO way "
                "for the patient to actually apply a correction at this "
                "point in the conversation, so treat this as affirmative "
                "(continue) UNLESS the reply states a SPECIFIC NEW value "
                "that contradicts something already recorded -- a genuine "
                "correction attempt. Restating, confirming, or explaining "
                "already-recorded information is NOT a correction attempt -- "
                "treat that as affirmative too.")):
            print("Agent: Understood -- re-run the relevant section to fix "
                  "those. (Targeted cross-section correction is the next thing "
                  "to build; flagging rather than silently proceeding.)")
            return

    # --- Final combined teach-back ------------------------------------------
    print("\n=== Everything I have ===")
    for section in sections:
        if hasattr(section, "teach_back"):
            print("\n" + section.teach_back())
    print("\n" + ped.teach_back())
    print("\nAgent: Is all of that correct? (yes to run the assessment)")
    if not is_affirmative(ask_user("You: ")):
        print("Agent: No problem -- tell me which section to revisit.")
        return

    if not resolve_holistic_audit(sections, transcript, ped=ped,
                                  existing_warnings=warnings):
        return


    final_state_path = os.environ.get("EXP3_FINAL_STATE_PATH")
    if final_state_path:
        try:
            dump = {
                "sections": {s.name: dict(s.values) for s in sections
                            if hasattr(s, "values")},
                "pedigree": ped.to_pedigree_rows(),
            }
            with open(final_state_path, "w") as f:
                json.dump(dump, f, indent=2, default=str)
        except Exception as e:

            print(f"\n[Exp3 instrumentation: couldn't write final-state dump: {e}]")

    print("\nAgent: Running your assessment...")
    results = submit_intake(personal, sections, ped)

    for model in ("breast", "ovarian"):
        r = results.get(model)
        if isinstance(r, dict) and "error" in r:
            print(f"\n[{model} model did not return a result: {r['error']}]")
            if r.get("detail"):
                print(f"[detail: {r['detail']}]")

    print("\n" + summarize_results(results))

    with open("last_pedigree.canrisk", "w") as f:
        f.write(results.get("pedigree", ""))
    print("\n[the exact CanRisk v4 file submitted was saved to "
          "last_pedigree.canrisk -- load it into canrisk.org to cross-check "
          "this result against the official tool]")

    if "breast" in results or "ovarian" in results:
        print("\n" + "=" * 60)
        print(build_comprehensive_report(results))
        print("=" * 60)
        print("[full raw API responses saved to last_boadicea_response.json "
              "and last_ovarian_response.json -- if any section above says "
              "'not available in this response', that's the file to check "
              "the real key names against]")


def _find(sections, cls):
    return next((s for s in sections if isinstance(s, cls)), None)


def submit_intake(personal, sections: List, ped: Pedigree) -> dict:

    username = os.environ.get("CANRISK_USERNAME")
    password = os.environ.get("CANRISK_PASSWORD")
    if not username or not password:
        return {"error": "Set CANRISK_USERNAME / CANRISK_PASSWORD to submit."}

    try:
        token = get_auth_token(username, password)
    except Exception as e:
        return {"error": f"CanRisk authentication failed: {e}"}

    prs = _find(sections, PRSSection)
    if prs is not None:
        vcf_error = _resolve_vcf_prs(prs, token)
        if vcf_error:
            print(f"\nAgent: (Note -- couldn't compute your PRS from the VCF "
                  f"file: {vcf_error}. Continuing without it.)")


    ay = personal.age_yob()
    estimated = ped.estimate_missing_years(ay[1] if ay else None)
    if estimated:
        print("\nAgent: I don't have exact birth years for everyone, so I've "
              "estimated based on typical family age gaps -- you can correct "
              "these later if you know better: " + "; ".join(estimated))

    dropped: List[str] = []
    pedigree_text = build_pedigree_file(
        ped,
        personal=personal,
        womens=_find(sections, WomensHealthSection),
        children=_find(sections, ChildrenSection),
        screening=_find(sections, BreastScreeningSection),
        medical=_find(sections, MedicalHistorySection),
        prs=prs,
        alcohol=_find(sections, AlcoholSection),
        dropped_out=dropped,
    )
    if dropped:
        print("\nAgent: (Note -- these relatives couldn't be connected to your "
              "family tree, so the model can't use them: "
              + "; ".join(dropped) + ")")


    problems = validate_pedigree_file(pedigree_text)
    if problems:
        return {"error": "Pedigree file failed local validation:\n  - "
                          + "\n  - ".join(problems),
                "pedigree": pedigree_text}

    advanced = _find(sections, AdvancedOptionsSection)
    params = (advanced.to_canrisk()["request_params"] if advanced
              else {"mut_freq": "UK", "cancer_rates": "UK"})

    out = {"pedigree": pedigree_text}
    for endpoint in ("boadicea", "ovarian"):
        try:
            out[endpoint.replace("boadicea", "breast")] = _call(
                endpoint, token, pedigree_text, params)
        except Exception as e:
            out[endpoint.replace("boadicea", "breast")] = {"error": str(e)}
    return out


def _resolve_vcf_prs(prs: PRSSection, token: str) -> Optional[str]:

    v = prs.values
    if v.get("prs_run") != "Y" or v.get("input_mode") != "vcf":
        return None
    ref_file = BREAST_PRS_REFERENCE_FILES.get(v.get("breast_panel"))
    if not ref_file:
        return "no breast reference panel selected"
    result = call_vcf2prs(token, ref_file, v.get("vcf_sample_name"),
                          v.get("vcf_file_path"))
    if "error" in result:
        return f"{result['error']} -- {result.get('detail')}"
    alpha, zscore = result.get("alpha"), result.get("zscore")
    if alpha is None or zscore is None:
        return f"unexpected response from /vcf2prs/: {result}"
    prs.values["breast_prs_alpha"] = alpha
    prs.values["breast_prs_zscore"] = zscore
    return None


def _call(endpoint: str, token: str, pedigree_data: str, params: dict) -> dict:

    resp = requests.post(
        "https://canrisk.org/" + endpoint + "/",
        json={"mut_freq": params.get("mut_freq", "UK"),
              "cancer_rates": params.get("cancer_rates", "UK"),
              "user_id": "intake_agent",
              "pedigree_data": pedigree_data},
        headers={"Authorization": f"Token {token}"},
        timeout=60,
    )
    if resp.status_code != 200:
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        return {"error": f"HTTP {resp.status_code}", "detail": body}
    return resp.json()


def ask_opening_question(section, transcript: List[dict]) -> None:

    question = compose_question(section, transcript)
    print(f"Agent: {question}")
    transcript.append({"role": "assistant", "content": question})


def _run_section(section, transcript: List[dict]) -> None:

    ask_opening_question(section, transcript)
    gather_until_complete(section, transcript)
    confirmed_plausibility, resolved_fidelity = set(), set()
    while True:
        confirmed_plausibility, resolved_fidelity = verify_until_clean(
            section, transcript, confirmed_plausibility, resolved_fidelity)
        summary = section.teach_back() if hasattr(section, "teach_back") else None
        if summary is None:
            break
        print("\n" + summary)
        print("\nAgent: Does that look right? (yes to continue)")
        resp = ask_user("You: ")
        if is_affirmative(resp, context=(
                f"{summary}\n\nQuestion asked: 'Does that look right?' If the "
                "reply mentions anything not already reflected above, that is "
                "a correction, not a confirmation.")):
            break
        touched = correct_at_teach_back(section, transcript)
        for f in touched:
            confirmed_plausibility.discard(f)
            resolved_fidelity.discard(f)
        if not section.is_complete():
            gather_until_complete(section, transcript)


if __name__ == "__main__":
    run_full_intake()