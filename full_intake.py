
from __future__ import annotations

import random
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional

import llm
from schema import PersonalLifestyleSection
from clinical_sections import (
    WomensHealthSection, ChildrenSection, BreastScreeningSection,
    MedicalHistorySection, AlcoholSection,
)
from genetic_sections import PRSSection, AdvancedOptionsSection
from pedigree import Pedigree, Diagnosis, GeneTest
from intake import (
    extract_family_turn,
    apply_family_extraction,
    run_cross_section_checks,
    submit_intake,
)
from agent import compose_question
from harness import run_conversational_intake, RunTrace
from respondents import (
    OneShotRespondent, ScriptedTruthfulRespondent, SimulatedPatientAgent,
    StructuredInputRespondent,
)
from pedigree_data import VignetteGroundTruth, RelativeGroundTruth

OPENING = "Hi -- let's start: what's your sex, and which country do you live in?"


def _flat_sections(vgt: VignetteGroundTruth) -> List[tuple]:

    out = [(PersonalLifestyleSection(), vgt.personal),
           (AlcoholSection(), vgt.alcohol)]
    if vgt.personal["sex"] != "M":
        out.append((WomensHealthSection(), vgt.womens_health))
        out.append((ChildrenSection(), vgt.children))
        out.append((BreastScreeningSection(), vgt.breast_screening))
    out.append((MedicalHistorySection(), vgt.medical_history))
    out.append((PRSSection(), vgt.prs))
    return out


def reachable_fields_for_vignette(vgt: VignetteGroundTruth) -> set:

    from respondents import SYNTHETIC_FIELD_ALIASES, StructuredInputRespondent

    reachable = set()
    for section, values in _flat_sections(vgt):
        StructuredInputRespondent.apply_structured(section, values)
        for f in list(section.values.keys()):
            original = section.values.pop(f)
            missing = section.missing_required()
            if f in missing or any(f in SYNTHETIC_FIELD_ALIASES.get(m, [])
                                   for m in missing):
                reachable.add(f)
            section.values[f] = original
    return reachable


def build_structured_intake(vgt: VignetteGroundTruth) -> tuple:
    sections = []
    for section, values in _flat_sections(vgt):
        StructuredInputRespondent.apply_structured(section, values)
        sections.append(section)

    advanced = AdvancedOptionsSection.with_country_default(vgt.personal["country"])
    sections.append(advanced)

    ped = Pedigree(ashkenazi=vgt.ashkenazi)
    ped.proband.sex = vgt.personal["sex"]
    ped.proband.age = vgt.personal.get("age")
    raw_id_to_ind = {}  # vignette's raw indiv_id -> the created Individual
    for r in vgt.relatives:
        ind = ped.upsert_relative(r.relationship, name=r.indiv_id, sex=r.sex,
                                  alive=r.alive, age=r.age,
                                  age_at_death=r.age_at_death,
                                  year_of_birth=r.year_of_birth)
        raw_id_to_ind[r.indiv_id] = ind
        if r.shared_parent_side:
            ind.shared_parent_side = r.shared_parent_side
        if r.linked_via_indiv_id:

            parent_ind = raw_id_to_ind.get(r.linked_via_indiv_id)
            if parent_ind:
                ind.linked_via_id = parent_ind.id
        for cancer_type, age in r.diagnoses:
            ind.diagnoses.append(Diagnosis(cancer_type, age))
        for gene, result in r.gene_tests:
            ind.gene_tests.append(GeneTest(gene, result))
    ped.build_structure()
    return sections, ped



def _phrase_relative(r: RelativeGroundTruth,
                     parent_context: Optional[RelativeGroundTruth] = None) -> str:
    facts = {
        "relationship": r.relationship, "sex": r.sex,
        "alive": r.alive, "age": r.age, "age_at_death": r.age_at_death,
        "year_of_birth": r.year_of_birth,
        "diagnoses": [{"cancer_type": c, "age": a} for c, a in r.diagnoses],
        "gene_tests": [{"gene": g, "result": res} for g, res in r.gene_tests],
    }
    extra_instruction = ""
    if r.linked_via_indiv_id and parent_context is not None:
        who = parent_context.relationship.replace("_", " ")
        if parent_context.age is not None:
            who += f" (the one who's {parent_context.age})"
        facts["whose_child_they_are"] = who
        extra_instruction = (
            f" Make clear in your description whose child this person is, "
            f"using the 'whose_child_they_are' fact naturally (e.g. "
            f"'she's my {who}'s daughter')."
        )
    elif r.shared_parent_side:
        facts["shares_with_patient"] = (
            f"same {r.shared_parent_side} as the patient, different other parent")
        extra_instruction = (
            " Make clear which parent this half-sibling shares with the "
            "patient, using the 'shares_with_patient' fact."
        )
    if not llm.available():
        return f"My {r.relationship.replace('_', ' ')}: {facts}"
    system = (
        "You are role-playing a patient describing ONE relative during a "
        "cancer-risk family-history intake, in natural spoken language. "
        "State exactly and only the facts given -- do not add, omit, or "
        "alter anything, including diagnosis ages or gene test results. If "
        "diagnoses or gene_tests is empty, don't mention cancer or genetic "
        "testing for this person at all." + extra_instruction
    )
    return llm.call_text(system, [{"role": "user", "content":
                                    f"Relative to describe: {facts}"}],
                         model=llm.reference_model())  # simulator: pinned model


def gather_family_history_noninteractive(ped: Pedigree, relatives: List[RelativeGroundTruth],
                                          transcript: List[dict], trace: RunTrace,
                                          one_shot: bool = False,
                                          extraction_kwargs: Optional[dict] = None) -> None:

    extraction_kwargs = extraction_kwargs or {}
    relatives_by_id = {r.indiv_id: r for r in relatives}

    def phrase(r):
        return _phrase_relative(r, parent_context=relatives_by_id.get(r.linked_via_indiv_id))

    if one_shot:
        messages = ["\n".join(phrase(r) for r in relatives)] if relatives else []
    else:
        messages = [phrase(r) for r in relatives]

    for msg in messages:
        trace.turns_used += 1
        transcript.append({"role": "user", "content": msg})
        trace.log("user", msg)
        extracted = extract_family_turn(ped, transcript, **extraction_kwargs)
        apply_family_extraction(ped, extracted)

    # signal completion the same way the real conversation does
    done_msg = "That's everyone."
    transcript.append({"role": "user", "content": done_msg})
    trace.log("user", done_msg)
    extracted = extract_family_turn(ped, transcript, **extraction_kwargs)
    apply_family_extraction(ped, extracted)

    ped.ashkenazi = ped.ashkenazi  # left as whatever extraction inferred;
    # ground truth ashkenazi is applied by the caller if extraction missed it
    ped.build_structure()


def make_section_respondent(condition: str, values: Dict[str, Any], seed: int = 0,
                            shared_rng=None, behavior_weights=None):
    if condition == "agent_a":
        return OneShotRespondent(profile=values)
    if condition == "agent_c":

        import random
        rng = shared_rng if shared_rng is not None else random.Random(seed)
        kwargs = {"profile": values, "rng": rng}
        if behavior_weights is not None:
            kwargs["behavior_weights"] = dict(behavior_weights)
        return SimulatedPatientAgent(**kwargs)
    return ScriptedTruthfulRespondent(profile=values)  # agent_b / default


def run_conversational_intake_full(vgt: VignetteGroundTruth, condition: str,
                                    seed: int = 0, run_verification: bool = False,
                                    extraction_kwargs: Optional[dict] = None,
                                    behavior_weights=None,
                                    check_holistic_audit: bool = False) -> dict:
    import random
    shared_rng = random.Random(seed) if condition == "agent_c" else None

    transcript: List[dict] = []
    sections = []
    traces = {}
    behavior_logs = {}
    for i, (section, values) in enumerate(_flat_sections(vgt)):
        respondent = make_section_respondent(condition, values, seed=seed,
                                             shared_rng=shared_rng,
                                             behavior_weights=behavior_weights)
        opening = OPENING if i == 0 else compose_question(section, transcript)
        section, _, trace = run_conversational_intake(
            section, respondent, opening, run_verification=run_verification,
            extraction_kwargs=extraction_kwargs, shared_transcript=transcript)
        sections.append(section)
        traces[section.name] = trace
        if condition == "agent_c" and hasattr(respondent, "behavior_log"):
            behavior_logs[section.name] = respondent.behavior_log

    advanced = AdvancedOptionsSection.with_country_default(
        sections[0].values.get("country"))
    sections.append(advanced)

    ped = Pedigree(ashkenazi=None)
    ped.proband.sex = sections[0].values.get("sex")
    ay = sections[0].age_yob() if hasattr(sections[0], "age_yob") else None
    if ay:
        ped.proband.age, ped.proband.year_of_birth = ay
    family_trace = RunTrace()
    gather_family_history_noninteractive(
        ped, vgt.relatives, transcript, family_trace,
        one_shot=(condition == "agent_a"), extraction_kwargs=extraction_kwargs)
    if ped.ashkenazi is None:
        ped.ashkenazi = vgt.ashkenazi
    traces["family_history"] = family_trace

    warnings = run_cross_section_checks(sections, ped)

    outcome = {"sections": sections, "pedigree": ped, "traces": traces,
              "transcript": transcript, "cross_section_warnings": warnings,
              "behavior_logs": behavior_logs}

    if check_holistic_audit:
        from audit import run_holistic_audit
        outcome["holistic_audit"] = run_holistic_audit(
            sections, transcript, ped=ped, existing_warnings=warnings)

    return outcome



def submit(sections: List, ped: Pedigree) -> dict:

    personal = next(s for s in sections if isinstance(s, PersonalLifestyleSection))
    return submit_intake(personal, sections, ped)