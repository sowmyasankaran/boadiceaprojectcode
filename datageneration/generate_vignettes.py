"""
vignette generation

1. SAMPLE  parametric sampler draws structured profiles across a difficulty
              grid (easy/moderate/hard). Deterministic given a seed -> reproducible.
2. ORACLE  feed each vignette's CanRisk file to the API to get the gold
            risk output. The model is the oracle; we never hand-label numbers.
3. RENDER   an LLM turns structured vignette into natural-language
              conversation, with knob that produces the Experiment 4
              conditions (clean, multi_fact, out_of_order, ambiguous, deferred, correction).
"""

from __future__ import annotations
import random
from typing import List, Optional

from vignette_schema import Vignette, FamilyMember, RiskFactors

_FEMALE_FIRST_BIRTH = [22, 25, 28, 30, 33]
_BMIS = [19.5, 22.0, 24.5, 27.0, 31.0]
_MENARCHE = [11, 12, 13, 14]


def _proband(rng: random.Random) -> FamilyMember:
    birth_year = rng.choice(range(1960, 1998))
    age = 2026 - birth_year
    return FamilyMember("prob1", "F", "dad01", "mum01", age=age, yob=birth_year,
                        is_target=True)


def _base_family(rng: random.Random) -> List[FamilyMember]:
    mother = FamilyMember("mum01", "F", "mgf01", "mgm01",
                          age=rng.randint(58, 68), yob=rng.randint(1958, 1968))
    father = FamilyMember("dad01", "M", age=rng.randint(60, 70), yob=rng.randint(1956, 1966))
    mgf = FamilyMember("mgf01", "M", dead=1, age=rng.randint(75, 85), yob=rng.randint(1938, 1948))
    mgm = FamilyMember("mgm01", "F", dead=1, age=rng.randint(75, 85), yob=rng.randint(1938, 1948))
    return [mgf, mgm, mother, father]


def sample_vignette(difficulty: str, rng: random.Random, idx: int) -> Vignette:
    """Draw one structured vignette at the given difficulty tier."""
    fam = _base_family(rng)
    proband = _proband(rng)
    members = fam + [proband]

    rf = RiskFactors(
        menarche=rng.choice(_MENARCHE), parity=rng.choice([0, 1, 2]),
        oc_use="N", mht_use="N", bmi=rng.choice(_BMIS),
        alcohol=rng.choice([0.0, 4.7, 10.0]),
        height=rng.choice([158, 162, 165, 170, 175]), tl="N", endo="N",
    )
    if rf.parity and rf.parity > 0:
        rf.first_live_birth = rng.choice(_FEMALE_FIRST_BIRTH)

    tags = ["exp1", "exp3", "exp5"]

    if difficulty in ("moderate", "hard"):
        aunt = FamilyMember("aunt01", "F", "mgf01", "mgm01",
                            age=rng.randint(65, 75), yob=rng.randint(1950, 1960),
                            bc1=rng.choice([42, 48, 52]))
        members.insert(-1, aunt)

    if difficulty == "hard":
        aunt = next(m for m in members if m.indiv_id == "aunt01")
        aunt.gene_tests = {"BRCA1": "S:P"}
        sister = FamilyMember("sis01", "F", "dad01", "mum01",
                              age=max(30, proband.age - 3), yob=proband.yob + 3)
        members.insert(-1, sister)
        rf.first_live_birth = None
        tags += ["exp2", "exp4"]

    return Vignette(
        vignette_id=f"gen_{difficulty}_{idx:03d}",
        difficulty=difficulty, fam_id=f"FAM{idx:03d}",
        members=members, risk_factors=rf,
        notes=f"Auto-sampled {difficulty} vignette.",
        tags=tags,
    )


def generate_bank(n_per_tier: int = 10, seed: int = 0) -> List[Vignette]:
    """Deterministic, reproducible bank across the three difficulty tiers."""
    rng = random.Random(seed)
    bank = []
    idx = 0
    for tier in ("easy", "moderate", "hard"):
        for _ in range(n_per_tier):
            bank.append(sample_vignette(tier, rng, idx))
            idx += 1
    return bank

def fetch_gold_output(vignette: Vignette) -> dict:
    """Feed the vignette's CanRisk file to the real API -> gold risk result.
    Requires canrisk_test_call.get_auth_token / call_endpoint and CANRISK_* env."""
    import os
    from canrisk_test_call import get_auth_token, call_endpoint
    token = get_auth_token(os.environ["CANRISK_USERNAME"], os.environ["CANRISK_PASSWORD"])
    pedigree = vignette.to_canrisk_file()
    return {
        "breast": call_endpoint("boadicea", token, pedigree),
        "ovarian": call_endpoint("ovarian", token, pedigree),
    }

STYLES = ["clean", "multi_fact", "out_of_order", "ambiguous", "deferred", "correction"]

STYLE_INSTRUCTIONS = {
    "clean": "Answer one field at a time, clearly and unambiguously.",
    "multi_fact": "Volunteer several facts in single messages, as real people do.",
    "out_of_order": "Give information in a different order than asked; jump around.",
    "ambiguous": "Give at least one vague/range answer (e.g. 'mid-40s', 'not sure').",
    "deferred": "Skip or dodge one required question early; only answer it if asked again later.",
    "correction": "State one value, then a few turns later correct it to a different value.",
}


def render_conversation_prompt(vignette: Vignette, style: str) -> dict:
    """Builds the LLM prompt that turns a structured vignette into a simulated
    patient's side of a conversation in the given style. The EXPECTED extracted
    fields are the vignette's own structured truth -- known already -- so this
    yields labeled Exp-4 data. (Run with your LLM key; interface only here.)"""
    truth = {
        "sex": vignette.target().sex,
        "age": vignette.target().age,
        "yob": vignette.target().yob,
        "height_cm": vignette.risk_factors.height,
        "bmi": vignette.risk_factors.bmi,
        "menarche": vignette.risk_factors.menarche,
        "parity": vignette.risk_factors.parity,
        "first_live_birth": vignette.risk_factors.first_live_birth,
        "alcohol_gpd": vignette.risk_factors.alcohol,
        "family": [
            {"id": m.indiv_id, "sex": m.sex, "bc1": m.bc1, "oc": m.oc,
             "genes": m.gene_tests} for m in vignette.members if not m.is_target
        ],
    }
    system = (
        "You are simulating a PATIENT being interviewed for a cancer-risk "
        "assessment. Produce ONLY the patient's side, as short natural messages. "
        "Convey exactly the facts in the profile -- do not add or omit medical "
        f"facts. Style for this run: {STYLE_INSTRUCTIONS[style]}"
    )
    import json
    return {
        "system": system,
        "user": f"PROFILE (ground truth to convey):\n{json.dumps(truth, indent=2)}",
        "expected_fields": truth,   # the label for Exp-4 scoring
        "style": style,
    }
