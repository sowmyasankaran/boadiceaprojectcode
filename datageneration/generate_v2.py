"""
the extended pipelinen:

  seeds (real, cited)  ->  AI-refined realistic variants (demographic diversity)
       +  parametric sampling  ->  graded by two axes:
            - difficulty  (family-structure complexity: easy/moderate/hard)
            - extremity   (how typical the values are: normal/medium/extreme)

Extreme cases stay INSIDE plausibility bounds -> they produce valid risk
    outputs and never trip the agent's own gate. Implausible cases live only in
    the Exp-2 perturbation set (perturbations.py).
Diversity: wide draws over ethnicity/ancestry/age/
    family structure/genes/cancers, with report to see what landed
"""

from __future__ import annotations
import copy
import random
from collections import Counter
from typing import List

from vignette_schema import Vignette, FamilyMember, RiskFactors
from seeds_real import all_real_seeds

# broad diversity pools (not quota-enforced -- just wide coverage)
ETHNICITIES = ["White British", "White Other", "Black African", "Black Caribbean",
               "Indian", "Pakistani", "Chinese", "Mixed", "Ashkenazi Jewish", "Other"]
ANCESTRIES = ["European", "African", "East Asian", "South Asian", "Mixed/Other"]

# extremity value pools -- extreme values sit at the EDGE of the typical
# (soft-bound) range but stay inside it, so outputs are valid and unflagged.
EXTREMITY_POOLS = {
    "normal":  {"menarche": [12, 13], "bmi": [21.5, 23.0, 24.5], "height": [162, 165, 168],
                "onset": [55, 60, 62], "n_affected": [0, 1]},
    "medium":  {"menarche": [11, 14], "bmi": [27.0, 29.5], "height": [158, 172],
                "onset": [48, 52], "n_affected": [1, 2]},
    "extreme": {"menarche": [9, 16], "bmi": [16.0, 33.0], "height": [148, 188],
                "onset": [32, 38], "n_affected": [3, 4]},   # unusual but valid
}


def _diversify(v: Vignette, rng: random.Random) -> Vignette:
    """Apply demographic diversity to a (seed or sampled) vignette."""
    c = copy.deepcopy(v)
    eth = rng.choice(ETHNICITIES)
    c.ethnicity = eth
    c.ancestry = rng.choice(ANCESTRIES)
    c.ashkenazi = (eth == "Ashkenazi Jewish") or (rng.random() < 0.05)
    if c.ashkenazi:
        for m in c.members:
            m.ashkn = 1
    return c


def variants_from_seed(seed: Vignette, n: int, rng: random.Random) -> List[Vignette]:
    """Derive n realistic variants from a real seed by jittering demographics
    and diversity while preserving the seed's affected-structure (the part that
    makes it a real, coherent hereditary pattern)."""
    out = []
    for i in range(n):
        c = _diversify(seed, rng)
        # jitter proband birth year / age slightly (keeps onset relationships)
        shift = rng.choice([-2, 0, 2, 4])
        for m in c.members:
            if m.yob:
                m.yob += shift
                if m.age:
                    m.age = max(m.age - shift, 1)
        c.vignette_id = f"{seed.vignette_id}__var{i:02d}"
        c.source = (seed.source or "") + " | AI-refined variant"
        out.append(c)
    return out


def sample_synthetic(difficulty: str, extremity: str, rng: random.Random, idx: int) -> Vignette:
    """Fully-synthetic vignette graded by difficulty x extremity, diversified,
    with values drawn to keep extreme cases valid (in-bounds)."""
    pool = EXTREMITY_POOLS[extremity]
    by = rng.randint(1962, 1996)
    age = 2026 - by
    proband = FamilyMember("prob1", "F", "dad01", "mum01", age=age, yob=by, is_target=True)
    mother = FamilyMember("mum01", "F", "mgf01", "mgm01", age=rng.randint(58, 68), yob=rng.randint(1958, 1968))
    father = FamilyMember("dad01", "M", age=rng.randint(60, 70), yob=rng.randint(1956, 1966))
    mgf = FamilyMember("mgf01", "M", dead=1, age=rng.randint(75, 85), yob=rng.randint(1938, 1948))
    mgm = FamilyMember("mgm01", "F", dead=1, age=rng.randint(75, 85), yob=rng.randint(1938, 1948))
    members = [mgf, mgm, mother, father, proband]

    rf = RiskFactors(
        menarche=rng.choice(pool["menarche"]), parity=rng.choice([0, 1, 2]),
        oc_use="N", mht_use="N", bmi=rng.choice(pool["bmi"]),
        alcohol=rng.choice([0.0, 4.7, 10.0]), height=rng.choice(pool["height"]),
        tl="N", endo="N",
    )
    if rf.parity:
        rf.first_live_birth = rng.choice([24, 28, 31])

    # affected relatives per extremity, with onset ages from the pool
    n_aff = rng.choice(pool["n_affected"])
    aff_ids = ["aunt01", "sis01", "cous01", "gaunt1"][:n_aff]
    for j, aid in enumerate(aff_ids):
        onset = rng.choice(pool["onset"])
        members.insert(-1, FamilyMember(aid, "F", "mgf01", "mgm01",
                       age=rng.randint(50, 75), yob=rng.randint(1948, 1968),
                       bc1=onset if rng.random() < 0.7 else 0,
                       oc=onset if rng.random() >= 0.7 else 0))
    # in extreme families, make one relative gene-positive
    if extremity == "extreme" and aff_ids:
        next(m for m in members if m.indiv_id == aff_ids[0]).gene_tests = {"BRCA1": "S:P"}

    v = Vignette(
        vignette_id=f"gen_{difficulty}_{extremity}_{idx:03d}",
        difficulty=difficulty, fam_id=f"FAM{idx:03d}", members=members, risk_factors=rf,
        extremity=extremity, source="sampled",
        notes=f"Synthetic {difficulty}/{extremity}.",
        tags=["exp1", "exp3", "exp5"] + (["exp2", "exp4"] if difficulty == "hard" else []),
    )
    return _diversify(v, rng)


def generate_v2_bank(seed: int = 7, variants_per_seed: int = 3,
                     synth_per_cell: int = 2) -> List[Vignette]:
    rng = random.Random(seed)
    bank: List[Vignette] = []

    # 1) real-derived, AI-refined variants (the realistic backbone)
    for s in all_real_seeds():
        bank.extend(variants_from_seed(s, variants_per_seed, rng))

    # 2) synthetic grid: difficulty x extremity (normal/medium/extreme all present)
    idx = 0
    for difficulty in ("easy", "moderate", "hard"):
        for extremity in ("normal", "medium", "extreme"):
            for _ in range(synth_per_cell):
                bank.append(sample_synthetic(difficulty, extremity, rng, idx))
                idx += 1
    return bank


def assert_valid(bank: List[Vignette]) -> dict:
    """Every MAIN-BANK vignette must pass plausibility with NO hard flags -- this
    is what keeps extreme-but-valid cases out of implausible territory and stops
    them contaminating Experiment 1. Returns any soft flags for visibility."""
    from plausibility import check_field_plausibility, check_cross_field
    hard_violations, soft_notes = [], []
    for v in bank:
        t = v.target()
        vals = {"height_cm": v.risk_factors.height, "current_age": t.age,
                "menarche_age": v.risk_factors.menarche,
                "menopause_age": v.risk_factors.menopause,
                "first_live_birth_age": v.risk_factors.first_live_birth}
        for key in ["height_cm", "menarche_age", "menopause_age", "first_live_birth_age"]:
            iss = check_field_plausibility(key, vals[key])
            if iss and iss.severity == "hard":
                hard_violations.append((v.vignette_id, iss.message))
            elif iss:
                soft_notes.append((v.vignette_id, iss.field))
        for iss in check_cross_field(vals):
            if iss.severity == "hard":
                hard_violations.append((v.vignette_id, iss.message))
    assert not hard_violations, f"Main bank has implausible cases: {hard_violations}"
    return {"ok": True, "n": len(bank), "soft_flag_count": len(soft_notes)}


def coverage_report(bank: List[Vignette]) -> dict:
    genes = Counter()
    cancers = Counter()
    for v in bank:
        for m in v.members:
            for g in m.gene_tests:
                genes[g] += 1
            if m.bc1: cancers["breast"] += 1
            if m.oc: cancers["ovarian"] += 1
            if m.pro: cancers["prostate"] += 1
            if m.pan: cancers["pancreatic"] += 1
    return {
        "n": len(bank),
        "by_difficulty": dict(Counter(v.difficulty for v in bank)),
        "by_extremity": dict(Counter(v.extremity for v in bank)),
        "ethnicities": dict(Counter(v.ethnicity for v in bank)),
        "ancestries": dict(Counter(v.ancestry for v in bank)),
        "ashkenazi": dict(Counter(v.ashkenazi for v in bank)),
        "genes_present": dict(genes),
        "cancers_present": dict(cancers),
        "real_seed_derived": sum(1 for v in bank if "seed_real" in (v.tags or [])
                                 or (v.source and "variant" in v.source)),
    }
