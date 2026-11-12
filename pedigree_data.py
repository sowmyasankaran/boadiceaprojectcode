from __future__ import annotations

import json
import os
from dataclasses import dataclass, field as dc_field
from datetime import date
from typing import Any, Dict, List, Optional

from pedigree import RELATIONSHIPS, CANCER_TYPES

DATA_DIR = os.path.dirname(__file__)

UNSCORED_FIELDS = {
    "oc_years", "oc_last_two_years", "hrt_years", "hrt_last_five_years",
    "hrt_type", "oophorectomy", "mastectomy", "children",
    "had_mammogram", "told_density",
}


def _load_json(filename: str, override_env: str):
    path = os.environ.get(override_env, os.path.join(DATA_DIR, filename))
    with open(path) as f:
        return json.load(f)


def load_vignettes(tag: Optional[str] = None) -> List[dict]:
    vignettes = _load_json("vignette_bank_v2.json", "VIGNETTE_BANK_PATH")
    if tag:
        vignettes = [v for v in vignettes if tag in v.get("tags", [])]
    return vignettes


def load_exp2_cases() -> List[dict]:
    return _load_json("exp2_cases.json", "EXP2_CASES_PATH")


def load_gold_outputs() -> Dict[str, dict]:
    return _load_json("gold_outputs.json", "GOLD_OUTPUTS_PATH")


def find_vignette(vignette_id: str, vignettes: Optional[List[dict]] = None) -> dict:
    vignettes = vignettes if vignettes is not None else load_vignettes()
    return next(v for v in vignettes if v["vignette_id"] == vignette_id)



def infer_relationships(vignette: dict) -> Dict[str, str]:
    members = {m["indiv_id"]: m for m in vignette["members"]}
    target = next(m for m in vignette["members"] if m.get("is_target"))
    pid = target["indiv_id"]
    p_mother, p_father = target.get("mother_id", "0"), target.get("father_id", "0")

    def sexed(label_f: str, label_m: str, member: dict) -> str:
        return label_f if member.get("sex") == "F" else label_m

    rel: Dict[str, str] = {pid: "self"}

    # parents
    if p_mother in members:
        rel[p_mother] = "mother"
    if p_father in members:
        rel[p_father] = "father"

    # full / half siblings, and children of the proband
    for mid, m in members.items():
        if mid == pid or mid in rel:
            continue
        m_mother, m_father = m.get("mother_id", "0"), m.get("father_id", "0")
        if m_father == pid or m_mother == pid:
            rel[mid] = sexed("daughter", "son", m)
        elif p_mother != "0" and p_father != "0" and m_mother == p_mother and m_father == p_father:
            rel[mid] = sexed("sister", "brother", m)
        elif (p_mother != "0" and m_mother == p_mother) or (p_father != "0" and m_father == p_father):
            rel[mid] = sexed("half_sister", "half_brother", m)

    # grandparents (via the already-identified mother/father)
    mgm = mgf = pgm = pgf = None
    if p_mother in members:
        mm = members[p_mother]
        mgf, mgm = mm.get("father_id", "0"), mm.get("mother_id", "0")
        if mgf in members and mgf not in rel:
            rel[mgf] = "maternal_grandfather"
        if mgm in members and mgm not in rel:
            rel[mgm] = "maternal_grandmother"
    if p_father in members:
        fm = members[p_father]
        pgf, pgm = fm.get("father_id", "0"), fm.get("mother_id", "0")
        if pgf in members and pgf not in rel:
            rel[pgf] = "paternal_grandfather"
        if pgm in members and pgm not in rel:
            rel[pgm] = "paternal_grandmother"

    # aunts/uncles: siblings of a parent (share that parent's own parents)
    for mid, m in members.items():
        if mid in rel:
            continue
        m_mother, m_father = m.get("mother_id", "0"), m.get("father_id", "0")
        if mgf and mgm and m_father == mgf and m_mother == mgm and mid != p_mother:
            rel[mid] = sexed("maternal_aunt", "maternal_uncle", m)
        elif pgf and pgm and m_father == pgf and m_mother == pgm and mid != p_father:
            rel[mid] = sexed("paternal_aunt", "paternal_uncle", m)

    # nieces/nephews: children of a sibling; cousins: children of an aunt/uncle
    sibling_ids = {mid for mid, r in rel.items()
                   if r in ("sister", "brother", "half_sister", "half_brother")}
    auntuncle_ids = {mid for mid, r in rel.items()
                     if r in ("maternal_aunt", "maternal_uncle",
                              "paternal_aunt", "paternal_uncle")}
    for mid, m in members.items():
        if mid in rel:
            continue
        m_mother, m_father = m.get("mother_id", "0"), m.get("father_id", "0")
        if m_mother in sibling_ids or m_father in sibling_ids:
            rel[mid] = sexed("niece", "nephew", m)
        elif m_mother in auntuncle_ids or m_father in auntuncle_ids:
            rel[mid] = "cousin"

    # anything left over (dataset doesn't currently have this case, but stay
    # honest about it rather than crashing if a future vignette does)
    for mid in members:
        rel.setdefault(mid, "other")

    return rel


def infer_parent_linkage(vignette: dict) -> Dict[str, dict]:

    members = {m["indiv_id"]: m for m in vignette["members"]}
    target = next(m for m in vignette["members"] if m.get("is_target"))
    p_mother = target.get("mother_id", "0")
    p_father = target.get("father_id", "0")
    rel = infer_relationships(vignette)

    sibling_ids = {mid for mid, r in rel.items()
                   if r in ("sister", "brother", "half_sister", "half_brother")}
    auntuncle_ids = {mid for mid, r in rel.items()
                     if r in ("maternal_aunt", "maternal_uncle",
                              "paternal_aunt", "paternal_uncle")}

    linkage: Dict[str, dict] = {}
    for mid, m in members.items():
        r = rel.get(mid)
        m_mother, m_father = m.get("mother_id", "0"), m.get("father_id", "0")
        if r in ("half_sister", "half_brother"):
            if p_mother != "0" and m_mother == p_mother:
                linkage[mid] = {"shared_parent_side": "mother"}
            elif p_father != "0" and m_father == p_father:
                linkage[mid] = {"shared_parent_side": "father"}
        elif r in ("niece", "nephew"):
            parent = (m_mother if m_mother in sibling_ids
                     else (m_father if m_father in sibling_ids else None))
            if parent:
                linkage[mid] = {"linked_via_indiv_id": parent}
        elif r == "cousin":
            parent = (m_mother if m_mother in auntuncle_ids
                     else (m_father if m_father in auntuncle_ids else None))
            if parent:
                linkage[mid] = {"linked_via_indiv_id": parent}
    return linkage


_DX_FIELD_MAP = {"bc1": "breast", "bc2": "contralateral_breast",
                  "oc": "ovarian", "pan": "pancreatic"}


def member_diagnoses(member: dict) -> List[tuple]:
    """[(cancer_type, age_at_diagnosis), ...] for a member's nonzero dx fields."""
    out = []
    for field, cancer_type in _DX_FIELD_MAP.items():
        age = member.get(field, 0)
        if age and age > 0:
            out.append((cancer_type, age))
    return out


def member_gene_tests(member: dict) -> List[tuple]:
    """[(gene, result), ...], parsed from the CanRisk column format
    ('S:P' -> ('GENE','P')). Only keeps results in the schema's accepted set."""
    out = []
    for gene, code in (member.get("gene_tests") or {}).items():
        result = code.split(":")[-1] if ":" in code else code
        if result in ("P", "N", "U", "HOM", "HET"):
            out.append((gene, result))
    return out


def _menopause_age_band(age: Optional[int]) -> Optional[str]:
    if age is None:
        return None
    if age < 40:
        return "<40"
    if age < 45:
        return "40-44"
    if age < 50:
        return "45-49"
    if age < 55:
        return "50-54"
    return "54+"


def personal_lifestyle_ground_truth(vignette: dict) -> Dict[str, Any]:
    target = next(m for m in vignette["members"] if m.get("is_target"))
    rf = vignette["risk_factors"]
    height = rf.get("height")
    bmi = rf.get("bmi")

    weight_kg = round(bmi * ((height / 100.0) ** 2), 1) if (height and bmi) else None
    return {
        "sex": target.get("sex", "F"),

        "country": "UK",
        "ethnic_group": vignette.get("ethnicity"),
        "age": target.get("age") if target.get("age") else None,
        "height_cm": height,
        "weight_kg": weight_kg,
    }


def alcohol_ground_truth(vignette: dict) -> Dict[str, Any]:

    from conversions import drink_grams_per_unit
    grams_per_day = vignette["risk_factors"].get("alcohol") or 0.0
    if grams_per_day <= 0:
        return {"drinks_alcohol": "N", "drinks": []}
    grams_per_week = grams_per_day * 7.0
    amount = round(grams_per_week / drink_grams_per_unit("wine"), 2)
    return {
        "drinks_alcohol": "Y",
        "drinks": [{"type": "wine", "frequency": "weekly", "amount": amount}],
    }


def womens_health_ground_truth(vignette: dict) -> Dict[str, Any]:
    rf = vignette["risk_factors"]
    menopause_age = rf.get("menopause")
    values = {
        "menarche_age": rf.get("menarche"),
        "oc_use": rf.get("oc_use", "U"),
        "menopause": "Y" if menopause_age is not None else "N",
        "hrt_use": rf.get("mht_use", "U"),
    }
    if menopause_age is not None:
        values["menopause_age_band"] = _menopause_age_band(menopause_age)

    if values["oc_use"] == "Y":
        values["oc_years"] = "5-9"
        values["oc_last_two_years"] = "N"
    if values["hrt_use"] == "Y":
        values["hrt_years"] = 1
        values["hrt_last_five_years"] = "N"
        values["hrt_type"] = "unknown"
    return values


def children_ground_truth(vignette: dict, proband_yob: Optional[int]) -> Dict[str, Any]:
    rf = vignette["risk_factors"]
    parity = rf.get("parity") or 0
    if not parity:
        return {"has_children": "N"}
    first_birth_age = rf.get("first_live_birth")
    values: Dict[str, Any] = {"has_children": "Y", "num_children": parity}

    if first_birth_age is not None and proband_yob is not None:
        first_yob = proband_yob + first_birth_age
        values["children"] = [{"year_of_birth": first_yob, "sex": "F"}
                              for _ in range(parity)]
    return values


def medical_history_ground_truth(vignette: dict) -> Dict[str, Any]:
    target = next(m for m in vignette["members"] if m.get("is_target"))
    rf = vignette["risk_factors"]
    dx = dict(member_diagnoses(target))
    values = {
        "endometriosis": rf.get("endo", "U"),
        "tubal_ligation": rf.get("tl", "U"),

        "oophorectomy": "N",
        "mastectomy": "N",
        "breast_cancer": "Y" if "breast" in dx else "N",
        "contralateral_breast_cancer": "Y" if "contralateral_breast" in dx else "N",
        "ovarian_cancer": "Y" if "ovarian" in dx else "N",
        "pancreatic_cancer": "Y" if "pancreatic" in dx else "N",
    }
    if "breast" in dx:
        values["breast_cancer_age"] = dx["breast"]
    if "contralateral_breast" in dx:
        values["contralateral_breast_cancer_age"] = dx["contralateral_breast"]
    if "ovarian" in dx:
        values["ovarian_cancer_age"] = dx["ovarian"]
    if "pancreatic" in dx:
        values["pancreatic_cancer_age"] = dx["pancreatic"]
    return values


def breast_screening_ground_truth(vignette: dict) -> Dict[str, Any]:
    return {"had_mammogram": "U"}


def prs_ground_truth(vignette: dict) -> Dict[str, Any]:
    # No PRS data in this dataset.
    return {"prs_run": "N"}


@dataclass
class RelativeGroundTruth:
    indiv_id: str
    relationship: str
    sex: Optional[str]
    alive: bool
    age: Optional[int]
    age_at_death: Optional[int]
    year_of_birth: Optional[int]
    diagnoses: List[tuple]     # (cancer_type, age)
    gene_tests: List[tuple]    # (gene, result)
    linked_via_indiv_id: Optional[str] = None  # niece/nephew/cousin: raw
                                                # indiv_id of their parent
                                                # (sibling/aunt-or-uncle)
    shared_parent_side: Optional[str] = None   # half_sister/half_brother:
                                                # "mother" or "father"


def family_ground_truth(vignette: dict) -> List[RelativeGroundTruth]:
    rel_map = infer_relationships(vignette)
    linkage = infer_parent_linkage(vignette)
    out = []
    for m in vignette["members"]:
        if m.get("is_target"):
            continue
        mid = m["indiv_id"]
        alive = m.get("dead", 0) == 0
        link = linkage.get(mid, {})
        out.append(RelativeGroundTruth(
            indiv_id=mid, relationship=rel_map.get(mid, "other"),
            sex=m.get("sex"), alive=alive,
            age=m.get("age") if alive else None,
            age_at_death=m.get("age") if not alive else None,
            year_of_birth=m.get("yob") if m.get("yob") else None,
            diagnoses=member_diagnoses(m), gene_tests=member_gene_tests(m),
            linked_via_indiv_id=link.get("linked_via_indiv_id"),
            shared_parent_side=link.get("shared_parent_side"),
        ))

    out.sort(key=lambda r: r.linked_via_indiv_id is not None)
    return out


@dataclass
class VignetteGroundTruth:
    vignette_id: str
    personal: Dict[str, Any]
    alcohol: Dict[str, Any]
    womens_health: Dict[str, Any]
    children: Dict[str, Any]
    medical_history: Dict[str, Any]
    breast_screening: Dict[str, Any]
    prs: Dict[str, Any]
    relatives: List[RelativeGroundTruth]
    ashkenazi: bool


def full_ground_truth(vignette: dict) -> VignetteGroundTruth:
    personal = personal_lifestyle_ground_truth(vignette)
    proband_yob = (date.today().year - personal["age"]) if personal.get("age") else None
    return VignetteGroundTruth(
        vignette_id=vignette["vignette_id"],
        personal=personal,
        alcohol=alcohol_ground_truth(vignette),
        womens_health=(womens_health_ground_truth(vignette)
                       if personal["sex"] != "M" else {}),
        children=(children_ground_truth(vignette, proband_yob)
                  if personal["sex"] != "M" else {}),
        medical_history=medical_history_ground_truth(vignette),
        breast_screening=(breast_screening_ground_truth(vignette)
                          if personal["sex"] != "M" else {}),
        prs=prs_ground_truth(vignette),
        relatives=family_ground_truth(vignette),
        ashkenazi=bool(vignette.get("ashkenazi")),
    )