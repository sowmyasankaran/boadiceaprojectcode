
from typing import Optional, List, Dict, Any

from pedigree import Pedigree, Individual


CANRISK_VERSION = "4.0"

# 28 columns, in spec order.
COLUMNS = [
    "FamID", "Name", "Target", "IndivID", "FathID", "MothID", "Sex", "MZtwin",
    "Dead", "Age", "Yob", "BC1", "BC2", "OC", "PRO", "PAN", "Ashkn",
    "BRCA1", "BRCA2", "PALB2", "ATM", "CHEK2", "BARD1", "RAD51D", "RAD51C",
    "BRIP1", "HOXB13", "ER:PR:HER2:CK14:CK56",
]

# Genes in pedigree-column order. HOXB13 is deliberately excluded here and
# handled separately -- its result codes differ from every other gene
# (HOM/HET/N rather than P/N), per spec column 27.
STANDARD_GENE_COLUMNS = ["BRCA1", "BRCA2", "PALB2", "ATM", "CHEK2", "BARD1",
                         "RAD51D", "RAD51C", "BRIP1"]

# ONS ethnic groups -> valid backgrounds (v4 spec table). Header value is
# "<group>;<background>".
ONS_ETHNICITY = {
    "White": ["English/Welsh/Scottish/Northern Irish/British", "Irish",
              "Gypsy or Irish Traveller", "Any other White background"],
    "Mixed/Multiple ethnic groups": ["White and Black Caribbean",
                                     "White and Black African",
                                     "White and Asian",
                                     "Any other Mixed/Multiple ethnic background"],
    "Asian or Asian British": ["Indian", "Pakistani", "Bangladeshi", "Chinese",
                               "Any other Asian background"],
    "Black or Black British": ["African", "Caribbean",
                               "Any other Black/African/Caribbean background"],
    "Other ethnic group": ["Arab", "Any other ethnic group"],
    "Unknown": [""],
}

BAND_MIDPOINTS = {
    "menopause_age_band": {
        "<40": 38,      # open band; just under the boundary
        "40-44": 42,
        "45-49": 47,
        "50-54": 52,
        "54+": 56,      # open band; just over the boundary
    },
    "oc_years": {
        "<1": 1,        # spec wants an integer year count; <1 rounds to 1
        "1-4": 3,
        "5-9": 7,
        "10-14": 12,
        "15+": 15,      # open band; at the boundary
    },
}


def _na(value) -> str:
    return "NA" if value is None else str(value)


def _alcohol_grams(alcohol, personal):
    """Grams/day if known, else None (omit the tag).
    Prefers the dedicated Alcohol section; falls back to any drinks captured
    on the personal section (the legacy single-section path)."""
    if alcohol is not None:
        v = alcohol.values
        if v.get("drinks_alcohol") == "N":
            return 0.0
        if v.get("drinks_alcohol") == "Y":
            return alcohol.alcohol_grams_per_day()
        return None  # unknown -> unspecified
    if personal is not None and personal.values.get("drinks"):
        return personal.alcohol_grams_per_day()
    return None


def build_header_lines(personal=None, womens=None, children=None,
                        screening=None, medical=None, prs=None,
                        alcohol=None) -> List[str]:

    lines: List[str] = [f"##CanRisk {CANRISK_VERSION}"]

    age = yob = None
    if personal is not None:
        ay = personal.age_yob()
        if ay:
            age, yob = ay

        eth = _format_ethnicity(personal.values.get("ethnic_group"))
        if eth:
            lines.append(f"##ethnicity={eth}")

        h = personal.values.get("height_cm")
        if h is not None:
            lines.append(f"##height={round(float(h), 1)}")
        bmi = personal.bmi()
        if bmi is not None:
            lines.append(f"##BMI={bmi}")

    grams = _alcohol_grams(alcohol, personal)
    if grams is not None:
        lines.append(f"##alcohol={grams}")

    if womens is not None:
        v = womens.values
        if v.get("menarche_age") is not None:
            lines.append(f"##menarche={v['menarche_age']}")

        # Menopause: spec allows an integer age, or 'N' if pre-menopausal.
        if v.get("menopause") == "N":
            lines.append("##menopause=N")
        elif v.get("menopause") == "Y":
            band = v.get("menopause_age_band")
            mapped = BAND_MIDPOINTS["menopause_age_band"].get(band)
            if mapped is not None:
                lines.append(f"##menopause={mapped}")
        # "U" (unknown) -> omit, i.e. unspecified.

        oc = _format_oc_use(v)
        if oc:
            lines.append(f"##oc_use={oc}")
        mht = _format_mht_use(v)
        if mht:
            lines.append(f"##mht_use={mht}")

    if children is not None:
        v = children.values
        if v.get("has_children") == "N":
            lines.append("##parity=0")
        elif v.get("num_children") is not None:
            lines.append(f"##parity={v['num_children']}")
            if yob is not None:
                afb = children.age_at_first_birth(yob)
                # Only emit a physically sensible value -- a negative or absurd
                # first-birth age means the intake data is wrong, and the
                # cross-section check will have already flagged it. Emitting it
                # anyway would either 400 or silently skew the model.
                if afb is not None and 0 < afb < 70:
                    lines.append(f"##first_live_birth={afb}")

    if screening is not None:
        v = screening.values
        if v.get("told_density") == "Y":
            method = v.get("density_method")
            if method == "BI-RADS" and v.get("birads_category"):
                lines.append(f"##BIRADS={v['birads_category']}")


    if medical is not None:
        v = medical.values
        if v.get("tubal_ligation") in ("Y", "N"):
            lines.append(f"##TL={v['tubal_ligation']}")
        if v.get("endometriosis") in ("Y", "N"):
            lines.append(f"##endo={v['endometriosis']}")

    if prs is not None:
        v = prs.values

        if v.get("prs_run") == "Y":
            bc = _format_prs(v.get("breast_prs_alpha"), v.get("breast_prs_zscore"))
            if bc:
                lines.append(f"##PRS_BC={bc}")
            oc = _format_prs(v.get("ovarian_prs_alpha"), v.get("ovarian_prs_zscore"))
            if oc:
                lines.append(f"##PRS_OC={oc}")
        # input_mode == 'vcf': the PRS must be computed from the VCF via the
        # /vcf2prs/ endpoint first; a file can't ride in this header.

    return lines


def _format_ethnicity(value: Optional[str]) -> Optional[str]:

    if not value:
        return None
    v = value.strip()
    if ";" in v:
        group, _, background = v.partition(";")
        if group.strip() in ONS_ETHNICITY:
            return f"{group.strip()};{background.strip()}"
        return None
    # exact group name given, no background
    for group, backgrounds in ONS_ETHNICITY.items():
        if v.lower() == group.lower():
            return f"{group};"
        for bg in backgrounds:
            if bg and v.lower() == bg.lower():
                return f"{group};{bg}"
    return None


def _format_oc_use(v: dict) -> Optional[str]:

    if v.get("oc_use") == "N":
        return "N"
    if v.get("oc_use") != "Y":
        return None   # unknown -> unspecified
    years = BAND_MIDPOINTS["oc_years"].get(v.get("oc_years"))
    if years is None:
        return None   # duration unknown; C/F both require a year count
    current = v.get("oc_last_two_years") == "Y"
    return f"{'C' if current else 'F'}:{years}"


def _format_mht_use(v: dict) -> Optional[str]:

    if v.get("hrt_use") == "N":
        return "N"
    if v.get("hrt_use") != "Y":
        return None
    current = v.get("hrt_last_five_years") == "Y"
    if not current:
        return "F"
    return "E" if v.get("hrt_type") == "oestrogen_only" else "C"


def _format_prs(alpha, zscore) -> Optional[str]:
    if alpha is None or zscore is None:
        return None
    return f"alpha={alpha}, zscore={zscore}"


def complete_parents(ped: Pedigree) -> List[str]:

    synthesized = []
    for ind in list(ped.individuals.values()):
        has_f, has_m = ind.father_id, ind.mother_id
        if bool(has_f) == bool(has_m):
            continue  # both or neither -- valid
        if not has_f:
            f = ped.add_relative("other", sex="M",
                                 name=f"unknown father of {ind.rel_to_proband}")
            ind.father_id = f.id
            synthesized.append(f"father of {ind.rel_to_proband}")
        else:
            m = ped.add_relative("other", sex="F",
                                 name=f"unknown mother of {ind.rel_to_proband}")
            ind.mother_id = m.id
            synthesized.append(f"mother of {ind.rel_to_proband}")
    return synthesized


def _dx_age(ind: Individual, cancer_type: str) -> str:
    """0 = unaffected, integer = age, AU = affected but age unknown."""
    for d in ind.diagnoses:
        if d.cancer_type == cancer_type:
            return str(d.age) if d.age is not None else "AU"
    return "0"


def _gene_cell(ind: Individual, gene: str) -> str:
    """type:result -- 0=untested, S=mutation search, T=direct test;
    result 0=untested, P=positive, N=negative."""
    for g in ind.gene_tests:
        if g.gene == gene:
            if g.result == "P":
                return "S:P"
            if g.result == "N":
                return "S:N"
    return "0:0"


def _hoxb13_cell(ind: Individual) -> str:
    """HOXB13 (column 27) uses different codes to every other gene: type T
    only, result HOM/HET/N. A 'P' result is NOT valid here, so anything that
    isn't an explicit HOM/HET/N is reported as untested rather than coerced."""
    for g in ind.gene_tests:
        if g.gene == "HOXB13" and g.result in ("HOM", "HET", "N"):
            return f"T:{g.result}"
    return "0:0"


def _row(ind: Individual, fam_id: str, ashkenazi: bool) -> str:
    dead = "1" if ind.alive is False else "0"
    # Age = "age at last follow up": current age if alive, age at death if not.
    age = ind.age if ind.age is not None else ind.age_at_death
    cells = [
        fam_id,
        (ind.name or ind.rel_to_proband)[:8],
        "1" if ind.is_proband else "0",
        ind.id[:7],
        (ind.father_id or "0")[:7],
        (ind.mother_id or "0")[:7],
        ind.sex or "F",
        "0",                                   # MZtwin -- not collected
        dead,
        str(age if age is not None else 0),
        str(ind.resolved_year_of_birth() if ind.resolved_year_of_birth() is not None else 0),
        _dx_age(ind, "breast"),
        _dx_age(ind, "contralateral_breast"),
        _dx_age(ind, "ovarian"),
        "0",                                   # PRO -- prostate not collected
        _dx_age(ind, "pancreatic"),
        "1" if ashkenazi else "0",
    ]
    cells += [_gene_cell(ind, g) for g in STANDARD_GENE_COLUMNS]
    cells.append(_hoxb13_cell(ind))
    cells.append(ind.pathology.to_column() if ind.pathology else "0:0:0:0:0")
    return "\t".join(cells)


def build_pedigree_file(ped: Pedigree, personal=None, womens=None,
                         children=None, screening=None, medical=None,
                         prs=None, alcohol=None, fam_id: str = "FAM1",
                         dropped_out: Optional[List[str]] = None) -> str:

    _apply_proband_medical_history(ped, medical)
    dropped = ped.prune_disconnected()
    if dropped_out is not None:
        dropped_out.extend(dropped)
    complete_parents(ped)

    lines = build_header_lines(personal, womens, children, screening, medical,
                               prs, alcohol)
    lines.append("##" + "\t".join(COLUMNS))
    ashkn = bool(ped.ashkenazi)
    for ind in ped.individuals.values():
        lines.append(_row(ind, fam_id[:13], ashkn))
    return "\n".join(lines) + "\n"


def _apply_proband_medical_history(ped: Pedigree, medical) -> None:

    if medical is None:
        return
    from pedigree import Diagnosis
    v = medical.values
    mapping = [
        ("breast_cancer", "breast_cancer_age", "breast"),
        ("contralateral_breast_cancer", "contralateral_breast_cancer_age",
         "contralateral_breast"),
        ("ovarian_cancer", "ovarian_cancer_age", "ovarian"),
        ("pancreatic_cancer", "pancreatic_cancer_age", "pancreatic"),
    ]
    existing = {d.cancer_type for d in ped.proband.diagnoses}
    for flag, agekey, cancer_type in mapping:
        if v.get(flag) == "Y" and cancer_type not in existing:
            ped.proband.diagnoses.append(Diagnosis(cancer_type, v.get(agekey)))


def validate_pedigree_file(text: str) -> List[str]:

    problems = []
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines or not lines[0].startswith("##CanRisk"):
        problems.append("missing mandatory ##CanRisk version header")
    col_idx = next((i for i, l in enumerate(lines) if l.startswith("##FamID")), None)
    if col_idx is None:
        problems.append("missing mandatory ##FamID column header")
        return problems

    rows = lines[col_idx + 1:]
    if not rows:
        problems.append("no pedigree data rows")
    ids = set()
    targets = 0
    for r in rows:
        cells = r.split("\t")
        if len(cells) != len(COLUMNS):
            problems.append(f"row has {len(cells)} columns, expected {len(COLUMNS)}: {cells[:4]}")
            continue
        ids.add(cells[3])
        if cells[2] == "1":
            targets += 1
        if len(cells[3]) > 7:
            problems.append(f"IndivID longer than 7 chars: {cells[3]}")
    if targets != 1:
        problems.append(f"expected exactly 1 target (Target=1), found {targets}")
    for r in rows:
        cells = r.split("\t")
        if len(cells) != len(COLUMNS):
            continue
        fath, moth = cells[4], cells[5]
        if (fath == "0") != (moth == "0"):
            problems.append(f"individual {cells[3]} has exactly one parent; "
                            f"spec requires both or neither")
        for pid, label in ((fath, "father"), (moth, "mother")):
            if pid != "0" and pid not in ids:
                problems.append(f"{label} id {pid} referenced by {cells[3]} "
                                f"has no row")

        diagnosis_cells = [cells[11], cells[12], cells[13], cells[15]]
        if any(c != "0" for c in diagnosis_cells) and cells[10] == "0":
            problems.append(f"individual {cells[3]} has a cancer diagnosis "
                            f"but Yob=0 -- CanRisk will reject this; either "
                            f"their age or year of birth must be known/estimated")
    return problems