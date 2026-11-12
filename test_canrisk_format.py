
from schema import PersonalLifestyleSection
from clinical_sections import (WomensHealthSection, ChildrenSection,
                                BreastScreeningSection, MedicalHistorySection)
from genetic_sections import PRSSection
from pedigree import Pedigree, Diagnosis, GeneTest
from canrisk_format import (build_pedigree_file, validate_pedigree_file,
                             build_header_lines, complete_parents, COLUMNS,
                             _format_ethnicity, _format_oc_use, _format_mht_use)

PASS, FAIL = [], []


def check(label, cond):
    (PASS if cond else FAIL).append(label)
    print(f"{'PASS' if cond else 'FAIL'}: {label}")


def personal(**over):
    s = PersonalLifestyleSection()
    vals = dict(sex="F", country="UK", age=44, height_cm=165, weight_kg=65)
    vals.update(over)
    for k, v in vals.items():
        s.set_value(k, v)
    return s


def header_of(text):
    return [l for l in text.split("\n") if l.startswith("##")
            and not l.startswith("##FamID")]


def rows_of(text):
    return [l for l in text.split("\n") if l and not l.startswith("##")]


print("=" * 68)
print("Header risk factors")
print("=" * 68)

lines = build_header_lines(personal=personal())
check("mandatory ##CanRisk 4.0 first", lines[0] == "##CanRisk 4.0")
check("height emitted", "##height=165.0" in lines)
check("BMI emitted", any(l.startswith("##BMI=") for l in lines))
check("alcohol NOT emitted when never asked (was silently defaulting to "
      "0.0 -- reporting an unasked patient as a confirmed non-drinker)",
      not any(l.startswith("##alcohol") for l in lines))

from clinical_sections import AlcoholSection
alc_no = AlcoholSection(); alc_no.set_value("drinks_alcohol", "N")
lines2 = build_header_lines(personal=personal(), alcohol=alc_no)
check("alcohol=0.0 emitted when patient confirms they don't drink",
      "##alcohol=0.0" in lines2)

alc_yes = AlcoholSection()
alc_yes.set_value("drinks_alcohol", "Y")
alc_yes.set_value("drinks", [{"type": "wine", "frequency": "weekly", "amount": 2}])
lines3 = build_header_lines(personal=personal(), alcohol=alc_yes)
check("alcohol grams/day emitted when drinks are reported",
      any(l.startswith("##alcohol=") and l != "##alcohol=0.0" for l in lines3))

alc_u = AlcoholSection(); alc_u.set_value("drinks_alcohol", "U")
lines4 = build_header_lines(personal=personal(), alcohol=alc_u)
check("alcohol omitted when patient answer is 'unknown' (unspecified, not 0)",
      not any(l.startswith("##alcohol") for l in lines4))

# Ethnicity -> ONS 'group;background'
check("ethnicity 'Indian' -> Asian or Asian British;Indian",
      _format_ethnicity("Indian") == "Asian or Asian British;Indian")
check("ethnicity 'African' -> Black or Black British;African",
      _format_ethnicity("African") == "Black or Black British;African")
check("ethnicity group-only 'White' -> 'White;'",
      _format_ethnicity("White") == "White;")
check("unrecognised ethnicity omitted rather than guessed (None)",
      _format_ethnicity("Martian") is None)
check("empty ethnicity -> None", _format_ethnicity(None) is None)

# Menopause: integer age, or N if pre-menopausal
w = WomensHealthSection()
for k, v in [("menarche_status", "given"), ("menarche_age", 12),
             ("oc_use", "U"), ("menopause", "N"), ("hrt_use", "U")]:
    w.set_value(k, v)
lines = build_header_lines(personal=personal(), womens=w)
check("menarche emitted", "##menarche=12" in lines)
check("pre-menopausal -> ##menopause=N", "##menopause=N" in lines)
check("unknown oc_use omitted (unspecified)",
      not any(l.startswith("##oc_use") for l in lines))
check("unknown mht_use omitted (unspecified)",
      not any(l.startswith("##mht_use") for l in lines))

w2 = WomensHealthSection()
for k, v in [("menarche_status", "unknown"), ("oc_use", "N"),
             ("menopause", "Y"), ("menopause_age_band", "45-49"),
             ("hrt_use", "N")]:
    w2.set_value(k, v)
lines = build_header_lines(personal=personal(), womens=w2)
check("menopause band 45-49 -> midpoint 47", "##menopause=47" in lines)
check("unknown menarche omitted",
      not any(l.startswith("##menarche") for l in lines))
check("oc_use never -> N", "##oc_use=N" in lines)
check("mht_use never -> N", "##mht_use=N" in lines)

# OC_use encoding: N / F:Years / C:Years
check("oc current, 5-9 yrs -> C:7",
      _format_oc_use({"oc_use": "Y", "oc_years": "5-9",
                      "oc_last_two_years": "Y"}) == "C:7")
check("oc former, 1-4 yrs -> F:3",
      _format_oc_use({"oc_use": "Y", "oc_years": "1-4",
                      "oc_last_two_years": "N"}) == "F:3")
check("oc yes but duration unknown -> omitted (C/F need a year count)",
      _format_oc_use({"oc_use": "Y"}) is None)

# MHT_use encoding: N / F / E / C
check("mht current oestrogen-only -> E",
      _format_mht_use({"hrt_use": "Y", "hrt_last_five_years": "Y",
                       "hrt_type": "oestrogen_only"}) == "E")
check("mht current combined -> C",
      _format_mht_use({"hrt_use": "Y", "hrt_last_five_years": "Y",
                       "hrt_type": "combined"}) == "C")
check("mht former -> F",
      _format_mht_use({"hrt_use": "Y", "hrt_last_five_years": "N"}) == "F")

# Parity / first live birth
c = ChildrenSection()
c.set_value("has_children", "Y"); c.set_value("num_children", 2)
c.set_value("children", [{"year_of_birth": 2008, "sex": "F"},
                          {"year_of_birth": 2011, "sex": "M"}])
lines = build_header_lines(personal=personal(age=44), children=c)  # born ~1982
check("parity emitted", "##parity=2" in lines)
check("first_live_birth derived (2008-1982=26)", "##first_live_birth=26" in lines)

c0 = ChildrenSection(); c0.set_value("has_children", "N")
lines = build_header_lines(personal=personal(), children=c0)
check("no children -> parity=0", "##parity=0" in lines)
check("no children -> no first_live_birth",
      not any(l.startswith("##first_live_birth") for l in lines))

# Impossible first-birth age must NOT be emitted
cbad = ChildrenSection()
cbad.set_value("has_children", "Y"); cbad.set_value("num_children", 1)
cbad.set_value("children", [{"year_of_birth": 1970, "sex": "F"}])  # before mother
lines = build_header_lines(personal=personal(age=44), children=cbad)
check("negative first_live_birth suppressed, not sent to the model",
      not any(l.startswith("##first_live_birth") for l in lines))

# BIRADS
s = BreastScreeningSection()
for k, v in [("had_mammogram", "Y"), ("told_density", "Y"),
             ("density_method", "BI-RADS"), ("birads_category", "c")]:
    s.set_value(k, v)
lines = build_header_lines(personal=personal(), screening=s)
check("BIRADS emitted", "##BIRADS=c" in lines)

s2 = BreastScreeningSection(); s2.set_value("had_mammogram", "N")
lines = build_header_lines(personal=personal(), screening=s2)
check("no mammogram -> no BIRADS",
      not any(l.startswith("##BIRADS") for l in lines))

# TL / endo
m = MedicalHistorySection()
for k, v in [("endometriosis", "Y"), ("tubal_ligation", "N"),
             ("oophorectomy", "N"), ("mastectomy", "N"),
             ("breast_cancer", "N"), ("ovarian_cancer", "N"),
             ("pancreatic_cancer", "N")]:
    m.set_value(k, v)
lines = build_header_lines(personal=personal(), medical=m)
check("TL emitted", "##TL=N" in lines)
check("endo emitted", "##endo=Y" in lines)

# PRS
prs = PRSSection()
for k, v in [("prs_run", "Y"), ("input_mode", "values"),
             ("breast_prs_alpha", 0.45), ("breast_prs_zscore", 1.8)]:
    prs.set_value(k, v)
lines = build_header_lines(personal=personal(), prs=prs)
check("PRS_BC emitted in spec format",
      "##PRS_BC=alpha=0.45, zscore=1.8" in lines)
check("PRS_OC omitted when not provided",
      not any(l.startswith("##PRS_OC") for l in lines))

prs_vcf = PRSSection()
prs_vcf.set_value("prs_run", "Y"); prs_vcf.set_value("input_mode", "vcf")
lines = build_header_lines(personal=personal(), prs=prs_vcf)
check("VCF mode emits no PRS header (needs /vcf2prs/ first)",
      not any(l.startswith("##PRS_") for l in lines))

print()
print("=" * 68)
print("Pedigree rows + structure")
print("=" * 68)

ped = Pedigree(ashkenazi=False)
ped.proband.sex = "F"; ped.proband.age = 44; ped.proband.year_of_birth = 1982
mom = ped.add_relative("mother", alive=True, age=68, year_of_birth=1958)
mom.diagnoses.append(Diagnosis("breast", 52))
mom.gene_tests.append(GeneTest("BRCA2", "P"))
gm = ped.add_relative("maternal_grandmother", alive=False, age_at_death=80,
                      year_of_birth=1940)
gm.diagnoses.append(Diagnosis("ovarian", None))   # affected, age unknown
ped.build_structure()
text = build_pedigree_file(ped, personal=personal())

check("column header present with 28 columns",
      any(l.startswith("##FamID") and len(l[2:].split("\t")) == 28
          for l in text.split("\n")))
for r in rows_of(text):
    if len(r.split("\t")) != 28:
        check(f"row has 28 columns", False)
        break
else:
    check("every data row has 28 columns", True)

prob = next(r.split("\t") for r in rows_of(text) if r.split("\t")[2] == "1")
check("exactly one Target=1",
      sum(1 for r in rows_of(text) if r.split("\t")[2] == "1") == 1)
check("proband age/yob correct", prob[9] == "44" and prob[10] == "1982")

momrow = next(r.split("\t") for r in rows_of(text) if r.split("\t")[1] == "mother")
check("mother BC1 = 52", momrow[11] == "52")
check("mother BRCA2 = S:P", momrow[18] == "S:P")
check("mother alive -> Dead=0", momrow[8] == "0")

gmrow = next(r.split("\t") for r in rows_of(text) if r.split("\t")[3] == gm.id)
check("grandmother deceased -> Dead=1", gmrow[8] == "1")
check("grandmother Age = age at death (80)", gmrow[9] == "80")
check("affected with unknown dx age -> OC='AU' (not 0/unaffected)",
      gmrow[13] == "AU")

check("file passes structural validation", validate_pedigree_file(text) == [])

# both-parents-or-neither rule
ped2 = Pedigree()
ped2.proband.sex = "F"; ped2.proband.age = 40
ped2.add_relative("mother", alive=True, age=70)
ped2.build_structure()
check("proband has exactly one parent before completion",
      bool(ped2.proband.mother_id) and not ped2.proband.father_id)
synth = complete_parents(ped2)
check("missing father synthesised to satisfy spec rule", len(synth) == 1)
check("proband now has both parents",
      bool(ped2.proband.mother_id) and bool(ped2.proband.father_id))

t2 = build_pedigree_file(ped2, personal=personal())
check("one-parent family still validates after completion",
      validate_pedigree_file(t2) == [])

# proband's own medical history reaches her pedigree row
m2 = MedicalHistorySection()
for k, v in [("endometriosis", "N"), ("tubal_ligation", "N"),
             ("oophorectomy", "N"), ("mastectomy", "N"),
             ("breast_cancer", "Y"), ("breast_cancer_age", 41),
             ("ovarian_cancer", "N"), ("pancreatic_cancer", "N")]:
    m2.set_value(k, v)
ped3 = Pedigree(); ped3.proband.sex = "F"; ped3.proband.age = 44
t3 = build_pedigree_file(ped3, personal=personal(), medical=m2)
prob3 = next(r.split("\t") for r in rows_of(t3) if r.split("\t")[2] == "1")
check("proband's own breast cancer (Medical History) lands on BC1",
      prob3[11] == "41")

# Ashkenazi
ped4 = Pedigree(ashkenazi=True); ped4.proband.sex = "F"; ped4.proband.age = 40
t4 = build_pedigree_file(ped4, personal=personal())
check("Ashkenazi status set on rows",
      all(r.split("\t")[16] == "1" for r in rows_of(t4)))

# HOXB13 uses its own codes
ped5 = Pedigree(); ped5.proband.sex = "F"; ped5.proband.age = 40
ped5.proband.gene_tests.append(GeneTest("HOXB13", "HET"))
t5 = build_pedigree_file(ped5, personal=personal())
p5 = next(r.split("\t") for r in rows_of(t5) if r.split("\t")[2] == "1")
check("HOXB13 HET -> T:HET (col 27)", p5[26] == "T:HET")

ped6 = Pedigree(); ped6.proband.sex = "F"; ped6.proband.age = 40
ped6.proband.gene_tests.append(GeneTest("HOXB13", "P"))  # invalid for HOXB13
t6 = build_pedigree_file(ped6, personal=personal())
p6 = next(r.split("\t") for r in rows_of(t6) if r.split("\t")[2] == "1")
check("invalid HOXB13 result 'P' -> untested, not coerced", p6[26] == "0:0")

print()
print("=" * 68)
print("Validator catches malformed files")
print("=" * 68)

bad = "##CanRisk 4.0\n##" + "\t".join(COLUMNS) + "\n" + "\t".join(["X"] * 28)
probs = validate_pedigree_file(bad)
check("no Target=1 is caught", any("target" in p.lower() for p in probs))

hdr = "##CanRisk 4.0\n##" + "\t".join(COLUMNS) + "\n"
short = hdr + "\t".join(["X"] * 10)
check("wrong column count is caught",
      any("columns" in p for p in validate_pedigree_file(short)))

orphan = hdr + "\t".join(
    ["F1", "me", "1", "I0", "GHOST", "GHOST2", "F", "0", "0", "40", "1986"]
    + ["0"] * 5 + ["0"] + ["0:0"] * 10 + ["0:0:0:0:0"])
check("parent id with no row is caught",
      any("has no row" in p for p in validate_pedigree_file(orphan)))

print()
print("=" * 68)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED:", f)
print("=" * 68)

print()
print("=" * 68)
print("Regression: duplicate relatives no longer orphan the pedigree")
print("=" * 68)

from experiments.oldintake import apply_family_extraction

ped7 = Pedigree(); ped7.proband.sex = "F"; ped7.proband.age = 37
apply_family_extraction(ped7, {"relatives": [
    {"relationship": "mother", "diagnoses": [{"cancer_type": "breast"}]},
    {"relationship": "maternal_grandmother", "diagnoses": [{"cancer_type": "ovarian"}]},
]})
# A follow-up turn with NO existing_id (worst case) must still merge, not duplicate
apply_family_extraction(ped7, {"relatives": [
    {"relationship": "mother", "alive": True, "age": 65},
    {"relationship": "maternal_grandmother", "alive": True, "age": 96},
]})
check("no duplicate mother created", len(ped7.get_by_rel("mother")) == 1)
check("no duplicate grandmother created",
      len(ped7.get_by_rel("maternal_grandmother")) == 1)
check("details merged onto the single mother node",
      ped7.get_by_rel("mother")[0].age == 65)
ped7.build_structure()
t7 = build_pedigree_file(ped7, personal=personal(age=37))
check("previously-duplicated family produces a CLEAN, connected pedigree",
      validate_pedigree_file(t7) == [])
orphans7 = [i for i in ped7.individuals.values()
           if i.id not in ped7.connected_to_proband()]
check("no individual is disconnected from the target (the exact HTTP 400 cause)",
      orphans7 == [])

print()
print("=" * 68)
print(f"FINAL RESULT: {len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED:", f)
print("=" * 68)

print()
print("=" * 68)
print("Regression: diagnosed relative without a resolvable year of birth")
print("(the exact 'Cancer Error' HTTP 400 from real testing)")
print("=" * 68)


ped8 = Pedigree(); ped8.proband.sex = "F"; ped8.proband.age = 37
mom8 = ped8.add_relative("mother", alive=True, age=65)
mom8.diagnoses.append(Diagnosis("breast", None))
check("age-only relative with a diagnosis: year of birth IS resolved",
      mom8.resolved_year_of_birth() is not None)
t8 = build_pedigree_file(ped8, personal=personal(age=37))
check("...and the generated file passes validation (this exact case 400'd "
      "in real testing before the fix)", validate_pedigree_file(t8) == [])

ped9 = Pedigree(); ped9.proband.sex = "F"; ped9.proband.age = 37
ped9.add_relative("mother", alive=True, age=65)
gm9 = ped9.add_relative("maternal_grandmother", alive=False, age_at_death=80)
gm9.diagnoses.append(Diagnosis("ovarian", 71))
ped9.build_structure()
check("deceased relative, only age-at-death known, with a diagnosis: "
      "year of birth correctly NOT resolvable", gm9.resolved_year_of_birth() is None)
check("...and missing_detail() flags it as a gap to ask about",
      any("year of birth" in g for g in ped9.missing_detail()))
t9 = build_pedigree_file(ped9, personal=personal(age=37))
check("grandmother NOT pruned (she IS connected via mother)",
      "maternal" in t9)
check("...and the validator catches the missing Yob locally instead of a live 400",
      any("Yob=0" in p for p in validate_pedigree_file(t9)))

print()
print("=" * 68)
print(f"FINAL RESULT (v2): {len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED:", f)
print("=" * 68)

print()
print("=" * 68)
print("Regression: reported bugs -- OC follow-ups, PRS enum choices,")
print("unresolvable YOB estimation, aunt/uncle connection")
print("=" * 68)

from clinical_sections import WomensHealthSection
w10 = WomensHealthSection()
w10.set_value("menarche_age", 13)
w10.set_value("oc_use", "Y")
w10.set_value("menopause", "N")
w10.set_value("hrt_use", "N")
missing10 = w10.missing_required()
check("oc_use='Y' requires oc_years follow-up", "oc_years" in missing10)
check("oc_use='Y' requires oc_last_two_years follow-up",
      "oc_last_two_years" in missing10)

w11 = WomensHealthSection()
w11.set_value("menarche_age", 13)
w11.set_value("oc_use", "N")
w11.set_value("menopause", "Y")
w11.set_value("hrt_use", "N")
check("menopause='Y' requires menopause_age_band follow-up",
      "menopause_age_band" in w11.missing_required())


ped10 = Pedigree()
ped10.proband.sex = "F"; ped10.proband.age = 37; ped10.proband.year_of_birth = 1989
mom10 = ped10.add_relative("mother")
mom10.diagnoses.append(Diagnosis("breast", 45))
uncle10 = ped10.add_relative("paternal_uncle", alive=True, age=60)
ped10.build_structure()

estimated10 = ped10.estimate_missing_years(ped10.proband.year_of_birth)
check("diagnosed relative with zero info gets an estimated YOB",
      len(estimated10) == 1 and "mother" in estimated10[0])

t10 = build_pedigree_file(ped10, personal=personal(age=37))
check("previously-400ing scenario (real reported bug) now validates cleanly",
      validate_pedigree_file(t10) == [])
orphans10 = [i for i in ped10.individuals.values()
            if i.id not in ped10.connected_to_proband()]
check("paternal uncle (mentioned with no father ever described) is connected, "
      "not silently dropped", uncle10.id not in [o.id for o in orphans10])
check("uncle actually has parent pointers set",
      uncle10.father_id is not None and uncle10.mother_id is not None)

print()
print("=" * 68)
print(f"FINAL RESULT (v3): {len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED:", f)
print("=" * 68)

print()
print("=" * 68)
print("Regression: Personal/Lifestyle asks sex+country grouped, then")
print("everything else one field at a time")
print("=" * 68)

import agent as _agent
from schema import PersonalLifestyleSection as _PLS

_captured = []
_agent.llm.call_text = lambda system, messages, max_tokens=1500: (_captured.append(system), "MOCK")[1]

_s = _PLS()
_agent.compose_question(_s, [])
check("sex+country asked together when both missing",
      "sex" in _captured[-1] and "country" in _captured[-1]
      and "together" in _captured[-1].lower())

_s.set_value("sex", "F")
_agent.compose_question(_s, [])
check("once sex is answered, country alone is asked (not re-grouped)",
      "EXACTLY ONE thing" in _captured[-1] and "country" in _captured[-1])

_s.set_value("country", "UK")
_agent.compose_question(_s, [])
check("fields after the opening pair (height/weight/dob) go one at a time",
      "EXACTLY ONE thing" in _captured[-1] and "height_cm" in _captured[-1])

print()
print("=" * 68)
print(f"FINAL RESULT (v4): {len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED:", f)
print("=" * 68)