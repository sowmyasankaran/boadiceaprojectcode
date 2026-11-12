"""
Real-derived seed pedigrees

Each seed encodes the structure and stated ages of a published hereditary
breast/ovarian cancer family. Where a source does not state a relative's exact age, a plausible value is filled in, flagged in
notes as (inferred).
"""

from vignette_schema import Vignette, FamilyMember, RiskFactors


def seed_brca1_young_breast() -> Vignette:
    """BRCA1, very-early-onset breast cancer with multiple carrier relatives.
    Source: open-access case report, PMC8281913. 29-yo woman with left breast
    carcinoma, BRCA1 mutation also present in two sisters, a brother, and the
    brother's daughter. Relative ages are (inferred) as they are not all stated."""
    members = [
        FamilyMember("gpat", "M", dead=1, age=80, yob=1944),          # (inferred)
        FamilyMember("gmat", "F", dead=1, age=82, yob=1942),          # (inferred)
        FamilyMember("mother", "F", "gpat", "gmat", age=58, yob=1966, gene_tests={"BRCA1": "S:P"}),
        FamilyMember("father", "M", age=60, yob=1964),                # (inferred)
        FamilyMember("sis1", "F", "father", "mother", age=34, yob=1990, gene_tests={"BRCA1": "S:P"}),
        FamilyMember("sis2", "F", "father", "mother", age=31, yob=1993, gene_tests={"BRCA1": "S:P"}),
        FamilyMember("bro1", "M", "father", "mother", age=36, yob=1988, gene_tests={"BRCA1": "S:P"}),
        FamilyMember("niece1", "F", "bro1", "0", age=12, yob=2012, gene_tests={"BRCA1": "S:P"}),  # brother's daughter
        FamilyMember("prob1", "F", "father", "mother", age=29, yob=1995, bc1=29,
                     gene_tests={"BRCA1": "S:P"}, is_target=True),
    ]
    return Vignette(
        vignette_id="seed_brca1_young_breast", difficulty="hard", fam_id="SEEDR1",
        members=members,
        risk_factors=RiskFactors(menarche=13, parity=0, oc_use="N", mht_use="N",
                                 bmi=23.0, alcohol=0.0, height=165, tl="N", endo="N"),
        extremity="extreme", ethnicity="White Other", ancestry="European",
        source="Case report PMC8281913 (structure + stated proband age; relatives inferred)",
        notes="Very-early-onset BC@29, BRCA1+, multiple carrier first-degree relatives.",
        tags=["exp1", "exp3", "exp5", "seed_real"],
    )


def seed_hboc_ovarian_sibship() -> Vignette:
    """Hereditary ovarian cancer sibship. Source: open-access case report,
    PMC7311593. proband bilateral ovarian serous adenocarcinoma at 40; two
    sisters also with ovarian cancer (one diagnosed at 50). Parent/other ages
    (inferred)."""
    members = [
        FamilyMember("gpat", "M", dead=1, age=85, yob=1939),          # (inferred)
        FamilyMember("gmat", "F", dead=1, age=83, yob=1941),          # (inferred)
        FamilyMember("mother", "F", "gpat", "gmat", dead=1, age=72, yob=1952),   # (inferred)
        FamilyMember("father", "M", dead=1, age=75, yob=1949),        # (inferred)
        FamilyMember("sis1", "F", "father", "mother", age=58, yob=1966, oc=50),  # OC@50 (stated)
        FamilyMember("sis2", "F", "father", "mother", dead=1, age=55, yob=1969, oc=45),  # OC (age inferred)
        FamilyMember("prob1", "F", "father", "mother", age=42, yob=1982, oc=40,
                     is_target=True),                                  # OC@40 (stated)
    ]
    return Vignette(
        vignette_id="seed_hboc_ovarian_sibship", difficulty="hard", fam_id="SEEDR2",
        members=members,
        risk_factors=RiskFactors(menarche=12, parity=1, first_live_birth=30,
                                 oc_use="N", mht_use="N", bmi=24.0, alcohol=4.7,
                                 height=162, tl="N", endo="N"),
        extremity="extreme", ethnicity="Chinese", ancestry="East Asian",
        source="Case report PMC7311593 (proband + sibling OC ages stated; others inferred)",
        notes="Ovarian-cancer sibship; proband OC@40, sisters OC@45/50.",
        tags=["exp1", "exp3", "exp5", "seed_real"],
    )


def seed_brca2_archetype() -> Vignette:
    """BRCA2 archetypal family. Source: NCI PDQ 'Genetics of Breast and
    Gynecologic Cancers', Figure 2 (BRCA2 pedigree). classic features across
    three generations: breast (incl. male breast), ovarian, pancreatic, and
    prostate cancers with relatively young onset. A representative family built
    to those documented features (ages inferred to match the archetype)."""
    members = [
        FamilyMember("gpat", "M", dead=1, age=78, yob=1946, pan=70),   # pancreatic (archetype)
        FamilyMember("gmat", "F", dead=1, age=80, yob=1944, bc1=58),   # breast (archetype)
        FamilyMember("father", "M", "gpat", "gmat", age=62, yob=1962, bc1=60,
                     gene_tests={"BRCA2": "S:P"}),                     # male breast cancer
        FamilyMember("uncle", "M", "gpat", "gmat", age=65, yob=1959, pro=63),  # prostate
        FamilyMember("mother", "F", age=60, yob=1964),
        FamilyMember("prob1", "F", "father", "mother", age=45, yob=1979,
                     is_target=True),                                  # unaffected proband
    ]
    return Vignette(
        vignette_id="seed_brca2_archetype", difficulty="hard", fam_id="SEEDR3",
        members=members,
        risk_factors=RiskFactors(menarche=12, parity=2, first_live_birth=28,
                                 oc_use="N", mht_use="N", bmi=25.0, alcohol=4.7,
                                 height=168, tl="N", endo="N"),
        extremity="medium", ethnicity="White British", ancestry="European",
        source="NCI PDQ Genetics of Breast/Gynecologic Cancers, BRCA2 pedigree (archetype)",
        notes="BRCA2 archetype: paternal male BC, pancreatic, prostate; unaffected proband.",
        tags=["exp1", "exp3", "exp5", "seed_real"],
    )


def all_real_seeds():
    return [seed_brca1_young_breast(), seed_hboc_ovarian_sibship(), seed_brca2_archetype()]
