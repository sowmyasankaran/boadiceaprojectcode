"""
Vignette schema

store only the structured truth + derived artifacts; risk numbers are filled
in by the real model, not by us

Family is modelled explicitly (variable structure) because BOADICEA risk is
dominated by family history. Column order matches the CanRisk 2.0
format we already validated against the live API (returned 200s).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json


# Gene panel columns, in the exact order the CanRisk row expects them.
GENE_COLUMNS = ["BRCA1", "BRCA2", "PALB2", "ATM", "CHEK2", "BARD1",
                "RAD51D", "RAD51C", "BRIP1"]


@dataclass
class FamilyMember:
    indiv_id: str                 # alphanumeric, <= 7 chars
    sex: str                      # "M" | "F"
    father_id: str = "0"
    mother_id: str = "0"
    dead: int = 0
    age: int = 0                  # 0 = unknown
    yob: int = 0                  # 0 = unknown
    bc1: int = 0                  # age at 1st breast cancer dx, 0 = none
    bc2: int = 0                  # contralateral
    oc: int = 0                   # ovarian
    pro: int = 0                  # prostate
    pan: int = 0                  # pancreatic
    ashkn: int = 0
    is_target: bool = False
    # gene test results: {gene: "S:P" | "S:N" | "0:0"} (test:result); default untested
    gene_tests: Dict[str, str] = field(default_factory=dict)

    def gene_cols(self) -> str:
        return "\t".join(self.gene_tests.get(g, "0:0") for g in GENE_COLUMNS)

    def to_row(self, fam_id: str, name: str = "NA") -> str:
        target = 1 if self.is_target else 0
        return (
            f"{fam_id}\t{name}\t{target}\t{self.indiv_id}\t{self.father_id}\t"
            f"{self.mother_id}\t{self.sex}\t0\t{self.dead}\t{self.age}\t{self.yob}\t"
            f"{self.bc1}\t{self.bc2}\t{self.oc}\t{self.pro}\t{self.pan}\t{self.ashkn}\t"
            f"{self.gene_cols()}\t0:0:0:0:0"
        )


@dataclass
class RiskFactors:
    """Header-level (##key=value) risk factors. None => omit the line."""
    menarche: Optional[int] = None
    parity: Optional[int] = None
    first_live_birth: Optional[int] = None
    oc_use: Optional[str] = None          # e.g. "N" or "C:2"
    mht_use: Optional[str] = None
    bmi: Optional[float] = None
    alcohol: Optional[float] = None       # grams/day
    height: Optional[float] = None        # cm
    tl: Optional[str] = None              # tubal ligation "Y"/"N"
    endo: Optional[str] = None            # endometriosis "Y"/"N"
    menopause: Optional[int] = None

    def header_lines(self) -> List[str]:
        mapping = {
            "menarche": self.menarche, "parity": self.parity,
            "first_live_birth": self.first_live_birth, "oc_use": self.oc_use,
            "mht_use": self.mht_use, "BMI": self.bmi, "alcohol": self.alcohol,
            "height": self.height, "TL": self.tl, "endo": self.endo,
            "menopause": self.menopause,
        }
        return [f"##{k}={v}" for k, v in mapping.items() if v is not None]


@dataclass
class Vignette:
    vignette_id: str
    difficulty: str                        # "easy" | "moderate" | "hard"
    fam_id: str
    members: List[FamilyMember]
    risk_factors: RiskFactors
    ashkenazi: bool = False
    # extremity axis (separate from difficulty): "normal" | "medium" | "extreme".
    # difficulty = family-structure complexity; extremity = how typical the
    # values are. Extreme cases stay INSIDE plausibility bounds (valid outputs);
    # implausible cases live only in the Exp-2 perturbation set.
    extremity: str = "normal"
    # diversity fields (clinically load-bearing, not cosmetic):
    ethnicity: Optional[str] = None        # ONS-style group -> baseline incidence
    ancestry: Optional[str] = None         # PRS genetic ancestry (distinct field)
    # provenance: citation for real-derived seeds, or "sampled" for generated
    source: Optional[str] = None
    # free-text description of the profile, for auditing / prompt seeding
    notes: str = ""
    # metadata used by experiments (which conditions this vignette supports)
    tags: List[str] = field(default_factory=list)

    def target(self) -> FamilyMember:
        return next(m for m in self.members if m.is_target)

    def to_canrisk_file(self) -> str:
        header = ["##CanRisk 2.0"] + self.risk_factors.header_lines()
        col_header = ("##FamID\tName\tTarget\tIndivID\tFathID\tMothID\tSex\tMZtwin\t"
                      "Dead\tAge\tYob\tBC1\tBC2\tOC\tPRO\tPAN\tAshkn\t"
                      + "\t".join(GENE_COLUMNS) + "\tER:PR:HER2:CK14:CK56")
        rows = [m.to_row(self.fam_id) for m in self.members]
        return "\n".join(header + [col_header] + rows) + "\n"

    def to_dict(self) -> dict:
        return asdict(self) | {"canrisk_file": self.to_canrisk_file()}

    @staticmethod
    def from_dict(d: dict) -> "Vignette":
        d = {k: v for k, v in d.items() if k != "canrisk_file"}
        d["members"] = [FamilyMember(**m) for m in d["members"]]
        d["risk_factors"] = RiskFactors(**d["risk_factors"])
        return Vignette(**d)


def save_bank(vignettes: List[Vignette], path: str):
    with open(path, "w") as f:
        json.dump([v.to_dict() for v in vignettes], f, indent=2)


def load_bank(path: str) -> List[Vignette]:
    with open(path) as f:
        return [Vignette.from_dict(d) for d in json.load(f)]
