"""
generates Experiment 2 (safety-net) data.

Each perturbation takes clean vignette and returns a corrupted copy plus the
expected verification verdict. Because we inject a known error, the expected
output is known.

Expected verdict shape:
  {"expect_flag": True/False, "severity": "hard"|"soft"|None,
   "field": <name>, "kind": <error class>}
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Callable, Optional, List

from vignette_schema import Vignette


@dataclass
class PerturbedCase:
    vignette: Vignette
    expect_flag: bool
    severity: Optional[str]
    field: Optional[str]
    kind: str
    description: str



def implausible_height(v: Vignette) -> PerturbedCase:
    c = copy.deepcopy(v)
    c.risk_factors.height = 125.6
    c.vignette_id += "__implausible_height"
    return PerturbedCase(c, True, "soft", "height_cm", "implausible_value",
                         "Adult height set to 125.6cm.")


def impossible_timeline(v: Vignette) -> PerturbedCase:
    c = copy.deepcopy(v)
    c.risk_factors.first_live_birth = c.target().age + 10
    c.vignette_id += "__impossible_timeline"
    return PerturbedCase(c, True, "hard", "first_live_birth_age", "impossible_timeline",
                         "Age at first birth exceeds current age.")


def unit_mixup_height(v: Vignette) -> PerturbedCase:
    c = copy.deepcopy(v)
    c.risk_factors.height = 65.0
    c.vignette_id += "__unit_mixup_height"
    return PerturbedCase(c, True, "soft", "height_cm", "unit_mixup",
                         "Height 65 (inches read as cm).")


def menopause_before_menarche(v: Vignette) -> PerturbedCase:
    c = copy.deepcopy(v)
    c.risk_factors.menarche = 13
    c.risk_factors.menopause = 10
    c.vignette_id += "__menopause_before_menarche"
    return PerturbedCase(c, True, "hard", "menopause_age", "impossible_ordering",
                         "Menopause (10) before menarche (13).")


def missing_yob(v: Vignette) -> PerturbedCase:
    c = copy.deepcopy(v)
    c.target().yob = 0
    c.target().age = 0
    c.vignette_id += "__missing_yob"
    return PerturbedCase(c, True, "soft", "dob", "silent_degradation",
                         "Proband age/YOB removed -> calculation degrades.")


def clean_passthrough(v: Vignette) -> PerturbedCase:
    c = copy.deepcopy(v)
    c.vignette_id += "__clean"
    return PerturbedCase(c, False, None, None, "clean",
                         "Unmodified clean profile (false-positive control).")


PERTURBATIONS: List[Callable[[Vignette], PerturbedCase]] = [
    implausible_height, impossible_timeline, unit_mixup_height,
    menopause_before_menarche, missing_yob, clean_passthrough,
]


def build_exp2_dataset(clean_vignettes: List[Vignette]) -> List[PerturbedCase]:
    cases = []
    for v in clean_vignettes:
        for fn in PERTURBATIONS:
            cases.append(fn(v))
    return cases
