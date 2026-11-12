"""
generates the Experiment 2 perturbed dataset and saves it
to disk.

Run:
    python3 gen_exp2_data.py
"""

import json

from vignette_schema import load_bank
from perturbations import build_exp2_dataset, PerturbedCase

BANK_PATH = "vignette_bank_v2.json"
OUT_PATH = "exp2_cases.json"


def case_to_dict(case: PerturbedCase) -> dict:
    """Flatten a PerturbedCase into a plain dict: the perturbed vignette's
    CanRisk file (what you'd feed the agent), plus the expected verdict
    (what the verification layer should say) and provenance."""
    return {
        "case_id": case.vignette.vignette_id,
        "source_bank": BANK_PATH,
        "kind": case.kind,
        "description": case.description,
        "expected": {
            "expect_flag": case.expect_flag,
            "severity": case.severity,
            "field": case.field,
        },
        "perturbed_vignette": case.vignette.to_dict(),
    }


def main():
    bank = load_bank(BANK_PATH)
    cases = build_exp2_dataset(bank)

    data = [case_to_dict(c) for c in cases]
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"saved {OUT_PATH}: {len(data)} perturbed cases "
          f"({len(bank)} vignettes x 6 perturbations)")

    from collections import Counter
    print("by kind:", dict(Counter(c.kind for c in cases)))
    print("by severity:", dict(Counter(c.severity for c in cases)))
    print(f"\nexample case (index 0): {cases[0].kind}")
    print(f"  expected: flag={cases[0].expect_flag}, "
          f"severity={cases[0].severity}, field={cases[0].field}")


if __name__ == "__main__":
    main()
