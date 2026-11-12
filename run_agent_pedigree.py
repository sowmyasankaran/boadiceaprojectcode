
import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from conversions import PersonalLifestyleProfile, profile_to_header_lines
from datetime import date


from canrisk_test_call import get_auth_token, call_endpoint


def build_pedigree_from_profile(profile: PersonalLifestyleProfile) -> str:
    """
    Builds a full pedigree string using the agent-extracted profile for the
    proband's personal/lifestyle fields, keeping the same family structure
    (parents, sibling, maternal aunt with BRCA1+) as the original full test
    pedigree so the two are directly comparable.
    """
    result = profile_to_header_lines(profile)
    h = result["header_lines"]
    p = result["proband_overrides"]

    sex_code = p["sex"]  # "M" or "F"

    header = (
        "##CanRisk 2.0\n"
        f"##oc_use=N\n"     
        f"##mht_use=N\n"   
        f"##BMI={h['BMI']}\n"
        f"##alcohol={h['alcohol']}\n"
        f"##height={h['height']}\n"
        f"##TL=N\n"            
        f"##endo=N\n"     
        "##FamID\tName\tTarget\tIndivID\tFathID\tMothID\tSex\tMZtwin\tDead\tAge\tYob"
        "\tBC1\tBC2\tOC\tPRO\tPAN\tAshkn\tBRCA1\tBRCA2\tPALB2\tATM\tCHEK2\tBARD1"
        "\tRAD51D\tRAD51C\tBRIP1\tER:PR:HER2:CK14:CK56\n"
    )

    row_mgf = (
        "TESTFAM\t0\t0\tmgf01\t0\t0\tM\t0\t1\t78\t1945"
        "\t0\t0\t0\t0\t0\t0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0:0:0:0\n"
    )
    row_mgm = (
        "TESTFAM\t1\t0\tmgm01\t0\t0\tF\t0\t1\t80\t1943"
        "\t0\t0\t0\t0\t0\t0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0:0:0:0\n"
    )
    row_mother = (
        "TESTFAM\t2\t0\tmum01\tmgf01\tmgm01\tF\t0\t0\t62\t1961"
        "\t0\t0\t0\t0\t0\t0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0:0:0:0\n"
    )
    row_aunt = (
        "TESTFAM\t3\t0\taunt01\tmgf01\tmgm01\tF\t0\t0\t70\t1953"
        "\t48\t0\t0\t0\t0\t0\tS:P\tS:N\tS:N\tS:N\tS:N\tS:N\tS:N\tS:N\tS:N\t0:0:0:0:0\n"
    )
    row_father = (
        "TESTFAM\t4\t0\tdad01\t0\t0\tM\t0\t0\t64\t1959"
        "\t0\t0\t0\t0\t0\t0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0:0:0:0\n"
    )
    row_sib = (
        "TESTFAM\t5\t0\tsib01\tdad01\tmum01\tF\t0\t0\t32\t1991"
        "\t0\t0\t0\t0\t0\t0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0:0:0:0\n"
    )
    row_proband = (
        f"TESTFAM\t6\t1\tprob1\tdad01\tmum01\t{sex_code}\t0\t0\t{p['age']}\t{p['yob']}"
        "\t0\t0\t0\t0\t0\t0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0:0:0:0\n"
    )

    return (
        header
        + row_mgf + row_mgm + row_mother + row_aunt + row_father + row_sib + row_proband
    )


def run_agent_driven_test(profile: PersonalLifestyleProfile):
    """Same auth + API round-trip as canrisk_test_call.py, but pedigree
    built from the agent's extracted profile instead of hardcoded values."""
    username = os.environ.get("CANRISK_USERNAME")
    password = os.environ.get("CANRISK_PASSWORD")
    if not username or not password:
        print("Set CANRISK_USERNAME / CANRISK_PASSWORD env vars first.")
        return

    result = profile_to_header_lines(profile)
    if result["unresolved"]["ethnic_group"]:
        print(
            f"Note: ethnic_group ('{result['unresolved']['ethnic_group']}') was "
            "collected but not yet submitted -- header tag syntax unconfirmed."
        )

    pedigree = build_pedigree_from_profile(profile)
    print("Generated pedigree:\n", pedigree)

    token = get_auth_token(username, password)
    boadicea_result = call_endpoint("boadicea", token, pedigree)
    print("\nBOADICEA result:")
    print(json.dumps(boadicea_result, indent=2))


if __name__ == "__main__":
    example_profile = PersonalLifestyleProfile(
        sex="F",
        country="UK",
        dob=date(1990, 3, 15),
        height_cm=165,
        weight_kg=64,
        drinks=[{"type": "wine", "frequency": "weekly", "amount": 2}],
    )
    run_agent_driven_test(example_profile)
