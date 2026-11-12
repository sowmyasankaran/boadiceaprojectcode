
import os
import sys
import json
import requests

BASE_URL = "https://canrisk.org"


def get_auth_token(username: str, password: str) -> str:
    """POST credentials to /auth-token/ and return the token string."""
    resp = requests.post(
        f"{BASE_URL}/auth-token/",
        json={"username": username, "password": password},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    # The API returns the token under a "token" key.
    token = data.get("token")
    if not token:
        raise RuntimeError(f"No token in response: {data}")
    return token


def build_full_pedigree() -> str:
    header = (
        "##CanRisk 2.0\n"
        "##menarche=13\n"
        "##parity=1\n"
        "##first_live_birth=28\n"
        "##oc_use=N\n"
        "##mht_use=N\n"
        "##BMI=23.5\n"
        "##alcohol=5\n"
        "##height=165\n"
        "##TL=N\n"
        "##endo=N\n"
        "##PRS_BC=alpha=0.45, zscore=0.2\n"
        "##PRS_OC=alpha=0.35, zscore=-0.1\n"
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
    # mother -- unaffected, untested
    row_mother = (
        "TESTFAM\t2\t0\tmum01\tmgf01\tmgm01\tF\t0\t0\t62\t1961"
        "\t0\t0\t0\t0\t0\t0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0:0:0:0\n"
    )
    # maternal aunt -- breast cancer diagnosed age 48, confirmed BRCA1 positive
    # (S:P = sequencing test, positive result)
    row_aunt = (
        "TESTFAM\t3\t0\taunt01\tmgf01\tmgm01\tF\t0\t0\t70\t1953"
        "\t48\t0\t0\t0\t0\t0\tS:P\tS:N\tS:N\tS:N\tS:N\tS:N\tS:N\tS:N\tS:N\t0:0:0:0:0\n"
    )
    # father -- unaffected, untested
    row_father = (
        "TESTFAM\t4\t0\tdad01\t0\t0\tM\t0\t0\t64\t1959"
        "\t0\t0\t0\t0\t0\t0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0:0:0:0\n"
    )
    # full sibling -- unaffected, untested
    row_sib = (
        "TESTFAM\t5\t0\tsib01\tdad01\tmum01\tF\t0\t0\t32\t1991"
        "\t0\t0\t0\t0\t0\t0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0:0:0:0\n"
    )
    # proband, the target of the calculation, age 35, currently unaffected,
    # not yet tested herself (untested is the realistic case for a risk
    # calculation prior to any genetic testing)
    row_proband = (
        "TESTFAM\t6\t1\tprob1\tdad01\tmum01\tF\t0\t0\t35\t1990"
        "\t0\t0\t0\t0\t0\t0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0\t0:0:0:0:0\n"
    )

    return (
        header
        + row_mgf
        + row_mgm
        + row_mother
        + row_aunt
        + row_father
        + row_sib
        + row_proband
    )


def call_endpoint(endpoint: str, token: str, pedigree_data: str, user_id: str = "test_user") -> dict:

    resp = requests.post(
        f"{BASE_URL}/{endpoint}/",
        json={
            "mut_freq": "UK",
            "cancer_rates": "UK",
            "user_id": user_id,
            "pedigree_data": pedigree_data,
        },
        headers={"Authorization": f"Token {token}"},
        timeout=60,
    )
    print(f"[{endpoint}] status code: {resp.status_code}")
    try:
        return resp.json()
    except ValueError:
        print(f"[{endpoint}] non-JSON response body:")
        print(resp.text)
        resp.raise_for_status()
        return {}


def main():
    username = os.environ.get("CANRISK_USERNAME")
    password = os.environ.get("CANRISK_PASSWORD")

    if len(sys.argv) == 3:
        username, password = sys.argv[1], sys.argv[2]

    if not username or not password:
        print(
            "Missing credentials. Set CANRISK_USERNAME / CANRISK_PASSWORD env vars, "
            "or run: python canrisk_test_call.py <username> <password>"
        )
        sys.exit(1)

    print("Requesting auth token...")
    token = get_auth_token(username, password)
    print("Got token.\n")

    pedigree = build_full_pedigree()

    print("Submitting test pedigree to /boadicea/ (breast cancer model)...")
    boadicea_result = call_endpoint("boadicea", token, pedigree)
    print("\nBOADICEA result:")
    print(json.dumps(boadicea_result, indent=2))

    print("\nSubmitting test pedigree to /ovarian/ (ovarian cancer model)...")
    ovarian_result = call_endpoint("ovarian", token, pedigree)
    print("\nOvarian model result:")
    print(json.dumps(ovarian_result, indent=2))


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()


def call_vcf2prs(token: str, reference_filename: str, sample_name: str,
                  vcf_path: str) -> dict:

    with open(vcf_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/vcf2prs/",
            data={"bc_prs_reference_file": reference_filename,
                  "sample_name": sample_name},
            files={"vcf_file": f},
            headers={"Authorization": f"Token {token}"},
            timeout=120,
        )
    if resp.status_code != 200:
        try:
            return {"error": f"HTTP {resp.status_code}", "detail": resp.json()}
        except ValueError:
            return {"error": f"HTTP {resp.status_code}", "detail": resp.text}
    return resp.json()