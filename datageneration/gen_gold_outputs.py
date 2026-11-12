"""
feeds every vignette's CanRisk file to the API and
saves the returned risk results as the gold labels for Experiments 1 and 5.

Needs: CANRISK_USERNAME / CANRISK_PASSWORD env vars, and canrisk_test_call.py

Run:
    export CANRISK_USERNAME="..."; export CANRISK_PASSWORD="..."
    python3 gen_gold_outputs.py
"""

import json
import time

from vignette_schema import load_bank
from generate_vignettes import fetch_gold_output

BANK_PATH = "vignette_bank_v2.json"
OUT_PATH = "gold_outputs.json"
PAUSE_SECONDS = 1.0


def main():
    bank = load_bank(BANK_PATH)
    results = {}
    for i, v in enumerate(bank, 1):
        try:
            results[v.vignette_id] = fetch_gold_output(v)
            print(f"[{i}/{len(bank)}] {v.vignette_id}: ok")
        except Exception as e:
            print(f"[{i}/{len(bank)}] {v.vignette_id}: FAILED -> {e}")
            results[v.vignette_id] = {"error": str(e)}
        time.sleep(PAUSE_SECONDS)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    ok = sum(1 for r in results.values() if "error" not in r)
    print(f"\nsaved {OUT_PATH}: {ok}/{len(bank)} succeeded")


if __name__ == "__main__":
    main()
