"""
Turns each vignette into a simulated patient's side of
a conversation, one per style, using OpenAI. Each is paired with the vignette's
ground-truth fields (the label) for Experiment-4 scoring, and the rendered text
is the free-text input for Experiments 1 and 3.

Needs: OPENAI_API_KEY env var and the openai package (pip install openai).

Run:
    export OPENAI_API_KEY="..."
    python3 gen_conversations.py
"""

import os
import json

from openai import OpenAI

from vignette_schema import load_bank
from generate_vignettes import render_conversation_prompt, STYLES

BANK_PATH = "vignette_bank_v2.json"
OUT_PATH = "conversations.json"
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
STYLES_TO_RUN = STYLES            # e.g. ["clean", "multi_fact"] for a cheap pass
MAX_VIGNETTES = None              # e.g. 3 to test before running the whole bank


def main():
    client = OpenAI()
    bank = load_bank(BANK_PATH)
    if MAX_VIGNETTES:
        bank = bank[:MAX_VIGNETTES]

    dataset = []
    for i, v in enumerate(bank, 1):
        for style in STYLES_TO_RUN:
            prompt = render_conversation_prompt(v, style)
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                )
                text = resp.choices[0].message.content
            except Exception as e:
                text = None
                print(f"    {v.vignette_id}/{style}: FAILED -> {e}")
            dataset.append({
                "vignette_id": v.vignette_id,
                "style": style,
                "conversation": text,                       # the input
                "expected_fields": prompt["expected_fields"],  # the label
            })
        print(f"[{i}/{len(bank)}] {v.vignette_id}: {len(STYLES_TO_RUN)} styles")

    with open(OUT_PATH, "w") as f:
        json.dump(dataset, f, indent=2)
    ok = sum(1 for d in dataset if d["conversation"])
    print(f"\nsaved {OUT_PATH}: {ok}/{len(dataset)} conversations rendered")


if __name__ == "__main__":
    main()
