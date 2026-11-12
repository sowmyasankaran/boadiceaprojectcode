# Experiment harness (real dataset)

Runs Experiments 1, 2, and 4 against your actual pipeline (`agent.py`,
`intake.py`, `schema.py`, `clinical_sections.py`, `pedigree.py`,
`verification.py`, `plausibility.py`, `canrisk_format.py`, `llm.py`) and the
real synthetic data (`vignette_bank_v2.json`, `exp2_cases.json`,
`gold_outputs.json`). Experiment 3 is intentionally not implemented here --
run it with the chat-based human-experimentation design you already have.

## Setup

```
pip install -r requirements.txt
export OPENROUTER_API_KEY=...
export CANRISK_USERNAME=...         # only needed for --submit / live BOADICEA calls
export CANRISK_PASSWORD=...
```

The three data files ship in this folder as-is. Override with
`VIGNETTE_BANK_PATH`, `EXP2_CASES_PATH`, `GOLD_OUTPUTS_PATH` if you move them.

## Files

- `pedigree_data.py` -- loaders + the ground-truth layer. Maps each of the
  27 full pedigrees (family graph + risk factors) onto every real section
  (`PersonalLifestyleSection`, `AlcoholSection`, `WomensHealthSection`,
  `ChildrenSection`, `BreastScreeningSection`, `MedicalHistorySection`,
  `PRSSection`) plus a reconstructed `Pedigree`, including a relationship
  inferrer (father_id/mother_id graph -> `RELATIONSHIPS` vocabulary like
  "maternal_aunt"). Fields the dataset has no ground truth for (a few
  Women's Health/Children follow-ups, oophorectomy/mastectomy) get a
  clearly-flagged filler and are listed in `UNSCORED_FIELDS`, excluded from
  every accuracy metric.
- `full_intake.py` -- non-interactive version of `intake.py`'s real flow
  (`gather_until_complete`/`extract_turn`/`extract_family_turn`/
  `apply_family_extraction`/`submit_intake`), driven by ground truth instead
  of `input()`. `build_structured_intake()` (no LLM) and
  `run_conversational_intake_full()` (real extraction, per-condition
  respondents).
- `respondents.py` -- `OneShotRespondent` (Agent A), `ScriptedTruthfulRespondent`
  (Agent B), `SimulatedPatientAgent` (Agent C, sharding + behavior sampling),
  `StructuredInputRespondent`. Generic over any `Section` subclass.
- `harness.py` -- shared non-interactive gather/verify loop (mirrors
  `agent.py`'s `gather_until_complete`/`verify_until_clean`) plus metrics
  (field concordance, Bland-Altman, NICE match, per-field accuracy) and
  CSV/JSON writers.
- `exp1_validity.py` -- 6-group design + Exp5's reliability/depth phase,
  scored against `gold_outputs.json` (already computed, no live call needed
  for the gold side).
- `exp2_verification.py` -- fully deterministic core: runs
  `plausibility.py` directly against all 162 cases in `exp2_cases.json` and
  scores against each case's labeled `expected` (no LLM needed). Plus two
  small live-LLM addenda for order-error and fidelity-error, which aren't in
  the static dataset (see docstring).
- `exp4_extraction_robustness.py` -- the three-agent design, scored across
  every flat-section field (not just personal/lifestyle).
- `run_all.py` -- runs 1, 2, 4 in sequence.

## Validated end-to-end (no API keys needed for this part)

- All 27 vignettes' ground truth reconstructs into a **locally valid**
  CanRisk pedigree file (`canrisk_format.validate_pedigree_file` returns no
  problems) via the structured (no-LLM) path.
- Relationship inference matches the known family structures exactly
  (checked against `seed_brca1_young_breast__var00`'s 9-person family).
- Exp2's core 162-case check runs with **zero LLM calls** and reproduces
  the expected pattern: 100% sensitivity + exact field/severity match on
  `implausible_value`, `unit_mixup`, `impossible_ordering`,
  `impossible_timeline`; 0% false positives on `clean`; **0% detection on
  `silent_degradation`**, which is the real finding (no mechanism currently
  checks a missing-but-optional field), not a bug in the test.

## What still needs a live LLM / CanRisk credentials to actually run

Exp1's conversational groups (2, 4, 5, 6), Exp1's depth/reliability phase,
and all of Exp4 except the `structured` condition need `ANTHROPIC_API_KEY`
or `OPENAI_API_KEY`. Submitting for real (vs. extraction-only scoring) also
needs `CANRISK_USERNAME`/`CANRISK_PASSWORD` and `--submit`.

## Known gaps / scoping decisions (read before citing results)

- **Exp1 group 3 (browser agent)** -- `BrowserAgentRunner.run()` raises
  `NotImplementedError`. Needs real browser automation against the live
  CanRisk web form.
- **Exp1 group 4 (backend API skill variants)** -- `agent.extract_turn()`
  has no system-prompt override hook, so both variants currently run
  identically (tagged with a `note` saying so).
- **Exp1 groups 5/6 (baseline minimal instruction, LLM-only prediction)** --
  intentionally return raw, ungraded LLM text; scoring needs a judge/parse
  step you design deliberately.
- **Exp2's order-error check** is a live, small-sample addendum (not in the
  162-case file), and by construction always reports the swap as
  undetectable -- there's currently no mechanism anywhere that checks
  whether a diagnosis age could belong to a different relative.
- **Exp4's Agent C** applies adversarial behavior sampling to every flat
  section, not just personal/lifestyle; family-history disclosure stays
  truthful across every condition except Agent A's one-shot dump (see
  `exp4_extraction_robustness.py`'s docstring for the reasoning).
- A handful of Women's Health/Children fields (`oc_years`,
  `oc_last_two_years`, `hrt_years`, `hrt_last_five_years`, `hrt_type`) and
  Medical History's `oophorectomy`/`mastectomy` have no ground truth in the
  dataset -- filled with a fixed default only so the section can complete,
  and excluded from every accuracy metric (`pedigree_data.UNSCORED_FIELDS`).
