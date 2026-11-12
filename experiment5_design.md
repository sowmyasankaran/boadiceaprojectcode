# Experiment 5: Model Choice + Reliability (folded into Experiment 1)

## What this experiment answers

Two questions, kept separate:

1. **Which model is best** across the things that matter together — accuracy,
   run-to-run consistency, cost, and latency — for driving the agent?
2. **How reproducible** is each model? For the same patient, run repeatedly at
   a fixed temperature, do you get the same NICE category and a tight spread of
   lifetime-risk numbers, or does the answer wander between runs?

Per the design update, Experiment 5 is not a standalone experiment; it is the
*model-sweep dimension* of Experiment 1, run in two phases:

- **Breadth (accuracy).** Each model, once per profile, across a diverse
  profile set and across the four Experiment 1 groups (structured /
  unstructured / backend-API-with-skills / browser agent). Gives per-model and
  per-group accuracy versus gold.
- **Depth (reliability).** A smaller subset of profiles, each run *N* times per
  model at a fixed temperature. Gives within-model consistency and
  between-model differences.

## The one validity requirement that dominates everything

**Only the system-under-test's model may vary. The patient simulator's model
must be held fixed.**

The pipeline makes LLM calls on two sides: the *agent* (extraction,
question-generation, verification/fidelity, the yes/no confirmation classifier,
result summarization) and the *simulated patient* (`respondents.py` —
`_phrase_truthful_answer`, `SimulatedPatientAgent._phrase`, `_phrase_relative`).
The simulator is experimental scaffolding, not the thing being measured. If the
simulator's model changes when the agent's model changes, "GPT-5 is more
accurate than Claude" becomes uninterpretable — it could be that GPT-5-the-agent
paired with a differently-behaving simulator, not that the agent is better.

So: sweep the model on every **agent-side** call; pin every **simulator-side**
call to a single fixed reference model for the entire experiment. Record which
reference model the simulator used, as a fixed experimental constant.

## Current state and the gap to close

The multi-model seam already exists but is currently **extraction-only**.
`full_intake.run_conversational_intake_full(..., extraction_kwargs=...)` forwards
`{model, temperature, provider_override, extra_instructions}` to every
`extract_turn` / `extract_family_turn` in the intake — so the sweep already
covers extraction across all sections. It does **not** currently reach the other
agent-side calls: `agent.compose_question` / `next_question`,
`verification.run_fidelity`, `agent.is_affirmative`, and
`agent.summarize_results`, all of which call `llm` without a model override.

For a fair "which model runs our agent" comparison, those calls are part of the
agent and should run on the swept model too. Two clean ways to close this
(the choice depends on `llm.py`'s API, which this design does not assume):

- **(a) Broaden the run config.** Thread a `RunConfig(provider, model,
  temperature, seed)` into `compose_question`, `run_fidelity`, `is_affirmative`,
  and `summarize_results` (mirroring how `extraction_kwargs` already reaches
  extraction), and leave the simulator calls pinned to the reference model.
  Requires `llm.call_text` (used by question-gen/summary) to accept a model
  override the way `llm.call_with_tool` already does.
- **(b) Process-level default + explicit simulator override.** Give `llm.py` a
  settable active model/provider for the run, set it to the swept model at the
  top of each run, and have the simulator override to the fixed reference model
  on its own calls. Cleaner if `llm.py` supports a default-plus-per-call
  override; the risk to avoid is the simulator silently inheriting the swept
  model.

Whichever is chosen, extraction already honors the override, so the delta is
the four non-extraction call sites plus the simulator pin. Confirm against
`llm.py` before implementing — writing this against a guessed `llm` interface is
the exact failure mode to avoid.

## Model registry — how "more models" is added

Models are `(provider, model_id)` pairs in one list the harness loops over:

```
MODELS = [
    ("anthropic", "claude-sonnet-4-5"),
    ("openai",    "gpt-5"),
    ("google",    "gemini-..."),     # if llm.py wires a Google provider
    # add a row to run another model; nothing else changes
]
```

Adding a model is one line. Two caveats:

- **Provider support is bounded by `llm.py`.** It currently auto-detects
  OpenAI/Anthropic from the environment; a third provider (Google, or an
  open-model endpoint) only works if `llm.py` has a client for it. Adding a
  provider is a `llm.py` change, not a harness change.
- **Tool-use is required.** Extraction and the fidelity/confirmation steps use
  tool calls. A model without reliable tool-use will fail those calls — record
  the failure as a result (a real finding about that model), do not silently
  drop it.

## Fixed conditions (the control)

Held identical across every model and every run: the vignette set, the group
definitions, the simulator model, the temperature, the seed policy, and all
prompts. Only the agent-side model varies. Temperature is fixed for the depth
phase (e.g. 0.0, or a single small value used for every model); set a `seed`
where the provider supports it, and note that some providers ignore temperature
or don't honor seeds, which is itself a reproducibility finding.

## Metrics

### Breadth phase — per (model, group)

Reuse the Experiment 1 metric functions in `harness.py` so this is computed
identically to the rest of Experiment 1:

- **Field-level concordance** (bin-level) and **all-fields-correct rate**.
- **Bland–Altman** on lifetime risk versus gold (mean difference + 95% limits
  of agreement).
- **NICE category match rate** versus gold.
- **Cost per run** and **latency per run** (see "Cost and latency").

Report per model *and* per group, so a model-by-group interaction is visible
(e.g. a model that's fine on structured input but degrades on unstructured).

### Depth phase — per model, from *N* repeats of each subset profile

Let each subset profile be run `k` times on a given model. Define:

- **Category consistency.** Report two:
  - *All-agree rate* — fraction of profiles where all `k` repeats produced the
    same NICE category. This is the strict, headline number.
  - *Modal-agreement* — mean over profiles of (repeats matching that profile's
    modal category / `k`). Softer, shows how close the misses are.
- **SD of lifetime risk.** Per profile, the standard deviation of lifetime risk
  across its `k` repeats; report the **mean within-profile SD** across the
  subset (a run-to-run "wobble" in risk points). Optionally the max, to surface
  worst-case instability.
- **ICC (run-to-run reliability of the continuous output).** One-way
  random-effects ICC(1,1) across repeats of lifetime risk:
  `ICC = (MSB − MSW) / (MSB + (k − 1)·MSW)`, where MSB is the between-profile
  mean square and MSW the within-profile (between-repeat) mean square. High ICC
  means repeats cluster tightly per profile relative to the spread across
  profiles — i.e. the model reproduces a given patient's number reliably.
- **Field agreement across repeats.** Per profile, the fraction of scored
  fields identical across all `k` repeats (all-agree), and/or the mean pairwise
  field-match across the repeat pairs. Averaged across the subset.

### The Experiment 5 table (one row per model)

`Model | category consistency (all-agree) | mean within-profile SD of lifetime
risk | accuracy vs gold | ICC | field agreement | cost/run | latency`

The design doc's original four columns (category consistency, SD of lifetime
risk, accuracy vs gold, cost/run) are the core; ICC, field agreement, and
latency are the natural additions the depth phase produces for free.

## Cost and latency

Both are first-class here (the table lists cost/run, and it matters given the
budget). Capture per run:

- **Latency** — wall-clock around each full pipeline run (already trivial to
  record in the harness).
- **Cost** — token usage × per-model price. This needs `llm.py` to surface
  token counts per call (or a pricing table keyed by model to estimate from
  usage). If `llm.py` doesn't expose usage yet, adding a usage accumulator is
  the prerequisite; without it, cost/run stays estimated, not measured — flag
  which it is in the output.

## What the harness reuses

Experiment 5 is an orchestration layer, not new pipeline code:

- Conversational groups → `full_intake.run_conversational_intake_full` with the
  run config (breadth: one run per profile; depth: `k` runs per subset profile).
- Structured control → `full_intake.build_structured_intake` (no model — it's a
  floor, constant across the sweep; include it once, not per model).
- Browser-agent group → `exp1_browser_agent.py` (already model-parameterized via
  `--provider`/`--model`; run it once per model as its own sweep).
- Metrics → `harness.py` (breadth) plus the depth-phase functions above
  (`category_consistency`, `mean_within_profile_sd`, `icc_oneway`,
  `field_agreement` — small, self-contained, unit-testable offline).
- Ground truth / gold → `pedigree_data` (`full_ground_truth`,
  `load_gold_outputs`).

Suggested driver: `exp5_reliability.py` looping `MODELS × groups × profiles ×
repeats`, writing per-run JSON (with model, group, profile, repeat index,
outputs, cost, latency) and a per-model summary, with a `--phase {breadth,depth}`
flag, `--repeats k`, `--subset-tag` for the depth subset, and `--simulator-model`
for the pinned reference model.

## Confounds and caveats to state in the write-up

- **Simulator model fixed** — the dominant one, above.
- **Same profiles / temperature / seed policy / prompts across models** — the
  control; only the agent model varies.
- **Provider fairness** — model ids, context limits, and tool-use reliability
  differ; a model that fails tool-use fails the run (recorded, not dropped).
- **Latency is provider-infrastructure-dependent** — report it, but caveat that
  it reflects the endpoint's serving conditions, not only the model.
- **Depth subset is small and deliberately chosen** — pick diverse/representative
  profiles; the subset trades coverage for the many repeats reliability needs,
  so consistency numbers are conditional on that subset.
- **Cost measured vs estimated** — label which, depending on whether `llm.py`
  reports real token usage.

## Sequencing / dependencies

- The breadth phase runs today on the extraction-only seam (extraction is the
  dominant accuracy driver), but the *faithful* whole-agent sweep needs the four
  non-extraction call sites threaded (or the process-level default) — a `llm.py`
  check first.
- The depth-phase metric functions are self-contained math on outputs and can
  be written and unit-tested with no API budget and no `llm.py`.
- Everything here needs model access to *run*; it can be fully built now and run
  when credits/keys are available.
