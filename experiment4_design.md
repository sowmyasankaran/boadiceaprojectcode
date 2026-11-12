# Experiment 4: Robustness of Extraction (Revised Design)

## Motivation

Real conversations don't deliver one clean fact per turn. Patients volunteer
multiple facts at once, give imprecise or range-bound answers, misstate and
then correct information, and generally don't behave like a form-filling
wizard. This experiment isolates the language-understanding / extraction
layer of the Risk Agent and asks: how much does accuracy degrade as input
moves from tidy and structured toward naturalistic and messy? And is the
degradation approximately linear, or does it break down sharply once
specific behaviors (e.g., corrections) are introduced?

This is a simplification of the original five-condition design (multi-field
turns, deferred/skipped fields, ambiguous answers, out-of-order answers,
mid-conversation correction) into three discrete agent conditions plus a
structured control, chosen for tractability. Two of the five original
conditions (multi-field turns, ambiguous/range answers) are preserved by
folding them into Agent C's behavior set rather than running as separate
conditions. Deferred/skipped fields and out-of-order answers are dropped
from this experiment's scope.

## Design Overview

Three simulated-patient agent conditions plus a structured-input control,
each interacting with the same Risk Agent pipeline and evaluated against the
same set of synthetic ground-truth patient profiles. By default the run uses
the 6 vignettes tagged `exp4`; `--all-vignettes` runs the full 27-vignette
bank. The 27-vignette run is the reported one — with 4.5x the data, several
single-vignette ceilings (e.g. Agent A / Agent B all-correct, ambiguous
accuracy, and especially the ambiguous flag rate) came down to their real,
more trustworthy values, and two Agent C rows now clearly miss their stated
targets rather than merely trailing.

### Agent A — One-shot clean input (upper-bound control)

- The simulated patient provides the entire profile in a single well-formed
  turn per section (and the whole family history as narrative), with no
  ambiguity, no errors, and no extraction burden left to conversational
  turn-taking.
- Purpose: establishes the ceiling — how well the pipeline performs when
  extraction has to happen but dialogue management does not.
- This is the structured/near-structured input control, distinct from the
  literal structured-form control (see "Structured input control" below).

### Agent B — Clean step-by-step (baseline)

- The simulated patient provides no information upfront and answers only
  what the Risk Agent asks, one field at a time, truthfully and
  unambiguously, in the order requested.
- Purpose: isolates the cost of conversational turn-taking alone, with no
  additional noise from patient-side messiness. This is the clean
  conversational baseline that Agent C is compared against, and the
  reference point for every "vs. baseline" delta in the results.

### Agent C — Adversarial/naturalistic step-by-step

- Same setup as Agent B (no upfront information, fully agent-driven question
  sequence), but the simulated patient's behavior is sampled per field from a
  defined behavior set rather than fixed to "truthful direct answer."
  Behaviors:

  1. **Truthful direct answer** — baseline behavior, same as Agent B.
  2. **Wrong-then-corrected** — patient gives an incorrect value, then later
     (in a subsequent turn, ideally while answering a *different* question)
     corrects it.
  3. **Multi-fact volunteering** — patient answers the asked question but
     also volunteers one or more additional, unrequested fields in the same
     turn.
  4. **Ambiguous/range answer** — patient gives an imprecise value instead
     of a precise one (e.g., "around 40," "somewhere in my 40s," "a few
     years ago") rather than a specific number or date.

- All four behaviors are sampled from the same ground-truth-holding
  simulator (see "Simulated Patient Agent" below); no separate
  infrastructure is needed per behavior.

**Scope of behavior sampling.** Agent C's adversarial sampling applies to
every *flat* section (personal/lifestyle, alcohol, women's health, children,
breast screening, medical history, PRS) — the same `SimulatedPatientAgent`
drives all of them under `condition == "agent_c"`. Family-history/pedigree
disclosure stays truthful and stepwise across all conditions except Agent A's
one-shot dump. The four behaviors are specified as per-*field* behaviors in a
flat-form sense and have no literal analog for "get a relative's diagnosis
age wrong" without conflating this experiment with Experiment 2's order-error
condition, which already covers relative-identity errors separately.

### Structured input control

- In addition to Agents A/B/C, the same ground-truth profiles are run through
  literal structured entry (`build_structured_intake`), bypassing free-text
  extraction entirely.
- Purpose: establishes what accuracy loss, if any, is attributable to
  free-text understanding at all, independent of conversational turn-taking.
  This is the true floor for "extraction cost."

## Simulated Patient Agent (Sharding Methodology)

Following Laban et al. (2025), a second LLM ("patient simulator") holds the
complete ground-truth patient profile and discloses it incrementally across
turns rather than all at once, governed by the behavior set assigned to its
condition.

- The simulator paces disclosure and selects a response type per field:
  direct answer, hedging/ambiguous answer, clarification, or (for Agent C
  only) an intentionally wrong answer slated for later correction. The
  field-by-field behavior chosen is recorded in `behavior_log` for the
  per-behavior accuracy breakdown.
- Only the "flat" sharding behaviors above are implemented; recap, snowball,
  and gradual multi-shard granularity sweeps from the source paper are out of
  scope for this experiment.
- Wrong answers used in the wrong-then-corrected behavior are drawn from the
  synthetic dataset's pre-labeled incorrect-value variants, so ground truth
  for both the wrong answer and the eventual correct answer is known in
  advance and scoring doesn't depend on the simulator's improvisation.

**Seed scoping.** The Agent C seed is scoped per vignette *and* per repeat
(`seed = 1000 + vignette_index * 97 + repeat`), not a single fixed
per-repeat seed. An earlier version reseeded identically for each repeat, so
every vignette replayed the same short random prefix and Agent C's behavior
distribution collapsed to a near-constant sequence; scoping the seed per
vignette fixed this and is why the per-behavior counts are now meaningful.

## Scoring and Reachability

- Fields listed in `UNSCORED_FIELDS` are excluded from scoring under every
  condition.
- For Agent B and Agent C *only*, fields that are `required=False` with no
  conditional gate that ever forces them (e.g. `ethnic_group`,
  `contralateral_breast_cancer`) are structurally unreachable — those
  conditions only ask `missing_required()` fields — so scoring them as
  "wrong" would penalize extraction for something the agent was never given a
  chance to ask. Reachable fields are computed by
  `reachable_fields_for_vignette`. Structured and Agent A always state every
  field regardless of required-ness and are scored on the full set.

## Verification

The pipeline runs verification after each completed section, matching the
deployed system rather than an artificially stripped-down version. Two
independent checks (see `verification.py`):

- **Fidelity (LLM):** a fresh whole-transcript re-read that confirms each
  recorded value matches what the patient actually said, returning per-field
  mismatches and untraceable/unconfirmed values. Because it re-reads the full
  conversation rather than the latest turn, it can catch accumulation errors
  and corrections that per-turn extraction missed.
- **Plausibility (deterministic):** are the recorded values physically /
  clinically coherent, individually and cross-field (see `plausibility.py`)?
  Hard bounds block submission; soft bounds surface a confirmation.

Runs default to verification **on**; `--no-verification` reproduces the
pre-fix numbers for comparison. Verification matters most for
wrong-then-corrected: without it, per-turn extraction is the only chance to
catch a stated correction.

## Controls

| Control | Purpose |
|---|---|
| Structured input (form entry) | Floor: cost of free-text understanding at all |
| Agent A (one-shot clean) | Ceiling: extraction without turn-taking burden |
| Agent B (clean step-by-step) | Baseline: turn-taking cost alone, no patient-side noise |

Agent C is compared against Agent B to isolate the marginal cost of
naturalistic patient behavior on top of turn-taking, and against the
structured-input floor for the full best-case-to-worst-case spread.

## Metrics

- **Per-field accuracy (bin-level)** — computed under each condition
  (structured, A, B, C), and additionally broken down *by behavior type*
  within Agent C (truthful / corrected / multi-fact / ambiguous), so failure
  modes within Agent C are distinguishable rather than pooled.
- **All-fields-correct rate** — proportion of profiles where every scored
  field is correct, per condition.
- **Correction rate** — for wrong-then-corrected, the fraction where the
  Risk Agent's *final recorded* value reflects the correction rather than the
  original wrong answer. This is end-to-end (did the correction actually land
  in the record), not merely whether it was detected.
- **Ambiguous flag rate** — of the fields Agent C answered ambiguously, the
  fraction where extraction recognized the ambiguity (via `ambiguous_fields`
  → a follow-up) at some point, regardless of whether the final value was
  correct. Plain accuracy can't distinguish "silently guessed right" from
  "correctly asked for precision, then recorded the precise answer"; this
  metric can.
- **Ambiguous silent-guess-correct rate** — of the ambiguous fields that were
  never flagged, the fraction that still ended up correct (the system guessed
  the exact right point value from a hedge) versus wrong (silently guessed and
  missed).
- **vs-baseline deltas (pp)** — every condition's two accuracy metrics
  expressed as percentage-point deltas against Agent B, computed post-hoc
  after all primary numbers are in.

Current numbers live in `results/exp4_summary.json`. As of the last full
27-vignette run, two Agent C rows miss their stated targets: mid-conversation
corrections (correction rate, target >95%) and the ambiguous flag rate
(target "higher"). Those two targets motivate the robustness work below.

## Robustness Improvements (implemented, pending validation)

The correction-rate shortfall was traced through the real pipeline rather than
patched around. The finding: the *deployed* agent (`agent.run`) already runs
three backstops — per-turn extraction plus a dedicated correction scan; a
VERIFY pass whose fidelity re-read is applied as a correction; and a teach-back
correction loop. The Experiment 4 harness was under-modeling that pipeline
(two of the three backstops were effectively inert), so the correction row
measured well below what the shipped agent achieves. The fixes below bring the
experiment — and, for two of them, the deployed pipeline — into line. All four
are implemented but **not yet validated**: no API budget was available to
rerun at the time of writing, so the numbers elsewhere in this doc are
unchanged pending that rerun.

**Already shipped, awaiting the rerun (extraction layer):**

- *Dedicated correction-scan pass.* A second, narrow LLM call per turn whose
  only job is to detect a correction to a previously recorded field, split off
  from the main extraction call so it isn't competing with answering the
  current question. Skips the call when nothing is recorded yet.
- *Strengthened ambiguity flagging.* An explicit trigger-word list ("around,"
  "roughly," "maybe," "somewhere between," ranges like "40–45") in both the
  system prompt and the extraction tool schema, framed as a must-flag rather
  than a case-by-case judgment.

**Newly implemented in this pass:**

1. **Simulator correction-delivery fix (experiment bug).**
   `wrong_then_corrected` stated the wrong value and deferred its correction to
   a *later* gather turn, but the gather loop stopped the moment required
   fields were all present — so a correction whose wrong value completed the
   section was never uttered, and for single-field sections it dropped every
   time. The simulator now reports a pending correction (`pending_disclosure`)
   and the gather loop stays open one more turn to voice it. This was a large,
   purely artefactual depressor of the correction rate: nothing downstream can
   catch a correction that was never said.
2. **Teach-back as a real third backstop.** The harness previously discarded
   the teach-back reply (an explicit no-op) and the base respondent
   rubber-stamped the read-back. Teach-back now mirrors the deployed agent's
   `correct_at_teach_back`: the ground-truth-holding respondent compares the
   recorded values against the truth (using the *same* tolerance-aware matcher
   as scoring, so a within-tolerance value is never spuriously "corrected"),
   voices any discrepancy, and the reply is re-extracted and re-verified. It
   applies to every conversational condition, exactly as deployment gives every
   patient a teach-back — so Agent A and Agent B all-correct rates are expected
   to rise toward their true deployed values too. This is a fidelity fix; the
   A/B/C/structured conditions themselves are unchanged. `--no-verification`
   disables this backstop to reproduce the pre-fix numbers.
3. **Fidelity supersession (deployed-pipeline improvement).** The fidelity
   re-read was already applied as a correction, but its prompt never told the
   model that a *later* correction supersedes an earlier statement — so when
   the wrong value had also been literally uttered, a superseded value still
   sitting in the transcript could pass as "faithful." The prompt now treats
   the latest stated value as authoritative. Unlike the first two, this changes
   the shipped agent, not just the experiment.

4. **Context-contamination fix (Agent C truthful-field underperformance).**
   The ablation was run: `--ablate-corrections` (Agent C with
   `wrong_then_corrected` disabled) brought `accuracy__truthful`,
   `accuracy__multi_fact`, and `accuracy__ambiguous` all up to a clean 100%,
   matching Agent B/structured. That confirms the diagnosis rather than just
   suggesting it — an uncorrected wrong value sitting in the "already
   recorded" extraction context was measurably degrading extraction of
   *other*, unrelated fields in the same conversation, not just the field
   that was actually wrong.

   The fix (now implemented, in `schema.py` + `agent.py`): every recorded
   field carries a provisional-vs-confirmed status (`Section.unconfirmed`). A
   field goes provisional the moment it's (re)set and is promoted to
   confirmed only once it survives one full subsequent turn without being
   re-touched by extraction or the dedicated correction scan
   (`apply_extraction` does the promotion, since it's the one place old and
   new extraction results are both available to compare). `extract_turn`'s
   primary extraction call now shows only `confirmed_values()` as "already
   recorded" context — a value still awaiting its deferred correction can no
   longer condition how the model reads other fields. The correction-scan
   pass is deliberately exempt and still sees the full value set, since its
   entire job is catching a restatement of something on record, confirmed or
   not — most corrections arrive before the original value has had a chance
   to be promoted, so hiding provisional fields from that pass specifically
   would make it unable to do its job.

   Verified with a standalone state-machine check (no LLM call): a value set
   in turn *N* stays out of `confirmed_values()` until turn *N+1* completes
   untouched, and a correction in a later turn correctly resets it to
   provisional again. Like the items above, the effect on Agent C's actual
   accuracy numbers is pending the next full rerun.

**Instrumentation.** Each `wrong_then_corrected` correction is now attributed
to the backstop that caught it — GATHER (extraction + scan), VERIFY (fidelity),
or TEACH-BACK — via per-stage value snapshots on the run trace
(`correction_provenance` in the summary). The next run therefore shows which
backstop each point of correction-rate improvement came from, rather than only
a change in the aggregate.

## Holistic Audit Pass (New Layer)

A third cross-checking layer, added after this doc's numbers were last
generated, so it's not yet reflected in the Metrics section below.

Two cross-checking layers already existed: per-section fidelity/plausibility
(`verification.py`, scoped to one section's own transcript and values), and
`run_cross_section_checks` (`intake.py`, deterministic — seven hand-written
rules spanning sections, e.g. age-at-first-birth vs. current age, a
relative's diagnosis age vs. their own age, a relative-identity swap
heuristic). Both are precise but each only catches what it was specifically
built to look for.

The new layer (`audit.py`) is an LLM re-read of the **entire** intake
transcript as one continuous story, run once immediately before submission —
after every section's own verification and after `run_cross_section_checks`
has already run. It's scoped to catch two things neither existing layer
can: narrative contradictions that aren't about a single numeric field (e.g.
Children records "no children" but a later Family History turn mentions "my
daughter's asthma" in passing), and whole-story clinical incoherence, where
every individual value is independently plausible and faithful but the
*combination* isn't (e.g. a bilateral mastectomy on record alongside an
active native-breast screening schedule). To avoid re-flagging the same
things `run_cross_section_checks` already caught, that check's own output is
passed in as `existing_warnings` with an explicit instruction not to
re-report anything on it.

Wired into the deployed pipeline in `intake.py`'s `run_full_intake`, right
after the final combined teach-back is confirmed. A blocking finding stops
submission with no "continue anyway" option, matching every other hard-block
in the pipeline; a soft finding gets the same "continue anyway" prompt
`run_cross_section_checks`' own warnings already use.

**Harness fix required to make this testable.** The Experiment 4 harness gave
each section its own fresh transcript, unlike the deployed pipeline's single
shared transcript spanning every section plus family history — so there was
no "whole conversation" object to hand the audit. Fixed: `harness.py`'s
`run_conversational_intake` now accepts a `shared_transcript` to append to,
and `full_intake.py` threads one transcript through the whole intake,
generating a real per-section opening question via `compose_question` for
every section after the first (the fixed opening line used previously would
have been visibly wrong once shared — e.g. a later section re-asking "what's
your sex and country"). `run_conversational_intake_full` takes an opt-in
`check_holistic_audit` flag, off by default so existing Exp1/Exp4 runs don't
pay for an LLM call they didn't ask for.

**Not yet done:** the audit pass has no scored metric of its own yet — it
runs and returns a verdict, but nothing in the Metrics section below counts
true/false positives for it. Whether to add that (and how — the synthetic
vignettes don't currently have any *intentional* cross-section/story-level
contradictions to detect) is an open question for a future revision, not
resolved here.

## What Was Dropped From the Original Five-Condition Design

- **Deferred/skipped-then-re-requested fields** — tests the Risk Agent's own
  follow-up/completeness logic rather than extraction fidelity per se; no
  natural home in the A/B/C structure as scoped.
- **Out-of-order answers** — patient answers a later question before the one
  currently asked; also dropped for scope.

If time permits after the core runs, these could be added as a light-touch
extension to Agent C's behavior set using the same simulator infrastructure,
rather than as new agent conditions.

## Open Questions to Resolve

1. Behavior sampling weights for Agent C (whether truthful should remain the
   plurality to keep transcripts realistic) are set in
   `respondents.SimulatedPatientAgent`; seed scoping is already resolved.
   Confirm the exact weights there.
2. Whether wrong-then-corrected and multi-fact volunteering can co-occur
   within the same field/turn, or are sampled mutually exclusively per field,
   is governed by `SimulatedPatientAgent`'s sampling — confirm.
3. Confirm the synthetic dataset's pre-labeled incorrect-value variants cover
   all field types needed for the wrong-answer behavior, not just a subset.
4. Whether the holistic audit pass gets its own scored metric (precision/
   recall against intentionally-planted cross-section contradictions), and if
   so, whether that requires a new labeled variant set — the current 27
   vignettes have no such contradictions to detect. Not resolved here.