# Experiment 3: Real-User Comparison (Agent vs. Manual CanRisk Form)

## Motivation

Experiments 1, 2, and 4 all establish that the pipeline extracts and verifies
data correctly *given a conversation happens*. None of them touch the actual
applied claim of the project: that a person who could not complete BOADICEA's
own structured form independently can do this through natural conversation,
without sacrificing accuracy. Experiment 3 is the one experiment that puts a
real, naive human in the loop, comparing the deployed conversational agent
against the real CanRisk web form on the same task.

This is deliberately scoped as a small, directional comparison, not a
powered study -- see "Sample Size and Analysis Plan" below. Its job is to
produce real signal on whether the agent is actually easier for a human,
and whether that ease costs anything in accuracy -- not to produce a
statistically validated effect size.

## Design

**Within-subjects, two conditions, both real systems:**

- **Agent condition:** the conversational web interface (`web_backend.py` +
  `web_frontend.html`), talking to the real, unmodified `intake.py` pipeline.
- **Form condition:** the real CanRisk web form at canrisk.org -- not a mock,
  not a paper form. This is a deliberate choice: comparing against a
  simulated form would understate the form condition's real friction (actual
  CanRisk has real field validation, real navigation, real UI quirks) and
  overstate the agent's advantage.

Each participant completes **both** conditions, entering a **different**
fictional vignette in each (see "Materials" below) -- not the same vignette
twice. Using the same vignette in both conditions would let whichever
condition comes second benefit from the participant having already
half-memorized the answers, which would confound condition with order rather
than measuring the condition itself.

## Materials

**Two vignettes, drawn from the existing 27-vignette Exp1 bank, matched on
difficulty tier** (both "medium," not one easy and one hard) so a difficulty
gap between the two vignettes can't masquerade as a condition effect. Avoid
any vignette that's already been shown in a talk or demo to someone likely to
be a participant (lab-mates in particular) -- prior exposure to the exact
answers would be a real practice-effect confound.

**Each vignette needs to exist as a natural-language "patient script,"**
not the raw structured profile. A participant reading raw field/value pairs
and typing them into either system isn't testing anything realistic -- real
patients don't receive their own history as a table. The script should read
the way `respondents.SimulatedPatientAgent`'s disclosures already read for
Exp4: a paragraph or two describing the fictional patient's own health and
family history in plain language, phrased the way a person would actually
describe their family, not itemized by field. This still needs to be
written from the two chosen vignettes' actual data -- once you've picked the
two vignettes, share the underlying profiles and I can draft both scripts.

**Observer tally sheet** for the form condition (a template is below) --
CanRisk's form can't be instrumented the way the agent can, so this
condition needs either a live observer or a screen recording coded
afterward against the same sheet.

**Short post-task questionnaire** (template below) -- perceived ease, trust,
and preference, to triangulate against the quantitative metrics.

## Procedure

1. **Consent and framing.** Participants are told upfront they're entering a
   *fictional* patient's information, not their own -- this substantially
   lowers the ethical/IRB weight of the study (no real personal health
   information is collected) and should be stated plainly so participants
   don't self-censor or feel exposed.
2. **Counterbalancing assignment.** Each participant is assigned, in
   rotation: (a) which condition comes first (agent or form), and (b) which
   vignette goes with which condition. With 2 orders x 2 vignette-to-condition
   mappings, that's 4 counterbalancing cells -- with 6+ participants, assign
   in rotation so cell sizes stay as even as possible (e.g. participants
   1/5/9... get cell 1, 2/6/10... get cell 2, and so on).
3. **Condition 1:** participant reads their assigned patient script privately,
   then completes intake using the assigned tool. Observer present (or
   recording running) for the form condition; the agent condition is
   self-logging (see "Instrumentation" below) but a live observer is still
   worth having for both, informally, in case something breaks.
4. **Short break** (~2 minutes) between conditions.
5. **Condition 2:** same as above, with the other script and the other tool.
6. **Post-task questionnaire.**
7. **Debrief:** explain the study's actual purpose, thank the participant.

Total time per participant: budget roughly 10-20 minutes per condition
(agent conversations in earlier testing have run in that range; CanRisk's
form may run faster or slower depending on the participant's familiarity
with structured medical forms -- worth timing a pilot participant first to
calibrate the session length you tell people to expect).

## Metrics

### User interventions (primary)

An "intervention" is any point where the participant had to go back and
change, clarify, or re-enter something, rather than the system correctly
capturing their answer the first time. This needs a concrete, countable
definition per condition so two different observers (or the same observer on
different days) would count the same thing:

**Agent condition** -- counted automatically from the saved session log
(see "Instrumentation"), as an occurrence of any of:
- A teach-back correction (`correct_at_teach_back` triggered -- the
  participant said something other than a clean confirmation).
- An ambiguous-field follow-up (`compose_ambiguous_followup` triggered).
- A plausibility or fidelity flag resolution during VERIFY.
- A cross-section-warning or holistic-audit "worth a second look" prompt.

**Form condition** -- counted live (or from a recording) by the observer,
using the tally sheet below. An intervention here is:
- A field the participant edited more than once.
- Any CanRisk validation error the participant had to resolve before
  proceeding.
- Any time the participant navigated back to an earlier section of the form
  to change something already entered.

```
FORM CONDITION -- OBSERVER TALLY SHEET
Participant #: _____   Vignette: _____   Order position: 1st / 2nd

Field re-edited (tally each occurrence):        | | | | | | | |
Validation error encountered (tally):           | | | | | | | |
Navigated back to an earlier section (tally):   | | | | | | | |

Notes on anything unusual (confusion, long pause, gave up on a field, etc.):
_________________________________________________________________
```

### Accuracy parity (primary)

Not "does the agent beat the form" -- it's a guardrail: does the agent's
accuracy match the form's, so any efficiency gain isn't bought at the cost
of correctness. For each condition, compare the participant's final entered
data against the vignette's known gold profile, field by field, the same
way Exp1 scores agent output against gold. For the agent condition this
comes directly from the submitted structured section values (see
"Instrumentation"); for the form condition, have the participant export or
you screen-record their final CanRisk pedigree file, since there's no other
way to recover their entered values afterward.

Report per-field accuracy and all-fields-correct rate for each condition,
same units as Exp1, so the two conditions -- and Exp1's own numbers -- are
directly comparable.

### Secondary metrics

- **Time on task**, start to submission, per condition.
- **Post-task questionnaire** (below) -- a quick read on subjective
  experience to sit alongside the two primary metrics, not a replacement
  for them.

```
POST-TASK QUESTIONNAIRE
Participant #: _____

For EACH condition (Agent / Form), rate 1 (strongly disagree) - 5 (strongly agree):
  "This was easy to complete."                          Agent: __  Form: __
  "I was confident I entered things correctly."         Agent: __  Form: __
  "I would be comfortable using this without help."     Agent: __  Form: __

Which did you prefer overall?   Agent / Form / No preference

Anything that confused you, in either one?
_________________________________________________________________
```

## Instrumentation

`web_backend.py` currently holds sessions only in memory -- nothing is
written to disk, so a session's transcript and timing are lost the moment
the process restarts, and there's no way to compute the agent-condition
intervention count or timing after the fact. This needs fixing before
running real participants; see the accompanying code change.

Two things need to be captured per agent-condition session, tied to a
participant/vignette identifier:
1. **The full transcript with timestamps**, so intervention counting and
   timing can both be computed after the session ends, not only live.
2. **The final structured state at submission** (every section's `.values`,
   plus the pedigree), so accuracy-parity scoring has something to compare
   against gold without needing to re-parse the transcript.

## Sample Size and Analysis Plan

6+ participants, within-subjects. This is still a small, directional
comparison -- report it that way in the paper, not as a powered result.
For each participant, report (condition, interventions, accuracy,
time-on-task, preference) as a row in a table; report means/medians/ranges
per condition as the summary, and since this is within-subjects, the
per-participant agent-minus-form differences are worth reporting too (does
the same person consistently need fewer interventions with the agent, not
just fewer on average across different people). Do not report a p-value
from this sample size as though it were a validated effect -- if the
pattern is clean and consistent across most participants, say so
descriptively; if it's mixed, say that too, rather than picking the
framing that looks best.

## Recruitment

Prioritize genuinely naive, non-clinical participants over convenient
labmates where possible -- the whole thesis is accessibility for people who
can't complete BOADICEA's own form, so a participant who's already
comfortable with structured medical intake forms is a worse test of that
claim than someone who isn't. Lab-mates are a reasonable fallback for
scheduling reasons, but note in the paper if that's who most participants
ended up being, since it's a real limitation on how strongly the results
generalize to the population the tool is actually meant for.

## What This Experiment Does Not Establish

Six-plus participants on two fictional vignettes doesn't establish
population-level usability, doesn't cover the full difficulty range the
27-vignette bank spans, and doesn't test how a genuinely distressed or
unwell patient (rather than someone role-playing a fictional one) behaves.
Frame it in the paper as a first, real-user signal on the accessibility
claim -- consistent with the rest of the project's evidence being about
capability and robustness, with this experiment being the one piece that's
actually about the person on the other end of the conversation.