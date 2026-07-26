# Improving the S1↔S2 harness — what our results say, and what to build

Companion to `evaluation-modes.md`. Written 2026-07-26 from the powered findings in
`~/s2_sessions/analysis_2026-07-22.md` and the project memory.

## The diagnosis our results force
Every **inference-time language lever is measured dead**: caption injection (powered null, all
p≥0.50), repositioning prefixes (−0.19 in the best-powered cell), switch-retiming (null), corrective
language (ignored). The **commitment probe** explains why: S1's instruction-sensitivity is highest
at reset and subgoal boundaries and **drops once it commits to a motion** (mid_sg1−reset −1.43,
p=0.003) — an in-distribution effect, not OOD. But at the handoff, where it *is* receptive, it still
fails at **P(all | sg1)≈.25**: it *hears* the new instruction and *can't execute* the recovery
(empty-gripper OOD states, limited repertoire).

**⇒ The binding constraint is S1's execution repertoire at the handoff — not S2's intelligence or
how often it talks.** A smarter or more talkative System 2 bolted onto today's S1 is predicted-dead
by our own measurements. Every harness change below is judged by whether it gives S2 a job S1 can
act on, and every one assumes S1 is *also* being trained (curriculum + I1b + finer annotations) to
widen that repertoire. Harness and policy must co-evolve, sharing one vocabulary.

Corollary (the **delivery-mechanism law**, cross-validated with the hierarchy project): the same
future/plan information *helps* as a training-time aux target, *collapses* as a decoded constraint,
and *hurts/washes* as an inference-time input. So System 2's intelligence should reach SR through
**training**, not through a message consumed at inference.

## Five improvements, each tied to a finding

**1. Reframe frontier S2 from *narrator* → *perceiver + gate*.** s2-mode calls the VLM every 40
steps on a wall clock — mostly into committed motion, where it's wasted. Use the frontier VLM for
what it beats a scripted checker at: **detecting OOD/failure the checker can't see** (gripper closed
on nothing, object slipped, arm wedged) and confirming true motion completion. Fire it adaptively on
suspected trouble/boundary, not on a clock. Bonus: routes around the `composite_subgoals.py` checker
bugs (SnackSorting `drawer_closed` too strict, FryingPan `any_burner_on` too loose) — run VLM +
checker together.

**2. Deliver only at receptive moments.** Gate S2 injection to S1's receptive windows (reset,
subgoal boundary, detected motion-completion), never mid-chunk — measured-dead there. **← a first
concrete version is implemented (see below).**

**3. Constrain frontier S2 to S1's *trained* vocabulary.** Free-form frontier language is dead
because S1 never saw it. Give the S2 the **menu of primitive/atomic instructions S1 was fine-tuned
on** and have it *select/compose* from that menu, not generate arbitrary text. (This is the
"LLM-picks-skill-specific-instructions" idea, now data-motivated.)

**4. Finer decomposition so S1 never faces a big handoff.** A frontier LLM planner decomposes the
composite into a **primitive-level next-step stream** (matching the primitive→atomic curriculum), so
S1 always has a small, in-distribution instruction and the cliff becomes a sequence of receptive
boundaries. Tightest coupling to the curriculum/I1b work now running.

**5. Harness as a training flywheel (highest-EV frontier use).** Per the delivery law, the frontier
VLM's best contribution is **offline**: generate fine-grained recovery/handoff annotations on
Method-1's augmented OOD start states, fine-tune S1, and let the harness be the data-gen +
evaluation loop. The model's intelligence reaches SR through training — the only delivery that has
moved the needle.

## Frontier vs local (Qwen3-VL), answered honestly
A frontier S2 (Claude/GPT/Gemini) buys better OOD perception (#1) and recovery planning (#3/#4) —
but both convert to SR **only after** S1's repertoire is widened. Run frontier-vs-Qwen3-VL as a
powered A/B *after* that training, not before (before, expect a wash — both bottlenecked on S1). And
whichever S2 you use, **power the s2 mode itself** (n≥240 paired vs oracle, checker bugs fixed,
intervention decisions logged) — the thesis mode currently has zero powered evidence.

## Sequencing
S1 trained on primitives/recovery vocab (curriculum + I1b, running) → S2 constrained to that vocab
(#3), gated to receptive windows (#2), decomposing to primitive granularity (#4), using perception
for failure detection (#1); frontier-vs-local A/B and the s2-mode power run come last.

## Concrete change implemented: `--switch-on-receptive` (#1–#2 first step)
`s2_guided_eval.py` now computes a **receptivity signal** — EEF span over a short trailing window
(`--receptive-window`, default 8) below `--receptive-eps` (default 0.02 m) ⇒ S1 is between motions
and listening — and:
- **Exposes `receptive` in `status.json`** so a (frontier) System 2 can time interventions to the
  window where S1 actually hears it (infrastructure for #1/the s2 path).
- **`--switch-on-receptive`** (oracle mode): defers the subgoal-instruction swap to the next
  receptive window (or `--max-defer` steps, default 60) instead of the fixed `--switch-buffer` (20)
  offset that lands the swap *into* the next motion. Switch events log `deferred`/`receptive`/`gated`
  for post-hoc analysis.

**A/B test (against the existing oracle baseline, same n≥240 paired protocol):**
```
# baseline arm (current behavior):  --mode oracle                (fixed switch-buffer 20)
# receptive arm:                     --mode oracle --switch-on-receptive
```
Serve DeMiAn-VLA, 10-task headroom set, seed-paired, McNemar on P(all | sg1). Defaults leave
behavior byte-identical, so this is a safe addition.

**RESULT (2026-07-26, n=240 paired) — NULL.** strict SR .154→.167 (p=0.76), P(sg1) .646→.646
(p=1.00), P(all) .192→.188 (p=1.00), P(all|sg1) .296→.290. The gating was *real*, not degenerate —
recv switch deferrals ran min 0 / **med 7** / max 60 (vs baseline's fixed 20), so ~2/3 of switches
delivered at a found receptive window and the delivery timing genuinely differed. It **did not
matter.** ⇒ The commitment effect is about **execution, not delivery timing**: S1 hears the sg2
instruction at the boundary regardless of exact timing; the bottleneck is that it can't *execute*
the handoff. This is the clean negative that **closes the inference-side timing question** —
captions, prefixes, switch-retiming, and now adaptive receptivity are all null. The remaining lever
is **training** (#3–#5 + the I1b/curriculum work now running). The `receptive` signal in status.json
still stands as useful infrastructure for a frontier S2's *perception/failure-detection* role (#1),
a different bet than delivery timing that remains untested.

Not yet built (next, in priority order): the frontier-VLM failure/OOD detector (#1 perception),
the trained-vocabulary menu (#3), and the primitive-stream planner (#4) — all gated on the
primitive/atomic curriculum checkpoints now training.
