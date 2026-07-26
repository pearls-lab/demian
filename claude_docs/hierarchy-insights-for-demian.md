# Hierarchy project → DeMiAn S2-language project: what transfers

Review of `hierarchy-project-state.md` (copied here 2026-07-26), read for what it means for OUR
work (S1↔S2 language interface + primitive/curriculum arms on RoboCasa composite). Bottom line:
**the hierarchy project independently establishes the exact lever our stacked-arm result was
pointing at — an auxiliary future-state head that improves COMPOSITE SR — with a much more mature,
powered, seed-hardened experiment. We should adopt it.**

## 1. The headline transfer: I1b (future-state anchor aux) is a PROVEN composite SR lever

Hierarchy's composite i1b experiment (their §"COMPOSITE i1b — 3-SEED FINAL" / "FULL 18-TASK"):
- **`-dynamic` (subgoal-chained, = our oracle mode): base 36.5% → i1b 43.1% = +6.5pp, McNemar
  p=0.00073, 3 seeds, n=720/arm.** On the doc's 18 tasks: **+4.9pp, p=1e-5, wins 12/18.**
- `-fix` (monolithic full-instruction): null (+2.5pp, p=0.25).

**i1b = a small MLP head off the action expert's suffix that regresses cumulative EEF displacement
at horizons {5,10,20,40,80,160}, aux weight 0.1, z-scored, never read at inference.** It is
exactly the "predict the future, don't consume it" delivery.

**Why this matters to us NOW:** our stacked arm (aug + **I1a** Sobolev) beat the init on sg_all
**p=0.011** (Part 9) — the one lever in our whole search that showed a powered win. The hierarchy
project shows the *sibling* aux mechanism (**I1b** anchor head) gives an even cleaner, seed-hardened
composite win **in the chained regime our eval uses.** Two independent projects, two aux-delivery
mechanisms, same conclusion: **aux future-state supervision raises composite SR.** This is the
strongest signal in either project for where SOTA actually comes from.

→ **Action: port I1b (the anchor head) into the demian pipeline as an arm** (the demian tree already
has `physics_aux`/`next_state_aux` scaffolding, and I already added `deriv_aux` = I1a). I1b is more
composite-validated than I1a; adding it (or I1a+I1b) to the curriculum/primitive arms is the
highest-expected-value SOTA move we have.

## 2. The "delivery mechanism, not content, decides" law — corroborated across BOTH projects

Hierarchy's central pattern (their §"PHASE 4 → ORCHEST IMPLICATION"), same future-state info,
three deliveries:
| delivery | result |
|---|---|
| **aux target** the policy PREDICTS (I1a/I1b) | **HELPS** (+7.3 atomic p=0.005; +6.5 composite-dynamic p=0.0007) |
| decoded-space CONSTRAINT (C7) | **COLLAPSES** (0.2% — every task ~0) |
| conditioning INPUT the policy consumes | **HURTS** (input < no-input) |

**Our project independently reproduced this same law from the other side:** every *inference-time
input* language lever we tried washed or hurt — captions (powered null, Part 3), repositioning
prefixes (negative, Part 4c), switch-retiming (null, Part 5b) — while the *aux* lever (I1a) helped
(Part 9). Two projects, opposite starting points, same law. This is now a robust, cross-validated
finding, not a one-off: **the S2→S1 summary should shape the policy as a training-time aux target,
not be fed as an inference-time conditioning message.** It also sharpens the honest tension we've
been living with — the delivery that *helps* isn't an "interface" (S2 not queried at inference =
representation-shaping), and the delivery that *is* an interface doesn't help.

## 3. Regime-dependence — and it favors our setup

i1b helps **only in the subgoal-chained (`-dynamic`) regime**, null on monolithic (`-fix`). Our
composite evals run oracle/dynamic subgoal-switching = **the chained regime where the aux helps.**
So an I1b/I1a arm in our pipeline is predicted to help, and I1a already did.

Mechanism (their fixed-checker re-run, §"MECHANISM RE-RUN"): the `-dynamic` gain is **roughly
balanced between better first-subgoal execution AND better handoff** (2nd-given-1st), ~53/42 —
directly relevant to our commitment/handoff bottleneck (Part 5). An earlier buggy-checker version
had mislabeled it ~80/20 first-subgoal; worth knowing their subgoal checkers needed fixing
(SnackSorting `drawer_closed` too strict, FryingPan `any_burner_on` too loose) — **the same
`composite_subgoals.py` checkers we use.** If we quote per-subgoal SR, cross-check those two.

## 4. Warnings we already respect, now with independent evidence

- **C7 / decoded-space constraint COLLAPSES the policy** (0.2%). We correctly never touched it; our
  physics_aux is the SAFE aux-target mode. Their result is the cautionary datapoint.
- **Baseline reconciliation**: their composite baselines match longhrz on the shared tasks once you
  control for task-selection (the "23% vs 43%" gap was 100% task-selection over floor tasks). Same
  discipline we've used (headroom set, n≥240 McNemar).

## 5. Ops lessons — I independently rediscovered most of these this session

The hierarchy notes document, repeatedly, the exact traps that cost me cycles here:
- **`pgrep -fc <pat>` / `[x]`-bracket patterns self-match the ssh/shell cmdline** → false "1 proc",
  stuck loops. (Hit me ~5× this session — probes, chains, kills. Their fix: `ps --no-headers -C
  python -o cmd | grep -c`, or match by logged PID.)
- **Train from NVMe, never `/data`** (mergerfs stalls/branch-loss). We stage to `~/orchest_data`.
- **The trainer NEVER prunes checkpoints (~40G each: 7.5G model + 13.5G optimizer)** → need a disk
  janitor. This is exactly the jeju-3 disk risk I flagged for the curriculum Stage 1 — confirmed
  real, and worse than I estimated (optimizer state doubles it).
- **Leaked grain/multiprocessing dataloader workers reparent to ppid=1, hold ~95G RAM, OOM the next
  run.** Reap by `ps -eo pid,ppid,cmd | awk '$2==1 && /openpi-venv.*(multiprocessing.spawn|resource_tracker)/'`.
- **`pi05_base_pytorch` at 6.8G** (we're using it as the curriculum init — confirms it's the right artifact).
- **nohup+setsid+& inside a one-shot ssh is racy** — verify by log mtime + real proc count, retry.
- SHORT hostnames don't resolve in fresh shells — FQDN.

## 6. Concrete next moves this suggests for us
1. **Add an I1b arm** (future-state anchor head, suffix, aux 0.1, z-scored, horizons
   {5,10,20,40,80,160}) to the demian pipeline — the mature, composite-proven version of the I1a
   win. Consider I1a+I1b together. This is the clearest SOTA lever available.
2. **When we quote per-subgoal SR**, apply their SnackSorting/FryingPan checker fixes (they patched
   `composite_subgoals.py` — the same file we serve).
3. **Add a checkpoint janitor** to our training launches (keep latest + stage checkpoints), given
   the confirmed ~40G/ckpt no-prune behavior — especially on the disk-tight boxes.
4. The regime-dependence says our **oracle/dynamic evals are the right place to see aux gains** —
   keep them as the primary readout for any aux arm.

Full source: `hierarchy-project-state.md` (this dir). Their reports live in `~/hierarchy/docs/`.
