# Evaluation modes — how DeMiAn S2-language results are scored

Harness: `~/demian/openpi/scripts/eval/s2_guided_eval.py`. Success is scored by RoboCasa
**simulator-state sub-goal checkers** (`composite_subgoals.py`), not by a VLM judge. The three
modes differ only in *where the instruction text comes from* and *what triggers a switch*.

## The bottom line
**Every powered arm result so far (aug vs ctrl, stacked/I1a, primitive probes, the pending I1b
arms) was run in `oracle` mode = pre-defined ground-truth sub-instructions + a simulator sub-goal
checker.** The true reactive-VLM harness (`s2`) exists and was wired to a Qwen3-VL instructor, but
it produced only the caption-injection powered-null and small-n diagnostics — never a load-bearing
arm number. The reactive-VLM-issuing-corrections condition (the project thesis) is still untested.

## The modes

### `oracle` — GT sub-instructions, checker-triggered switching  ← all powered results
- The composite task's **pre-defined subgoal decomposition** supplies the instruction for each
  subgoal (ground-truth text, not model-generated).
- The switch from subgoal-1's instruction to subgoal-2's is **triggered by the RoboCasa sub-goal
  success checker** reading simulator state (`composite_subgoals.py`) — with a switch-buffer delay
  (default ~20 steps into the next motion).
- No VLM in the loop: nothing decides *what* to say or *when* beyond the scripted decomposition +
  checker signal. This deliberately **isolates the S1 policy** from reasoner quality.
- This is the setting for the mandated gate: **n≥240 paired episodes + McNemar**, on the 10-task
  headroom set.

### `fix` — monolithic
- The full composite instruction given **once**, no subgoal switching. The `-fix` regime in the
  hierarchy project's terms; the aux levers (I1b) are null here.

### `s2` — the true reactive-VLM harness (thesis condition; not yet powered)
- File-bridge: frame + `status.json` dumped every `--tick` steps; a reasoner replies with commands
  via `inbox/cmd_NNN.json`. A stuck-detector triggers early ticks.
- A Qwen3-VL **instructor** was wired in for the caption/instructor comparison. Result so far:
  caption injection is a **powered null** (oracle .129 / gtcap .108 / instructor .125, all p≥0.50)
  plus small-n probe diagnostics. No arm result rides on this mode yet.

## How the harness loop works (and how often each model is called)

One closed control loop per episode: `for t in range(max_steps)` (default max ~600 steps, RoboCasa
at 20 fps). Each iteration:
1. `client.infer(obs, instruction, reasoning_text)` → **S1** (DeMiAn-VLA, served) returns an action
   chunk. The client caches the chunk and only runs a NEW policy forward when the previous chunk is
   exhausted (`actions_from_chunk_completed >= open_loop_horizon`, default **8**) — so between
   inferences the 8-step chunk executes **open-loop**. On any instruction switch the chunk is
   force-expired so S1 re-plans immediately.
2. `env.step(action)` advances the sim.
3. Sub-goal checkers (`composite_subgoals.py`) fire milestones on rising edges; a stuck detector
   flags when EEF displacement span over a trailing window falls below threshold.
4. Instruction/reasoning switching — **mode-specific** (see below).
5. Break on `success` / `done` / `trunc`.

**Call frequency:**
- **S1 policy** — a forward pass every `open_loop_horizon = 8` sim steps (~2.5×/s at 20 fps),
  regardless of mode. It is the thing being evaluated.
- **S2 instructor VLM (Qwen3-VL)** — depends on mode:
  - **`oracle`** (all powered results): invoked **only at sub-goal boundaries** — once at episode
    start (`_phase_reasoning(phases[0])`) and once at each checker-triggered handoff (after
    `switch_buffer = 20` steps). For a 2-subgoal task that's **~2 instructor calls/episode**, and
    when the arm uses GT captions the live VLM isn't called at all. Timing is driven by the
    simulator checker, not the VLM.
  - **`s2`** (true reactive harness): invoked at every `_s2_checkpoint` = every `--tick = 40` sim
    steps (~1×/2 s), **plus immediately on a stuck event**. A ~400–600-step episode → ~10–15
    ticks + stuck-triggered calls. Between ticks S1 runs open-loop on the last instruction.

Defaults: `--open-loop-horizon 8`, `--tick 40`, `--switch-buffer 20` (`s2_guided_eval.py`).

## What "success" means
- Scored by simulator-state checkers (grasp / place / subgoal completion), so reports quote
  **P(subgoal-1)** and **P(all | subgoal-1)** rather than raw SR alone. Measured cliff:
  P(sg1)≈.64, P(all|sg1)≈.25; `sg2_handoff` is the top failure class (~37% of episodes).
- Serve S1 from the vendored openpi with `--obs-keys demian`; run the eval with
  `/data/bkim/envs/robocasa/bin/python`, `MUJOCO_GL=egl`. Exact-scene restore uses `ep_meta`
  pinning (`env.reset(seed=)` alone does NOT reproduce a RoboCasa scene).

## Caveat — checker bugs (found by the sibling hierarchy project, 2026-07)
The same `composite_subgoals.py` checkers we serve have two known bugs: **SnackSorting
`drawer_closed` too strict** and **FryingPan `any_burner_on` too loose**. Cross-check / apply their
patches before quoting per-subgoal SR on those two tasks. Full context:
`~/demian/claude_docs/hierarchy-insights-for-demian.md`.
