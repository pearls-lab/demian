# S2-guided composite eval (`s2_guided_eval.py`)

Closed-loop harness for studying the System-2 (VLM/human) → System-1 (DeMiAn-VLA)
**language-instruction interface** on RoboCasa composite tasks. One continuous env per
episode (controller state survives instruction switches), three instruction drivers:

| mode | instruction source | corresponds to |
|---|---|---|
| `fix` | constant full composite instruction (env's per-episode lang) | longhrz `-fix` |
| `oracle` | `TASK_SUBGOAL_PLAN` sim-state triggers + switch buffer | longhrz `-dynamic` |
| `s2` | external System 2 over a file bridge (see below) | the research object |

## Quick start

```bash
# 1. Serve a policy (openpi-i1b tree; aug26k = Method-1 composite-split model):
cd ~/openpi-i1b && OPENPI_NO_COMPILE=1 PYTHONPATH=src:packages/openpi-client/src \
  CUDA_VISIBLE_DEVICES=7 ~/openpi-venv/bin/python scripts/serve_policy.py --port 8210 \
  policy:checkpoint --policy.config pi05_robocasa_composite_split_aug \
  --policy.dir ~/ckpts/aug26k/25999
PYTHONPATH=src:packages/openpi-client/src ~/openpi-venv/bin/python ~/warmup_infer.py 8210

# 2. Run (from this directory, with the robocasa eval python):
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=6 /data/bkim/envs/robocasa/bin/python \
  s2_guided_eval.py --task ThawInSink --mode s2 --block --tick 40 \
  --host localhost --port 8210 --seed 42 --episodes 1 --max-steps 700 \
  --log-state --video --session-dir ~/s2_sessions/my_run
```

`--obs-keys i1b` (default) remaps this repo's `agentview_*` request keys to the
original-openpi `image/wrist_image/image_right` convention and slices the returned
12-dim action to the effective 7-dim (`action[5:12]`, no-nav split). Use
`--obs-keys demian` against this repo's own vendored `serve_policy.py`.

## The s2 file bridge

At every `--tick` steps (plus step 0, and immediately on a stuck event) the harness
writes `ep{N}/t{step}.png` (left|right|wrist) and `ep{N}/status.json`
(instruction, milestones, canonical subgoal decomposition, `next_cmd`), then — with
`--block` — pauses until the file named by `next_cmd` appears:

```bash
echo '{"instruction": "pick up the water bottle", "note": "hover>100 steps, rephrase"}' \
  > <session>/ep00/inbox/cmd_003.json      # {} or {"note": ...} = keep current text
```

`reasoning` may also be set per command; it is forwarded as `reasoning_text`
(only DeMiAn LM/caption-trained policies consume it).

## Instrumentation (all modes)

- strict env success (`info["success"]`) — the headline SR signal;
- lenient `SUBGOAL_CHECKERS` milestones, latched on a rising edge from the reset
  baseline (pre-satisfied milestones excluded), with first-fire steps;
- full switch history (`source`: initial/oracle/s2, with S2 notes);
- `--log-state`: per-step EEF/gripper/action JSONL;
- stuck events (EEF span < `--stuck-eps-m` over `--stuck-window` steps) with frames;
- frames at milestones/switches/final, `--snap N` for periodic snapshots,
  `--video` for a subtitled mp4 (uses imageio-ffmpeg's bundled binary).

Per-episode artifacts land in `<session>/ep{N}/` (`result.json`, `events.jsonl`,
`state_log.jsonl`, frames); the session dir gets `args.json` + merged `results.json`.

## Caveats

- Small-n runs are diagnostics: the measured per-episode flip noise is ~35 %
  (longhrz), so claims need the n>=240 paired-McNemar gate
  (`~/orchest/docs/c0-headroom-taskset.md`).
- Milestone leniency: per-subgoal numbers use the lenient checkers (th=0.05);
  strict full-task SR is not comparable to them (see orchest c0-progress notes).
- This repo's registries cover the 18 2-subtask baseline tasks; the richer
  25-task registry (long-horizon ClearCuttingBoard, CerealAndBowl, ...) lives in
  `~/orchest/eval/chai-stack/composite_subgoals.py`.
