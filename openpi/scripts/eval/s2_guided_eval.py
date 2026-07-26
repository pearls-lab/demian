#!/usr/bin/env python
"""System-2-guided closed-loop eval for RoboCasa composite tasks.

One continuous env per episode (no segment restarts -> controller state is
preserved across instruction switches), with the instruction channel driven
by one of three System-2 modes:

  --mode fix     constant full composite instruction    (the -fix baseline)
  --mode oracle  TASK_SUBGOAL_PLAN sim-state triggers swap per-subgoal
                 instructions with a switch buffer      (the -dynamic baseline)
  --mode s2     file-bridge System 2: every --tick steps (and on milestone /
                 stuck events) the harness dumps a 3-camera frame +
                 status.json into the session dir, then (if --block) waits for
                 a command file before continuing. Any external System 2 -- a
                 human, Claude, or a VLM script -- can drive System 1 by
                 writing inbox/cmd_NNN.json:
                     {"instruction": "...",            # new prompt (optional)
                      "reasoning": "...",              # reasoning_text (optional,
                                                       #   only LM-trained policies use it)
                      "note": "why"}                   # logged, not sent to policy
                 An empty command ({}) means "continue with the current text".

Instrumentation (all modes): strict env success (info["success"]), lenient
per-subgoal milestones from composite_subgoals.SUBGOAL_CHECKERS latched on a
rising edge from the reset baseline (pre-satisfied milestones excluded),
first-fire steps, full instruction switch history, per-step EEF/gripper log,
stuck detection (trailing-window EEF displacement span), frames at
milestones/switches/stuck, optional subtitled video.

Run with the robocasa eval python + MUJOCO_GL=egl, cwd = this directory,
against a serve_policy.py server, e.g.:

  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=7 /data/bkim/envs/robocasa/bin/python \
      s2_guided_eval.py --task ThawInSink --mode s2 --block \
      --host localhost --port 8210 --seed 42 --episodes 1 \
      --session-dir ~/s2_sessions/thaw_demo
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

import gymnasium as gym
import robocasa  # noqa: F401  registers envs

import run_robocasa_eval as R
import composite_subgoals as CS
from robocasa_task_list import ALL_TASKS

_CAMS = [
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
]

# The vendored openpi in this repo names camera keys agentview_left/right +
# eye_in_hand; the original openpi trees (openpi-i1b/-orchest/-main, which serve
# the longhrz/aug checkpoints) use image / wrist_image / image_right.
_I1B_KEYMAP = {
    "observation/agentview_left": "observation/image",
    "observation/eye_in_hand": "observation/wrist_image",
    "observation/agentview_right": "observation/image_right",
}


def _adapt_client_keys(client: "R.RoboCasaClient") -> None:
    """Remap request keys / action dims so this harness can talk to an
    openpi-i1b-style server (which also returns the full 12-dim RoboCasa
    action; the effective control is action[5:12] in the no-nav split)."""
    inner = client.client.infer

    def infer(req: dict, _inner=inner) -> dict:
        result = _inner({_I1B_KEYMAP.get(k, k): v for k, v in req.items()})
        actions = np.asarray(result["actions"])
        if actions.ndim == 2 and actions.shape[-1] == 12:
            result = {**result, "actions": actions[:, 5:12]}
        return result

    client.client.infer = infer


def _save_frame(obs: dict, path: Path) -> None:
    """left | right | wrist, side by side."""
    imgs = [np.asarray(obs[k], np.uint8) for k in _CAMS]
    Image.fromarray(np.concatenate(imgs, axis=1)).save(path)


class EventLog:
    def __init__(self, path: Path):
        self._f = open(path, "a")

    def add(self, **kv) -> None:
        self._f.write(json.dumps(kv) + "\n")
        self._f.flush()


class S2Bridge:
    """File-based System-2 command channel for one episode.

    Commands are consumed strictly in order: inbox/cmd_001.json, cmd_002.json, ...
    (zero-padding optional -- files are matched by the integer in the name).
    """

    def __init__(self, ep_dir: Path):
        self.inbox = ep_dir / "inbox"
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.next_idx = 1

    def _candidate(self) -> Path | None:
        for p in self.inbox.glob("cmd_*.json"):
            try:
                if int(p.stem.split("_")[1]) == self.next_idx:
                    return p
            except (IndexError, ValueError):
                continue
        return None

    def poll(self, block: bool, timeout_s: float) -> dict | None:
        deadline = time.monotonic() + timeout_s
        while True:
            p = self._candidate()
            if p is not None:
                # Retry the read once: the writer may not have finished.
                for _ in range(10):
                    try:
                        cmd = json.loads(p.read_text())
                        break
                    except json.JSONDecodeError:
                        time.sleep(0.2)
                else:
                    cmd = {}
                self.next_idx += 1
                return cmd
            if not block or time.monotonic() > deadline:
                return None
            time.sleep(0.5)


def write_status(path: Path, **kv) -> None:
    path.write_text(json.dumps(kv, indent=1, default=str))


def run_guided_episode(
    env,
    client: "R.RoboCasaClient",
    task: str,
    mode: str,
    seed: int | None,
    ep_dir: Path,
    max_steps: int,
    tick: int,
    block: bool,
    wait_timeout: float,
    switch_buffer: int,
    switch_on_receptive: bool,
    receptive_eps: float,
    receptive_window: int,
    max_defer: int,
    oracle_caption_key: str | None,
    instructor: "R.QwenInstructor | None",
    stuck_window: int,
    stuck_eps_m: float,
    snap: int,
    record_video: bool,
    log_state: bool,
    instruction_override: str | None,
) -> dict:
    raw = env.unwrapped
    ep_dir.mkdir(parents=True, exist_ok=True)
    events = EventLog(ep_dir / "events.jsonl")
    bridge = S2Bridge(ep_dir) if mode == "s2" else None

    obs, _ = env.reset(seed=seed)
    client.reset()
    full_instruction = (
        instruction_override
        or obs.get("annotation.human.task_description")
        or ALL_TASKS.get(task, task)
    )

    # Oracle plan (also exposed to S2 as reference subgoal decomposition).
    # With oracle_caption_key set, phases also carry the GT caption for that key,
    # injected as reasoning_text (the DeMiAn "dynamic-gtcap" arm).
    plan = CS.get_subgoal_plan(task)
    swap_mode = "both" if oracle_caption_key else "prompt"
    phases = (
        CS.resolve_subgoal_runtime(plan, oracle_caption_key, full_instruction, None, swap_mode)
        if plan
        else None
    )
    canon = [p["instruction"] for p in phases] if phases else [full_instruction]

    checker = CS.get_subgoal_checker(task)
    keys = list(checker(raw).keys()) if checker else []
    # Milestones already true at reset are excluded (checker leniency: some
    # object-under-fixture layouts read True at t=0).
    reset_ms = {k: bool(v) for k, v in (checker(raw).items() if checker else {})}
    reached = {k: False for k in keys}
    first = {k: -1 for k in keys}

    def _phase_reasoning(phase: dict, cur_obs: dict) -> str | None:
        """Reasoning for an oracle phase: instructor caption if wired, else GT caption."""
        if instructor is not None:
            try:
                import time as _time
                t0 = _time.monotonic()
                cap = instructor.generate(
                    [cur_obs[k] for k in _CAMS], phase["instruction"])
                dt = _time.monotonic() - t0
                if R._is_valid_caption(cap):
                    events.add(t=-1, event="instructor_caption", phase=phase["name"],
                               caption=cap, gen_time_sec=round(dt, 2))
                    return cap
                events.add(t=-1, event="instructor_caption_rejected", raw=cap[:80])
            except Exception as e:
                events.add(t=-1, event="instructor_error", error=str(e)[:120])
        return phase.get("reasoning") or None

    if mode == "oracle" and phases:
        instruction = phases[0]["instruction"]
        reasoning: str | None = _phase_reasoning(phases[0], obs)
    else:
        instruction = full_instruction
        reasoning = None

    switches = [{"step": 0, "instruction": instruction, "source": mode, "note": "initial"}]
    events.add(t=0, event="start", task=task, seed=seed, mode=mode,
               instruction=instruction, canonical_subgoals=canon, reset_ms=reset_ms)
    _save_frame(obs, ep_dir / "t00000.png")

    def _status(step: int, waiting: bool, extra_event: str | None = None) -> None:
        d = checker(raw) if checker else {}
        write_status(
            ep_dir / "status.json",
            task=task, seed=seed, mode=mode, step=step, max_steps=max_steps,
            instruction=instruction, reasoning=reasoning,
            milestones_now={k: bool(v) for k, v in d.items()},
            reached_ever=reached, first_fire=first, reset_ms=reset_ms,
            canonical_subgoals=canon, full_instruction=full_instruction,
            waiting_for_command=waiting, event=extra_event,
            # Commitment-effect signal for System 2: True when S1 is between motions
            # (low EEF velocity) and therefore receptive to a new instruction. A
            # frontier S2 should time its interventions to this window.
            receptive=bool(cur_receptive),
            frame=str(ep_dir / f"t{step:05d}.png"),
            next_cmd=f"inbox/cmd_{bridge.next_idx:03d}.json" if bridge else None,
        )

    def _s2_checkpoint(step: int, why: str) -> None:
        """Dump frame+status; in --block mode wait for the next S2 command."""
        nonlocal instruction, reasoning
        _save_frame(obs, ep_dir / f"t{step:05d}.png")
        _status(step, waiting=block, extra_event=why)
        cmd = bridge.poll(block=block, timeout_s=wait_timeout)
        if cmd is None:
            return
        note = cmd.get("note", "")
        new_instr = cmd.get("instruction")
        if "reasoning" in cmd:
            reasoning = cmd["reasoning"] or None
        if new_instr and new_instr != instruction:
            instruction = new_instr
            client.actions_from_chunk_completed = client.open_loop_horizon  # fresh chunk
            switches.append({"step": step, "instruction": instruction,
                             "source": "s2", "note": note})
            events.add(t=step, event="s2_switch", instruction=instruction, note=note)
        else:
            events.add(t=step, event="s2_continue", note=note)
        _status(step, waiting=False)

    # Commitment-effect receptivity: True when S1's EEF is nearly still over a short
    # trailing window (between motions), the moment its instruction-sensitivity is
    # highest (analysis Part 5). Reset is receptive by definition.
    cur_receptive = True

    # s2 mode: give System 2 the chance to set the opening instruction.
    if mode == "s2":
        _s2_checkpoint(0, "episode_start")

    video: list[np.ndarray] = []
    state_log = open(ep_dir / "state_log.jsonl", "a", buffering=1) if log_state else None
    eef_hist: deque[np.ndarray] = deque(maxlen=stuck_window)
    stuck_cooldown = 0
    oracle_pending: int | None = None
    oracle_idx = 0
    strict = False
    t = 0

    for t in range(max_steps):
        ret = client.infer(obs, instruction, reasoning_text=reasoning)
        if record_video:
            video.append(
                R._add_subtitle(ret["viz"], [f"Task: {instruction}",
                                             f"Reasoning: {reasoning}" if reasoning else ""],
                                t, fixed_bar_lines=4))
        obs, _rew, done, trunc, info = env.step(ret["action_dict"])

        eef = np.asarray(obs["state.end_effector_position_relative"], np.float64)
        grip = np.asarray(obs["state.gripper_qpos"], np.float64)
        eef_hist.append(eef)
        if state_log is not None:
            state_log.write(json.dumps({
                "t": t, "eef": eef.round(4).tolist(), "grip": grip.round(4).tolist(),
                "action": np.asarray(ret["raw_action"]).round(4).tolist(),
                "instr_idx": len(switches) - 1}) + "\n")

        if info.get("success", False):
            strict = True

        # Milestones: rising edge from a False-at-reset baseline.
        if checker is not None:
            d = checker(raw)
            for k in keys:
                if d[k] and not reset_ms.get(k, False) and not reached[k]:
                    reached[k] = True
                    first[k] = t
                    _save_frame(obs, ep_dir / f"milestone_{k}_t{t:05d}.png")
                    events.add(t=t, event="milestone", name=k)

        # Stuck detection: EEF span over the trailing window below threshold.
        stuck = False
        if len(eef_hist) == stuck_window and stuck_cooldown == 0:
            arr = np.stack(eef_hist)
            span = float(np.linalg.norm(arr.max(0) - arr.min(0)))
            if span < stuck_eps_m:
                stuck = True
                stuck_cooldown = stuck_window
                _save_frame(obs, ep_dir / f"stuck_t{t:05d}.png")
                events.add(t=t, event="stuck", eef_span_m=round(span, 4),
                           instruction=instruction)
        stuck_cooldown = max(0, stuck_cooldown - 1)

        # Receptivity: EEF span over a short trailing window below threshold => the
        # policy is between motions and receptive to a new instruction (commitment
        # effect, analysis Part 5). Used to time instruction delivery to the window
        # where S1 actually listens, and exposed to System 2 via status.json.
        if len(eef_hist) >= receptive_window:
            rwin = np.stack(list(eef_hist)[-receptive_window:])
            cur_receptive = bool(np.linalg.norm(rwin.max(0) - rwin.min(0)) < receptive_eps)
        else:
            cur_receptive = False

        # Oracle subgoal switching (the -dynamic baseline).
        if mode == "oracle" and phases and oracle_idx < len(phases) - 1:
            cur = phases[oracle_idx]
            if oracle_pending is None:
                try:
                    if bool(cur["trigger_fn"](raw)):
                        oracle_pending = t
                except Exception:
                    pass
            # Default: swap a fixed switch_buffer steps after the trigger fires (into
            # the next motion). --switch-on-receptive instead defers the swap to the
            # next RECEPTIVE window (or --max-defer steps, whichever first) so the new
            # subgoal instruction lands while S1 is still listening, not mid-motion.
            if switch_on_receptive:
                ready = oracle_pending is not None and (
                    cur_receptive or (t - oracle_pending) >= max_defer)
            else:
                ready = oracle_pending is not None and (t - oracle_pending) >= switch_buffer
            if ready:
                oracle_idx += 1
                instruction = phases[oracle_idx]["instruction"]
                if instructor is not None or oracle_caption_key:
                    reasoning = _phase_reasoning(phases[oracle_idx], obs)
                client.actions_from_chunk_completed = client.open_loop_horizon
                switches.append({"step": t, "instruction": instruction,
                                 "source": "oracle", "note": phases[oracle_idx]["name"]})
                events.add(t=t, event="oracle_switch", instruction=instruction,
                           deferred=t - oracle_pending, receptive=cur_receptive,
                           gated="receptive" if switch_on_receptive else "fixed")
                _save_frame(obs, ep_dir / f"switch_p{oracle_idx}_t{t:05d}.png")
                oracle_pending = None

        # Periodic snapshot frames (post-hoc failure analysis in fix/oracle mode).
        if snap and t > 0 and t % snap == 0:
            _save_frame(obs, ep_dir / f"t{t:05d}.png")

        # S2 decision points: periodic tick, or immediately when stuck.
        if mode == "s2" and t > 0 and (t % tick == 0 or stuck):
            _s2_checkpoint(t, "stuck" if stuck else "tick")

        if strict or done or trunc:
            break

    _save_frame(obs, ep_dir / f"final_t{t:05d}.png")
    if state_log is not None:
        state_log.close()

    result = {
        "task": task, "seed": seed, "mode": mode, "success": bool(strict),
        "steps": t + 1, "reached": reached, "first_fire": first,
        "reset_ms": reset_ms, "switches": switches,
        "final_instruction": instruction,
        "canonical_subgoals": canon, "full_instruction": full_instruction,
    }
    (ep_dir / "result.json").write_text(json.dumps(result, indent=1))
    _status(t, waiting=False, extra_event="episode_end")
    events.add(t=t, event="end", success=bool(strict), reached=reached)

    if record_video and video:
        try:
            import mediapy
            try:  # no system ffmpeg on this box; use imageio-ffmpeg's bundled binary
                import imageio_ffmpeg
                mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
            except ImportError:
                pass
            mediapy.write_video(ep_dir / "rollout.mp4", video, fps=15)
        except Exception as e:
            print(f"  [video] write failed ({e}); keeping subsampled frames", flush=True)
            np.savez_compressed(ep_dir / "rollout_frames.npz",
                                frames=np.stack(video[::4]))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--mode", choices=["fix", "oracle", "s2"], default="oracle")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42,
                    help="episode i uses seed+i (matches longhrz seed-based eval)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open-loop-horizon", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--tick", type=int, default=40,
                    help="[s2] steps between System-2 decision points")
    ap.add_argument("--block", action="store_true",
                    help="[s2] pause the rollout at each decision point until a command arrives")
    ap.add_argument("--wait-timeout", type=float, default=600.0,
                    help="[s2 --block] max seconds to wait for a command before continuing")
    ap.add_argument("--switch-buffer", type=int, default=20,
                    help="[oracle] steps between trigger fire and instruction swap")
    ap.add_argument("--switch-on-receptive", action="store_true",
                    help="[oracle] deliver the pending subgoal instruction at the next "
                         "RECEPTIVE window (low EEF velocity = between motions) instead of a "
                         "fixed switch-buffer offset — tests the commitment-effect hypothesis "
                         "that S1 only listens at motion boundaries. Falls back to --max-defer. "
                         "A/B this against the default (fixed switch-buffer) oracle baseline.")
    ap.add_argument("--receptive-eps", type=float, default=0.02,
                    help="EEF span (m) over --receptive-window below which S1 is deemed receptive")
    ap.add_argument("--receptive-window", type=int, default=8,
                    help="trailing EEF samples for the receptivity (velocity) test")
    ap.add_argument("--max-defer", type=int, default=60,
                    help="[--switch-on-receptive] max steps to defer a swap waiting for a "
                         "receptive window before forcing it")
    ap.add_argument("--oracle-caption-key", default=None,
                    help="[oracle] also inject the phase's GT caption (e.g. "
                         "'current_action') as reasoning_text — the dynamic-gtcap arm")
    ap.add_argument("--instructor-url", default=None,
                    help="[oracle] Qwen instructor server URL (e.g. http://localhost:8001); "
                         "generates a scene-specific caption per phase from the current "
                         "cameras — the dynamic-instructor arm (overrides the GT caption)")
    ap.add_argument("--stuck-window", type=int, default=80)
    ap.add_argument("--stuck-eps-m", type=float, default=0.01)
    ap.add_argument("--snap", type=int, default=0,
                    help="save a frame every N steps (0 = off)")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--log-state", action="store_true")
    ap.add_argument("--instruction-override", default=None)
    ap.add_argument("--obs-keys", choices=["i1b", "demian"], default="i1b",
                    help="camera key convention of the policy server: 'i1b' for the "
                         "original openpi trees (image/wrist_image/image_right), "
                         "'demian' for this repo's vendored openpi (agentview_*)")
    args = ap.parse_args()

    session = Path(args.session_dir).expanduser()
    session.mkdir(parents=True, exist_ok=True)
    (session / "args.json").write_text(json.dumps(vars(args), indent=1))

    env = gym.make(f"robocasa/{args.task}", split="target",
                   camera_heights=256, camera_widths=256)
    client = R.RoboCasaClient(args.host, args.port,
                              open_loop_horizon=args.open_loop_horizon)
    if args.obs_keys == "i1b":
        _adapt_client_keys(client)
    instructor = R.QwenInstructor(args.instructor_url) if args.instructor_url else None

    results = []
    for i in range(args.episodes):
        ep_dir = session / f"ep{i:02d}"
        done_marker = ep_dir / "result.json"
        if done_marker.exists():  # resume: skip episodes completed by a prior run
            results.append(json.loads(done_marker.read_text()))
            print(f"[{args.task}] episode {i} already done, skipping", flush=True)
            continue
        print(f"[{args.task}] episode {i} (seed {args.seed + i}, mode {args.mode}) "
              f"-> {ep_dir}", flush=True)
        res = run_guided_episode(
            env, client, args.task, args.mode, args.seed + i, ep_dir,
            max_steps=args.max_steps, tick=args.tick, block=args.block,
            wait_timeout=args.wait_timeout, switch_buffer=args.switch_buffer,
            switch_on_receptive=args.switch_on_receptive,
            receptive_eps=args.receptive_eps, receptive_window=args.receptive_window,
            max_defer=args.max_defer,
            oracle_caption_key=args.oracle_caption_key,
            instructor=instructor,
            stuck_window=args.stuck_window, stuck_eps_m=args.stuck_eps_m,
            snap=args.snap, record_video=args.video, log_state=args.log_state,
            instruction_override=args.instruction_override,
        )
        results.append(res)
        print(f"  success={res['success']} steps={res['steps']} "
              f"reached={res['reached']} switches={len(res['switches'])}", flush=True)

    (session / "results.json").write_text(json.dumps(results, indent=1))
    n = len(results)
    sr = sum(r["success"] for r in results) / n if n else 0.0
    print(f"\n== {args.task} mode={args.mode}: SR {sr:.2f} ({n} eps) ==", flush=True)
    env.close()


if __name__ == "__main__":
    main()
