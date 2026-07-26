#!/usr/bin/env python
"""Does System 1 actually understand the instruction? — behavioral probes.

Two experiments, both anchored on the *handoff-entry state* (the OOD state the
policy lands in after finishing subgoal 1, where the composite cliff lives).

  --exp discriminate
      From one identical sim state, run each instruction of a battery R times
      (fresh flow-matching noise each time) for --probe-steps steps and record
      the EEF trajectory. Compares BETWEEN-instruction separation against
      WITHIN-instruction spread. If between <= within, the policy's behavior is
      independent of the instruction text — it does not "understand" it, it is
      running a state-conditioned prior. Includes nonsense/empty controls, so
      the null distribution is measured, not assumed. Also reports
      target-directedness: cosine between the realized EEF displacement and the
      direction to the fixture named in the instruction.

  --exp recovery
      The intervention test. From the same handoff-entry state, paired arms:
        direct : canonical subgoal-2 instruction for the whole budget
        reposi : a repositioning instruction ("move your arm to the <target>")
                 for --prefix-steps, then the canonical subgoal-2 instruction
      Scores subgoal-2 milestone attainment. Tests the hypothesis that a simple
      repositioning instruction pulls the arm from an OOD state back into an
      in-domain one from which the next atomic skill fires.

Run with the robocasa eval python + MUJOCO_GL=egl, against serve_policy.py.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import gymnasium as gym
import robocasa  # noqa: F401

import run_robocasa_eval as R
import composite_subgoals as CS
from s2_guided_eval import _CAMS, _adapt_client_keys, _save_frame

# Fixtures we can resolve a world position for, to score target-directedness.
_FIXTURE_ATTRS = ["coffee_machine", "sink", "stove", "cab", "counter", "drawer",
                  "blender", "microwave", "dishwasher", "toaster", "fridge"]


def _fixture_positions(raw) -> dict[str, np.ndarray]:
    out = {}
    for name in _FIXTURE_ATTRS:
        fx = getattr(raw, name, None)
        if fx is None:
            continue
        for getter in ("pos", "_pos"):
            p = getattr(fx, getter, None)
            if p is not None:
                try:
                    out[name] = np.asarray(p, dtype=np.float64)[:3]
                except Exception:
                    pass
                break
    return out


def _eef(obs: dict) -> np.ndarray:
    return np.asarray(obs["state.end_effector_position_relative"], np.float64)


def _eef_world(raw) -> np.ndarray:
    """Gripper world position — the frame fixture positions live in, so
    target-directedness is computed without mixing frames."""
    for site in ("gripper0_right_grip_site", "gripper0_grip_site", "robot0_right_hand"):
        try:
            return np.asarray(raw.sim.data.get_site_xpos(site), np.float64).copy()
        except Exception:
            pass
    try:
        bid = raw.sim.model.body_name2id("robot0_right_hand")
        return np.asarray(raw.sim.data.body_xpos[bid], np.float64).copy()
    except Exception:
        return np.full(3, np.nan)


def _reset_to(env, seed: int, flat_state: np.ndarray | None, ep_meta: dict | None = None):
    """Restore the exact harvested scene+state.

    env.reset(seed=...) alone does NOT reliably reproduce the same scene — the
    layout/object set can be re-sampled, changing the flattened-state length.
    Pinning `ep_meta` before reset is RoboCasa's supported way to reproduce a
    scene exactly (same mechanism the demo-replay eval stack uses); rebuilding
    from a captured model XML is NOT safe, because the env's Python-side fixture
    references still point at the freshly sampled scene.

    Every arm of a probe must start from a byte-identical state, so a residual
    mismatch is raised rather than silently tolerated.
    """
    raw = env.unwrapped
    if ep_meta is not None:
        raw.set_ep_meta(ep_meta)
    obs, _ = env.reset(seed=seed)
    if flat_state is not None:
        if len(raw.sim.get_state().flatten()) != len(flat_state):
            raise RuntimeError(
                "sim state size differs from harvest even with ep_meta pinned "
                f"({len(raw.sim.get_state().flatten())} vs {len(flat_state)})")
        raw.sim.set_state_from_flattened(flat_state)
        raw.sim.forward()
        if hasattr(raw, "update_state"):
            raw.update_state()
        raw.sim.set_state_from_flattened(flat_state)  # settle
        raw.sim.forward()
        if hasattr(raw, "update_state"):
            raw.update_state()
        obs = _recon_obs(raw, obs)
    return obs


def _recon_obs(raw, template: dict) -> dict:
    """Re-render the gym-format obs after a state teleport."""
    from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter as KC
    robo = raw._get_observations(force_update=True)
    obs = dict(KC.map_obs_in_eval(robo))
    for vid in _CAMS:
        cam = vid.replace("video.", "")
        obs[vid] = np.asarray(robo[cam + "_image"])[::-1]
    return obs


def roll(env, client, obs, instruction, steps, checker=None, keys=None,
         reset_ms=None, reasoning=None) -> dict:
    """Run `steps` closed-loop steps under one instruction. Returns trace."""
    client.reset()
    raw = env.unwrapped
    traj = [_eef(obs)]
    wtraj = [_eef_world(raw)]
    reached = {k: False for k in (keys or [])}
    for t in range(steps):
        ret = client.infer(obs, instruction, reasoning_text=reasoning)
        obs, _r, done, trunc, _info = env.step(ret["action_dict"])
        traj.append(_eef(obs))
        wtraj.append(_eef_world(raw))
        if checker is not None:
            d = checker(raw)
            for k in keys:
                if d[k] and not (reset_ms or {}).get(k, False):
                    reached[k] = True
        if done or trunc:
            break
    traj, wtraj = np.stack(traj), np.stack(wtraj)
    return {"traj": traj, "disp": traj[-1] - traj[0], "obs": obs,
            "wstart": wtraj[0], "wdisp": wtraj[-1] - wtraj[0],
            "path_len": float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()),
            "reached": reached}


def harvest_reset_state(env, task, seed, out_dir) -> dict:
    """The IN-DISTRIBUTION anchor: the episode's own reset state, which every
    demo starts from. Paired against the handoff anchor, this isolates whether
    instruction sensitivity degrades once the policy is off-distribution."""
    raw = env.unwrapped
    plan = CS.get_subgoal_plan(task)
    phases = CS.resolve_subgoal_runtime(plan, None, "", None, "prompt")
    checker = CS.get_subgoal_checker(task)
    obs, _ = env.reset(seed=seed)
    _save_frame(obs, out_dir / f"reset_seed{seed}.png")
    return {"seed": seed, "step": 0,
            "flat_state": np.asarray(raw.sim.get_state().flatten()),
            "ep_meta": raw.get_ep_meta(),
            "reset_ms": {k: bool(v) for k, v in checker(raw).items()},
            "keys": list(checker(raw).keys()),
            "next_instruction": phases[1]["instruction"],
            "next_name": phases[1]["name"]}


def harvest_handoff_state(env, client, task, seed, max_steps, out_dir,
                          want: str = "handoff") -> dict | None:
    """Roll the canonical subgoal-1 instruction until its milestone fires.

    want="handoff"  -> state at the trigger = the OOD handoff-entry state.
    want="mid_sg1"  -> state at ~half the trigger step: still MID-subgoal-1, so
                       in-distribution, but with an extended arm mid-motion.
                       This is the control that separates "off-distribution"
                       from "unusual arm pose" in the sensitivity comparison.
    """
    raw = env.unwrapped
    plan = CS.get_subgoal_plan(task)
    phases = CS.resolve_subgoal_runtime(plan, None, "", None, "prompt")
    checker = CS.get_subgoal_checker(task)
    obs, _ = env.reset(seed=seed)
    keys = list(checker(raw).keys())
    reset_ms = {k: bool(v) for k, v in checker(raw).items()}
    client.reset()
    instr = phases[0]["instruction"]
    history: list[tuple[int, np.ndarray, dict]] = []
    for t in range(max_steps):
        if want == "mid_sg1" and t % 5 == 0:
            history.append((t, np.asarray(raw.sim.get_state().flatten()),
                            raw.get_ep_meta()))
        ret = client.infer(obs, instr)
        obs, _r, done, trunc, _i = env.step(ret["action_dict"])
        try:
            fired = bool(phases[0]["trigger_fn"](raw))
        except Exception:
            fired = False
        if fired and want == "mid_sg1":
            if not history:
                return None
            mid_t, mid_state, mid_meta = min(
                history, key=lambda h: abs(h[0] - t // 2))
            return {"seed": seed, "step": mid_t, "flat_state": mid_state,
                    "ep_meta": mid_meta, "reset_ms": reset_ms, "keys": keys,
                    "next_instruction": phases[1]["instruction"],
                    "next_name": phases[1]["name"], "trigger_step": t}
        if fired:
            flat = np.asarray(raw.sim.get_state().flatten())
            _save_frame(obs, out_dir / f"handoff_seed{seed}_t{t}.png")
            return {"seed": seed, "step": t, "flat_state": flat,
                    "ep_meta": raw.get_ep_meta(),
                    "reset_ms": reset_ms, "keys": keys,
                    "next_instruction": phases[1]["instruction"],
                    "next_name": phases[1]["name"]}
        if done or trunc:
            break
    return None


def build_battery(task: str, next_instruction: str, fixtures: list[str]) -> list[dict]:
    """Instruction battery: in-distribution, OOD-repositioning, and controls."""
    b = [
        {"id": "canonical_sg2", "kind": "in_dist", "text": next_instruction, "target": None},
        {"id": "canonical_sg1", "kind": "in_dist",
         "text": CS.resolve_subgoal_runtime(CS.get_subgoal_plan(task), None, "", None,
                                            "prompt")[0]["instruction"], "target": None},
    ]
    for fx in fixtures:
        pretty = fx.replace("_", " ")
        b.append({"id": f"move_to_{fx}", "kind": "ood_reposition",
                  "text": f"move your arm to the {pretty}", "target": fx})
    b += [
        {"id": "lift_up", "kind": "ood_reposition",
         "text": "lift your arm up above the counter", "target": None},
        {"id": "nonsense", "kind": "control", "text": "xyzzy plugh frobnicate", "target": None},
        {"id": "empty", "kind": "control", "text": "", "target": None},
    ]
    return b


def exp_discriminate(env, client, st, task, args, out_dir) -> dict:
    raw = env.unwrapped
    fx = _fixture_positions(raw)
    battery = build_battery(task, st["next_instruction"], list(fx)[:args.max_fixtures])
    print(f"  battery: {len(battery)} instructions x {args.repeats} repeats "
          f"x {args.probe_steps} steps; fixtures={list(fx)}", flush=True)

    runs: dict[str, list[dict]] = {}
    for instr in battery:
        runs[instr["id"]] = []
        for r in range(args.repeats):
            obs = _reset_to(env, st["seed"], st["flat_state"], st.get("ep_meta"))
            base_eef_world = None
            out = roll(env, client, obs, instr["text"], args.probe_steps)
            runs[instr["id"]].append({"disp": out["disp"], "traj": out["traj"],
                                      "wstart": out["wstart"], "wdisp": out["wdisp"],
                                      "path_len": out["path_len"]})
            print(f"    [{instr['id']:>22s} r{r}] disp={np.round(out['disp'],3)} "
                  f"path={out['path_len']:.2f}", flush=True)

    # Within- vs between-instruction separation on the final displacement.
    def _d(a, b):
        return float(np.linalg.norm(a - b))

    within = [ _d(x["disp"], y["disp"])
               for v in runs.values() for x, y in itertools.combinations(v, 2) ]
    between = [ _d(x["disp"], y["disp"])
                for i1, i2 in itertools.combinations(runs, 2)
                for x in runs[i1] for y in runs[i2] ]
    w, b = float(np.mean(within)), float(np.mean(between))

    # Target-directedness, entirely in world frame: cosine between the realized
    # gripper displacement and the vector from its start pose to the named
    # fixture. Reported for EVERY instruction against EVERY fixture, so a named
    # target can be compared against the same fixture under other instructions
    # (the state prior's own drift is the baseline to beat).
    directed = {}
    for iid, v in runs.items():
        row = {}
        for fname, fpos in fx.items():
            cs = []
            for run in v:
                d, s = run["wdisp"], run["wstart"]
                ref = fpos[:3] - s
                if not np.all(np.isfinite(d)) or np.linalg.norm(d) < 1e-6 \
                        or np.linalg.norm(ref) < 1e-6:
                    continue
                cs.append(float(np.dot(d, ref) / (np.linalg.norm(d) * np.linalg.norm(ref))))
            if cs:
                row[fname] = float(np.mean(cs))
        if row:
            directed[iid] = row

    res = {
        "task": task, "seed": st["seed"], "handoff_step": st["step"],
        "probe_steps": args.probe_steps, "repeats": args.repeats,
        "within_mean_m": w, "between_mean_m": b, "ratio_between_within": b / w if w else None,
        "per_instruction": {
            k: {"mean_disp": np.mean([r["disp"] for r in v], axis=0).round(4).tolist(),
                "disp_norm": float(np.mean([np.linalg.norm(r["disp"]) for r in v])),
                "path_len": float(np.mean([r["path_len"] for r in v]))}
            for k, v in runs.items()},
        "target_directedness": directed,
        "battery": battery,
    }
    (out_dir / f"discriminate_{task}_s{st['seed']}.json").write_text(
        json.dumps(res, indent=1, default=float))
    print(f"  WITHIN {w:.4f} m   BETWEEN {b:.4f} m   ratio {b/w if w else 0:.2f}", flush=True)
    return res


def exp_recovery(env, client, st, task, args, out_dir) -> dict:
    checker = CS.get_subgoal_checker(task)
    keys, reset_ms = st["keys"], st["reset_ms"]
    fx = _fixture_positions(env.unwrapped)
    targets = list(fx)[:args.max_fixtures]
    arms = [{"id": "direct", "prefix": None}]
    for t in targets:
        arms.append({"id": f"reposition_{t}",
                     "prefix": f"move your arm to the {t.replace('_',' ')}"})
    arms.append({"id": "reposition_lift", "prefix": "lift your arm up above the counter"})

    results = {}
    for arm in arms:
        outs = []
        for r in range(args.repeats):
            obs = _reset_to(env, st["seed"], st["flat_state"], st.get("ep_meta"))
            if arm["prefix"]:
                pre = roll(env, client, obs, arm["prefix"], args.prefix_steps,
                           checker, keys, reset_ms)
                obs = pre["obs"]
            budget = args.recovery_steps - (args.prefix_steps if arm["prefix"] else 0)
            out = roll(env, client, obs, st["next_instruction"], budget,
                       checker, keys, reset_ms)
            got = bool(out["reached"].get(st["next_name"], False)) or \
                all(out["reached"].get(k, False) for k in keys)
            outs.append({"reached": out["reached"], "sg2": got})
            print(f"    [{arm['id']:>22s} r{r}] sg2={got} reached={out['reached']}",
                  flush=True)
        results[arm["id"]] = {
            "n": len(outs),
            "sg2_rate": sum(o["sg2"] for o in outs) / len(outs),
            "runs": outs,
        }
    res = {"task": task, "seed": st["seed"], "handoff_step": st["step"],
           "next_instruction": st["next_instruction"], "arms": results}
    (out_dir / f"recovery_{task}_s{st['seed']}.json").write_text(
        json.dumps(res, indent=1, default=float))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--exp", choices=["discriminate", "recovery", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8211)
    ap.add_argument("--obs-keys", choices=["i1b", "demian"], default="demian")
    ap.add_argument("--open-loop-horizon", type=int, default=8)
    ap.add_argument("--anchor", choices=["handoff", "reset", "mid_sg1"], default="handoff",
                    help="state the probe runs from: 'handoff' = the OOD state after "
                         "subgoal 1 fires; 'reset' = the in-distribution episode start")
    ap.add_argument("--harvest-steps", type=int, default=400)
    ap.add_argument("--probe-steps", type=int, default=40)
    ap.add_argument("--prefix-steps", type=int, default=60)
    ap.add_argument("--recovery-steps", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-fixtures", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=1))

    env = gym.make(f"robocasa/{args.task}", split="target",
                   camera_heights=256, camera_widths=256)
    client = R.RoboCasaClient(args.host, args.port,
                              open_loop_horizon=args.open_loop_horizon)
    if args.obs_keys == "i1b":
        _adapt_client_keys(client)

    summary = []
    for seed in args.seeds:
        print(f"[{args.task}] seed {seed}: harvesting {args.anchor} state...", flush=True)
        st = (harvest_reset_state(env, args.task, seed, out_dir)
              if args.anchor == "reset"
              else harvest_handoff_state(env, client, args.task, seed,
                                         args.harvest_steps, out_dir,
                                         want=args.anchor))
        if st is None:
            print(f"  seed {seed}: subgoal-1 never reached; skipping", flush=True)
            continue
        print(f"  handoff at step {st['step']}; next = {st['next_instruction']!r}", flush=True)
        entry = {"seed": seed, "handoff_step": st["step"]}
        if args.exp in ("discriminate", "both"):
            entry["discriminate"] = exp_discriminate(env, client, st, args.task, args, out_dir)
        if args.exp in ("recovery", "both"):
            entry["recovery"] = exp_recovery(env, client, st, args.task, args, out_dir)
        summary.append(entry)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    env.close()

    for e in summary:
        if "discriminate" in e:
            d = e["discriminate"]
            print(f"\nseed {e['seed']} DISCRIMINATE: within {d['within_mean_m']:.4f} "
                  f"between {d['between_mean_m']:.4f} ratio {d['ratio_between_within']:.2f}")
        if "recovery" in e:
            print(f"seed {e['seed']} RECOVERY: " + "  ".join(
                f"{k}={v['sg2_rate']:.2f}" for k, v in e["recovery"]["arms"].items()))


if __name__ == "__main__":
    main()
