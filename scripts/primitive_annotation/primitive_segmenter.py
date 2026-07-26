#!/usr/bin/env python3
"""Segment an atomic-subtask RoboCasa episode into MOTION PRIMITIVES.

Below the atomic-task level (`composite_split` = per-atomic-subtask) lies a
primitive level: reach / grasp / lift / transport / lower / release / retract
for pick-place, and turn / open / close / press for articulated subtasks. Each
primitive is a (segment, instruction) training pair — the granularity an S2
instructor actually operates at, and the level S1 was never trained to follow
(the "commitment" bottleneck: S1 ignores mid-motion re-instruction because every
training prompt spanned a whole atomic subtask).

Boundaries are anchored on GRIPPER EVENTS (semantically unambiguous: open->close
= grasp, close->open = release), refined inside the holding phase by EEF
vertical-motion turning points (lift vs carry vs lower). No VLM needed for
boundaries — the VLM only writes the instruction (see the captioner).

Action layout (raw 12-dim): [base(0:4), mode(4), eef_pos(5:8), eef_rot(8:11), grip(11)].
State layout (16-dim): [base_pos(0:3), base_quat(3:7), eef_pos_rel(7:10),
                        eef_quat(10:14), grip_qpos(14:16)].
"""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

GRIP_IDX = 11          # action gripper channel (+1 close .. -1 open)
EEF_Z_STATE = 9        # observation.state eef relative z
EEF_XYZ_STATE = slice(7, 10)


@dataclass
class Primitive:
    kind: str          # reach|grasp|lift|transport|lower|release|retract|articulate
    start: int
    end: int           # inclusive
    holding: bool      # is an object grasped during this segment
    dz: float          # net vertical EEF displacement over the segment (m)
    dxy: float         # net horizontal EEF displacement over the segment (m)


def _gripper_events(grip: np.ndarray, hi: float = 0.5, lo: float = -0.5) -> list[tuple[int, str]]:
    """Hysteretic gripper state machine -> [(frame, 'close'|'open'), ...]."""
    events, state = [], "open" if grip[0] < 0 else "closed"
    for t in range(1, len(grip)):
        if state == "open" and grip[t] > hi:
            events.append((t, "close")); state = "closed"
        elif state == "closed" and grip[t] < lo:
            events.append((t, "open")); state = "open"
    return events


def _turning_points(z: np.ndarray, lo: int, hi: int, min_span: float = 0.03) -> list[int]:
    """Interior frames in (lo,hi) where EEF vertical motion reverses direction
    by at least `min_span` metres — splits lift / carry / lower."""
    if hi - lo < 8:
        return []
    seg = z[lo:hi + 1]
    # smooth, then find sign changes of the vertical velocity with amplitude gate
    v = np.convolve(np.diff(seg), np.ones(5) / 5, mode="same")
    pts, last = [], 0
    for i in range(1, len(v)):
        if np.sign(v[i]) != np.sign(v[i - 1]) and v[i] != 0:
            if abs(seg[i] - seg[last]) >= min_span:
                pts.append(lo + i); last = i
    return pts


def segment_episode(actions: np.ndarray, states: np.ndarray) -> list[Primitive]:
    T = len(actions)
    grip = actions[:, GRIP_IDX]
    z = states[:, EEF_Z_STATE]
    xyz = states[:, EEF_XYZ_STATE]
    events = _gripper_events(grip)

    def feats(a: int, b: int) -> tuple[float, float]:
        dz = float(z[b] - z[a])
        dxy = float(np.linalg.norm(xyz[b, :2] - xyz[a, :2]))
        return dz, dxy

    prims: list[Primitive] = []
    if not any(e[1] == "close" for e in events):
        # No grasp anywhere: an articulated subtask (turn knob / open / close /
        # press). One segment; the caption disambiguates the action.
        dz, dxy = feats(0, T - 1)
        return [Primitive("articulate", 0, T - 1, False, dz, dxy)]

    grasp = next(t for t, k in events if k == "close")
    release = next((t for t, k in events if k == "open" and t > grasp), T - 1)

    # reach: start -> grasp
    dz, dxy = feats(0, max(1, grasp - 1))
    prims.append(Primitive("reach", 0, grasp - 1, False, dz, dxy))
    # grasp: a short window around the close event
    g0, g1 = max(0, grasp - 3), min(T - 1, grasp + 3)
    dz, dxy = feats(g0, g1)
    prims.append(Primitive("grasp", g0, g1, True, dz, dxy))

    # holding phase grasp..release: split by vertical turning points
    tps = _turning_points(z, grasp + 3, release)
    bounds = [grasp + 3] + tps + [release]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 4:
            continue
        dz, dxy = feats(a, b)
        if dz > 0.04:
            kind = "lift"
        elif dz < -0.04:
            kind = "lower"
        else:
            kind = "transport"
        prims.append(Primitive(kind, a, b, True, dz, dxy))

    # release + retract
    r0, r1 = max(0, release - 3), min(T - 1, release + 3)
    dz, dxy = feats(r0, r1)
    prims.append(Primitive("release", r0, r1, False, dz, dxy))
    if r1 < T - 5:
        dz, dxy = feats(r1, T - 1)
        prims.append(Primitive("retract", r1, T - 1, False, dz, dxy))
    return prims


def load_episode(lerobot_dir: str, ep_idx: int):
    chunk = ep_idx // 1000
    pq = f"{lerobot_dir}/data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
    df = pd.read_parquet(pq)
    actions = np.stack(df["action"].to_numpy())
    states = np.stack(df["observation.state"].to_numpy())
    ep = None
    for line in open(f"{lerobot_dir}/meta/episodes.jsonl"):
        r = json.loads(line)
        if r["episode_index"] == ep_idx:
            ep = r; break
    instr = (ep or {}).get("tasks", [""])[0]
    return actions, states, instr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-dir", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_out = []
    for ep in args.episodes:
        actions, states, instr = load_episode(args.lerobot_dir, ep)
        prims = segment_episode(actions, states)
        print(f"\nep{ep} ({len(actions)} frames)  subtask: {instr!r}")
        for p in prims:
            print(f"  [{p.start:4d}-{p.end:4d}] {p.kind:11s} "
                  f"hold={int(p.holding)} dz={p.dz:+.3f} dxy={p.dxy:.3f}")
        all_out.append({"episode": ep, "subtask": instr, "n_frames": len(actions),
                        "primitives": [asdict(p) for p in prims]})
    if args.out:
        Path(args.out).write_text(json.dumps(all_out, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
