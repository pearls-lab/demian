#!/usr/bin/env python
"""Summarize s2_guided_eval session dirs into a failure-analysis report.

Usage:
    python s2_session_report.py <session_dir> [<session_dir> ...] [--md out.md]

For each episode: outcome, per-milestone first-fire, instruction switch history,
gripper open/close transitions (grasp attempts), motion profile, and a coarse
failure classification:

  success            strict env success
  sg1_never          first milestone never reached (lost in subgoal 1)
  sgK_handoff        milestone k reached but k+1 never (the handoff cliff)
  grasp_loop         >=2 close/open cycles without the next milestone
  empty_transport    gripper closed >=100 steps with no milestone change
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def grip_transitions(rows: list[dict]) -> list[tuple[str, int]]:
    out, state = [], "open"
    for r in rows:
        g = r["grip"][0]
        if state == "open" and g < 0.005:
            out.append(("close", r["t"])); state = "closed"
        elif state == "closed" and g > 0.03:
            out.append(("open", r["t"])); state = "open"
    return out


def classify(res: dict, grips: list[tuple[str, int]], steps: int) -> str:
    if res.get("success"):
        return "success"
    first = res.get("first_fire") or {}
    keys = list(first.keys())
    fired = [k for k in keys if first[k] >= 0]
    closes = [t for s, t in grips if s == "close"]
    if not fired:
        if len(closes) >= 2:
            return "grasp_loop"
        # closed for a long tail without any milestone -> empty transport
        if grips and grips[-1][0] == "close" and steps - grips[-1][1] >= 100:
            return "empty_transport"
        return "sg1_never"
    if len(fired) < len(keys):
        k = len(fired)
        return f"sg{k + 1}_handoff"
    return "milestones_but_strict_fail"


def episode_report(ep_dir: Path) -> dict | None:
    res_p = ep_dir / "result.json"
    events = _load_jsonl(ep_dir / "events.jsonl")
    if res_p.exists():
        res = json.loads(res_p.read_text())
    else:  # crashed episode: reconstruct what we can from events
        start = next((e for e in events if e.get("event") == "start"), None)
        end = next((e for e in reversed(events) if e.get("event") == "end"), None)
        if start is None:
            return None
        ms = [e for e in events if e.get("event") == "milestone"]
        res = {
            "task": start.get("task"), "seed": start.get("seed"),
            "mode": start.get("mode"), "success": bool(end and end.get("success")),
            "steps": (end or events[-1]).get("t", -1) + 1,
            "first_fire": {m["name"]: m["t"] for m in ms},
            "switches": [e for e in events
                         if e.get("event") in ("oracle_switch", "s2_switch")],
            "partial": True,
        }
    state = _load_jsonl(ep_dir / "state_log.jsonl")
    grips = grip_transitions(state) if state else []
    res["grip_transitions"] = grips
    res["failure_class"] = classify(res, grips, res.get("steps", 0))
    res["stuck_events"] = [e["t"] for e in events if e.get("event") == "stuck"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    for sess in args.sessions:
        sess = Path(sess).expanduser()
        eps = sorted(d for d in sess.glob("ep*") if d.is_dir())
        if not eps:
            continue
        emit(f"\n## {sess.name}")
        emit("| ep | seed | mode | outcome | class | steps | milestones (first-fire) | switches | grasp closes |")
        emit("|---|---|---|---|---|---|---|---|---|")
        for ep in eps:
            r = episode_report(ep)
            if r is None:
                continue
            ms = ", ".join(f"{k}@{v}" for k, v in (r.get("first_fire") or {}).items()
                           if v is not None and v >= 0) or "none"
            closes = [t for s, t in r["grip_transitions"] if s == "close"]
            emit(f"| {ep.name} | {r.get('seed')} | {r.get('mode')} "
                 f"| {'SUCCESS' if r.get('success') else 'fail'} | {r['failure_class']} "
                 f"| {r.get('steps')} | {ms} | {len(r.get('switches') or []) } "
                 f"| {len(closes)}{' @' + str(closes) if closes else ''} |")

    if args.md:
        Path(args.md).expanduser().write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
