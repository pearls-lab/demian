#!/usr/bin/env python
"""Aggregate s2_instruction_probe outputs into a readable verdict.

Usage: python s2_probe_report.py <probe_dir> [<probe_dir> ...]

Discrimination: per seed, reports the between/within displacement ratio, and —
the part that actually answers "does it understand?" — how far each instruction
sits from the *nonsense/empty controls*. An instruction that lands on top of the
controls produced no behavioral effect; one that lands far from them did.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONTROLS = ("nonsense", "empty")


def report_discriminate(files: list[Path]) -> None:
    rows, per_instr = [], {}
    for f in files:
        d = json.loads(f.read_text())
        pi = d["per_instruction"]
        ctrl = [np.asarray(pi[c]["mean_disp"]) for c in CONTROLS if c in pi]
        if not ctrl:
            continue
        cmean = np.mean(ctrl, axis=0)
        within = d["within_mean_m"]
        rows.append((d["seed"], within, d["between_mean_m"], d["ratio_between_within"]))
        for k, v in pi.items():
            dist = float(np.linalg.norm(np.asarray(v["mean_disp"]) - cmean))
            # z-score the control-distance by the within-instruction spread:
            # >1 means the instruction moved the policy further than noise does.
            per_instr.setdefault(k, []).append(dist / within if within else np.nan)

    if not rows:
        return
    print(f"  seeds: {len(rows)}   ratio between/within: " +
          ", ".join(f"{r[3]:.2f}" for r in rows) +
          f"   (mean {np.mean([r[3] for r in rows]):.2f})")
    print(f"  {'instruction':>24s}  dist-from-control / within-noise  (>1 = real effect)")
    for k, v in sorted(per_instr.items(), key=lambda kv: -np.mean(kv[1])):
        vals = np.asarray(v, dtype=float)
        flag = "  <== controls" if k in CONTROLS else ""
        print(f"  {k:>24s}  mean {np.nanmean(vals):5.2f}   per-seed "
              f"[{', '.join(f'{x:.1f}' for x in vals)}]{flag}")


def report_recovery(files: list[Path]) -> None:
    agg: dict[str, list[float]] = {}
    n_seeds = 0
    for f in files:
        d = json.loads(f.read_text())
        n_seeds += 1
        for arm, v in d["arms"].items():
            agg.setdefault(arm, []).append(v["sg2_rate"])
    if not agg:
        return
    base = np.mean(agg.get("direct", [np.nan]))
    print(f"  seeds: {n_seeds}   (sg2 rate per arm, averaged over seeds)")
    for arm, v in sorted(agg.items(), key=lambda kv: -np.mean(kv[1])):
        m = float(np.mean(v))
        delta = "" if arm == "direct" else f"   delta vs direct {m - base:+.3f}"
        print(f"  {arm:>24s}  {m:.3f}{delta}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    for d in args.dirs:
        p = Path(d).expanduser()
        disc = sorted(p.glob("discriminate_*.json"))
        rec = sorted(p.glob("recovery_*.json"))
        if not disc and not rec:
            continue
        print(f"\n=== {p.name} ===")
        if disc:
            print(" [discriminate]")
            report_discriminate(disc)
        if rec:
            print(" [recovery]")
            report_recovery(rec)


if __name__ == "__main__":
    main()
