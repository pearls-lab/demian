#!/usr/bin/env python
"""Paired McNemar analysis over powered s2_guided_eval runs.

Usage:
    python s2_paired_analysis.py <root> --arms oracle gtcap instructor

Expects <root>/<task>_<arm>/results.json. Pairs episodes by (task, seed) and runs
an exact McNemar test (binomial on discordant pairs) for every arm pair, on
(a) strict success and (b) each milestone index reached ("sg1", "sg2" = first /
all milestone keys latched).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n * 2
    return min(1.0, p)


def load_arm(root: Path, task: str, arm: str) -> dict[tuple, dict]:
    f = root / f"{task}_{arm}" / "results.json"
    if not f.exists():
        return {}
    out = {}
    for r in json.loads(f.read_text()):
        keys = list((r.get("reached") or {}).keys())
        fired = [k for k in keys if r["reached"].get(k)]
        out[(task, r["seed"])] = {
            "success": bool(r["success"]),
            "sg1": len(fired) >= 1,
            "sg_all": len(keys) > 0 and len(fired) == len(keys),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--arms", nargs="+", default=["oracle", "gtcap", "instructor"])
    args = ap.parse_args()
    root = Path(args.root).expanduser()

    tasks = sorted({d.name.rsplit("_", 1)[0] for d in root.iterdir()
                    if d.is_dir() and d.name.rsplit("_", 1)[-1] in args.arms})
    data = {arm: {} for arm in args.arms}
    for arm in args.arms:
        for task in tasks:
            data[arm].update(load_arm(root, task, arm))

    print(f"tasks: {len(tasks)}  " +
          "  ".join(f"{a}: n={len(data[a])}" for a in args.arms))

    # Per-arm marginals.
    for metric in ("success", "sg1", "sg_all"):
        print(f"\n== {metric} ==")
        for arm in args.arms:
            vals = [v[metric] for v in data[arm].values()]
            if vals:
                print(f"  {arm:12s} {sum(vals):3d}/{len(vals)} = {sum(vals)/len(vals):.3f}")
        for a1, a2 in itertools.combinations(args.arms, 2):
            common = sorted(set(data[a1]) & set(data[a2]))
            if not common:
                continue
            b = sum(1 for k in common if data[a1][k][metric] and not data[a2][k][metric])
            c = sum(1 for k in common if not data[a1][k][metric] and data[a2][k][metric])
            p = mcnemar_exact(b, c)
            diff = (sum(data[a2][k][metric] for k in common)
                    - sum(data[a1][k][metric] for k in common)) / len(common)
            print(f"  {a1} vs {a2}: n={len(common)} paired, "
                  f"discordant {b}/{c}, delta={diff:+.3f}, McNemar p={p:.4f}")

    # Per-task success table.
    print("\n== per-task success ==")
    print(f"{'task':26s}" + "".join(f"{a:>12s}" for a in args.arms))
    for task in tasks:
        row = f"{task:26s}"
        for arm in args.arms:
            vals = [v["success"] for k, v in data[arm].items() if k[0] == task]
            row += f"{sum(vals):>5d}/{len(vals):<6d}" if vals else f"{'--':>12s}"
        print(row)


if __name__ == "__main__":
    main()
