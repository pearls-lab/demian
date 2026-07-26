#!/usr/bin/env python3
"""Transfer episode-level captions from source composite_atomic_split roots to
their Method-1 aug clips, via the source mapping recorded by the patched
builder (episodes.jsonl: source_episode_index / source_offset).

Each aug clip is a head-trimmed slice (offset <= 12 frames) of one source
episode, so the source episode's captions describe the clip faithfully.

Usage:
    python transfer_captions_to_aug.py \
        --src-base ~/orchest_data/robocasa_v1.0/composite_atomic_split \
        --aug-base ~/orchest_data/robocasa_v1.0/composite_atomic_split_aug_v2 \
        --model-slug Qwen3-VL-30B-A3B-Instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_captions(caps_file: Path) -> dict[int, dict]:
    out = {}
    if caps_file.exists():
        for line in caps_file.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[int(rec["episode_index"])] = rec.get("annotations", {})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-base", required=True)
    ap.add_argument("--aug-base", required=True)
    ap.add_argument("--model-slug", default="Qwen3-VL-30B-A3B-Instruct")
    args = ap.parse_args()
    src_base = Path(args.src_base).expanduser()
    aug_base = Path(args.aug_base).expanduser()

    n_roots = n_done = n_eps = n_missing = 0
    missing_roots = []
    for aug_root in sorted(aug_base.glob("*/*/lerobot")):
        n_roots += 1
        task, phase = aug_root.parts[-3], aug_root.parts[-2]
        src_candidates = sorted((src_base / task / phase).glob("*/lerobot"))
        if not src_candidates:
            missing_roots.append(f"{task}/{phase}: no source root")
            continue
        src_caps = load_captions(
            src_candidates[0] / "meta" / f"captions_{args.model_slug}.jsonl")
        if not src_caps:
            missing_roots.append(f"{task}/{phase}: source has no captions yet")
            continue

        lines = []
        for line in (aug_root / "meta" / "episodes.jsonl").read_text().splitlines():
            ep = json.loads(line)
            sei = ep.get("source_episode_index")
            if sei is None:
                raise SystemExit(
                    f"{aug_root}: episodes.jsonl lacks source_episode_index — "
                    "built with the unpatched builder?")
            ann = src_caps.get(int(sei))
            if ann is None:
                n_missing += 1
                continue
            lines.append(json.dumps(
                {"episode_index": ep["episode_index"], "annotations": ann,
                 "source_episode_index": int(sei)}))
            n_eps += 1
        (aug_root / "meta" / f"captions_{args.model_slug}.jsonl").write_text(
            "\n".join(lines) + "\n")
        n_done += 1

    print(f"roots: {n_done}/{n_roots} transferred, {n_eps} clip captions written, "
          f"{n_missing} clips without a source caption")
    for m in missing_roots[:10]:
        print("  MISSING:", m)
    if len(missing_roots) > 10:
        print(f"  ... and {len(missing_roots) - 10} more")


if __name__ == "__main__":
    main()
