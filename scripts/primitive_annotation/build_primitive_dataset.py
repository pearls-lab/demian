#!/usr/bin/env python3
"""Combined driver: segment atomic-subtask episodes into motion primitives and
caption each with an instructor-aligned imperative instruction.

Per primitive we send 3 KEYFRAMES (start / mid / end) as images plus a kinematic
hint (dz, dxy, hold) — faster and more robust than per-primitive video clips,
and the motion direction is already in the features. Captions come from the
running vLLM Qwen3-VL server (the instructor's own model family -> alignment by
construction). Output per root: meta/primitives_<slug>.jsonl with one record per
primitive: {episode_index, primitive_idx, kind, start, end, hold, dz, dxy,
subtask, instruction}.

Parallel: shard ROOTS across workers, each worker pinned to one vLLM port.

Run with the robocasa venv (needs cv2 + pandas):
  /data/bkim/envs/robocasa/bin/python build_primitive_dataset.py \
    --split-base ~/orchest_data/robocasa_v1.0/composite_atomic_split \
    --url http://localhost:8301/v1 --worker-id 0 --n-workers 4 --max-eps 25
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from primitive_segmenter import segment_episode, load_episode  # noqa: E402

_KIND_HINT = {
    "reach": "arm moving toward an object to pick up, gripper open",
    "grasp": "gripper closing on an object",
    "lift": "arm lifting the held object upward",
    "transport": "arm carrying the held object horizontally",
    "lower": "arm lowering the held object toward a target",
    "release": "gripper opening to release the object",
    "retract": "arm moving away, empty, after releasing",
    "articulate": "arm operating a fixture (turn / open / close / press)",
}

SYSTEM = (
    "You are a robot supervisor giving a SHORT imperative command for the current "
    "motion segment, exactly as you would tell a robot arm what to do next. "
    'Output JSON {"instruction": "..."} — one lower-case order, 4-10 words, no '
    'trailing period. It must be a COMMAND ("lift the mug up toward the cabinet"), '
    'never a description ("the arm moves up"). Name the object and target when '
    "visible in the frames. Valid JSON only."
)


def _frame_b64(cap, idx: int) -> str | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def _keyframes(video: str, start: int, end: int) -> list[str]:
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (end + 1)
    idxs = [start, (start + end) // 2, min(end, n - 1)]
    out = [b for i in idxs if (b := _frame_b64(cap, min(i, n - 1)))]
    cap.release()
    return out


def caption(url: str, model: str, frames: list[str], kind: str, subtask: str,
            dz: float, dxy: float) -> str:
    hint = _KIND_HINT.get(kind, "")
    direction = ("upward" if dz > 0.04 else "downward" if dz < -0.04 else "level")
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f}"}} for f in frames]
    content.append({"type": "text", "text":
        f"Parent subtask: {subtask}\nThree frames (start, middle, end) of one "
        f"motion segment: {hint}; net vertical motion {direction} "
        f"({dz:+.2f} m), horizontal travel {dxy:.2f} m.\n"
        f"Give the imperative command for THIS segment only."})
    for attempt in range(3):
        try:
            r = requests.post(f"{url}/chat/completions", json={
                "model": model, "temperature": 0.4, "max_tokens": 48,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": content}]}, timeout=90)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip()
            try:
                return json.loads(txt)["instruction"].strip().lower()
            except Exception:
                return txt.strip().strip('"').lower()
        except Exception:
            time.sleep(2 * (attempt + 1))
    return ""


def find_video(lerobot_dir: str, ep: int, cam="robot0_agentview_left") -> str:
    hits = glob.glob(f"{lerobot_dir}/videos/chunk-*/observation.images.{cam}/"
                     f"episode_{ep:06d}.mp4")
    return hits[0] if hits else ""


def episode_indices(lerobot_dir: str) -> list[int]:
    idx = []
    for line in open(f"{lerobot_dir}/meta/episodes.jsonl"):
        if line.strip():
            idx.append(json.loads(line)["episode_index"])
    return idx


def process_root(lerobot_dir: str, url: str, model: str, slug: str,
                 max_eps: int) -> tuple[int, int]:
    out_path = Path(lerobot_dir) / "meta" / f"primitives_{slug}.jsonl"
    if out_path.exists():
        return (0, 0)  # resume: skip roots already done
    eps = episode_indices(lerobot_dir)[:max_eps]
    lines, n_prim = [], 0
    for ep in eps:
        video = find_video(lerobot_dir, ep)
        if not video:
            continue
        try:
            actions, states, subtask = load_episode(lerobot_dir, ep)
        except Exception:
            continue
        prims = segment_episode(actions, states)
        frames_by_prim = None
        cap = cv2.VideoCapture(video)
        nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(actions)
        for i, p in enumerate(prims):
            idxs = [p.start, (p.start + p.end) // 2, min(p.end, nfr - 1)]
            frs = []
            for fi in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(fi, nfr - 1))
                ok, fr = cap.read()
                if ok:
                    ok2, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok2:
                        frs.append(base64.b64encode(buf.tobytes()).decode())
            if not frs:
                continue
            instr = caption(url, model, frs, p.kind, subtask, p.dz, p.dxy)
            if not instr or len(instr) < 4:
                continue
            lines.append(json.dumps({
                "episode_index": ep, "primitive_idx": i, "kind": p.kind,
                "start": p.start, "end": p.end, "hold": p.holding,
                "dz": round(p.dz, 3), "dxy": round(p.dxy, 3),
                "subtask": subtask, "instruction": instr}))
            n_prim += 1
        cap.release()
    out_path.write_text("\n".join(lines) + "\n")
    return (len(eps), n_prim)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-base", required=True)
    ap.add_argument("--url", default="http://localhost:8301/v1")
    ap.add_argument("--model", default="Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--slug", default="Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--max-eps", type=int, default=25)
    ap.add_argument("--roots-glob", default="*/*/*/lerobot")
    args = ap.parse_args()

    roots = sorted(str(p) for p in Path(args.split_base).glob(args.roots_glob)
                   if (p / "meta" / "episodes.jsonl").is_file())
    mine = [r for i, r in enumerate(roots) if i % args.n_workers == args.worker_id]
    print(f"[w{args.worker_id}] {len(mine)}/{len(roots)} roots, url={args.url}", flush=True)
    t0 = time.time()
    tot_eps = tot_prim = 0
    for k, root in enumerate(mine):
        ne, npr = process_root(root, args.url, args.model, args.slug, args.max_eps)
        tot_eps += ne; tot_prim += npr
        rel = root.split("composite_atomic_split/")[-1].replace("/lerobot", "")
        print(f"[w{args.worker_id}] {k+1}/{len(mine)} {rel}: {ne} eps -> {npr} prims "
              f"(total {tot_prim}, {time.time()-t0:.0f}s)", flush=True)
    print(f"[w{args.worker_id}] DONE {tot_prim} primitives from {tot_eps} eps "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
