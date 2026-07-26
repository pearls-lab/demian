#!/usr/bin/env python3
"""Caption each motion primitive with an INSTRUCTOR-ALIGNED imperative instruction.

Alignment-by-construction (requirement 1): the captions are written by the SAME
model family that serves as System 2 at eval (Qwen3-VL-30B-A3B-Instruct, which
DeMiAn's instructor is fine-tuned from). Train-time primitive instructions and
eval-time instructor instructions therefore share a language distribution, so
S1 does not face out-of-distribution phrasing at deployment.

Two guards make the alignment concrete rather than assumed:
  1. IMPERATIVE prompt: the VLM is told to emit a short command a supervisor
     would give ("lift the arm up toward the cabinet"), NOT a description
     ("the robot arm moves up"). This matches how an instructor issues subgoals.
  2. ROUND-TRIP check (optional, --validate): re-show the VLM the same segment
     with only the generated instruction and ask "does this instruction match
     this motion?" — a cheap consistency filter that drops captions the
     instructor itself would not have produced.

Input: primitive_segmenter output JSON (per episode: primitives with frame
spans) + the episode's video. Output: per-primitive
{episode, primitive_idx, kind, start, end, instruction} appended to the root's
meta/primitives_<slug>.jsonl. Uses the running vLLM server(s).
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
from pathlib import Path

import requests

# Kinematic hint the VLM gets alongside the frames, so the instruction names the
# right direction/target even when a single primitive is visually ambiguous.
_KIND_HINT = {
    "reach": "the arm is moving toward an object to pick up, gripper open",
    "grasp": "the gripper is closing on an object",
    "lift": "the arm is lifting the held object upward",
    "transport": "the arm is carrying the held object horizontally",
    "lower": "the arm is lowering the held object toward a target",
    "release": "the gripper is opening to release the object",
    "retract": "the arm is moving away, empty, after releasing",
    "articulate": "the arm is operating a fixture (turn/open/close/press)",
}

SYSTEM = (
    "You are a robot supervisor issuing a SHORT imperative command for the "
    "current motion segment, exactly as you would tell a robot arm what to do "
    "next. Output one command, lower-case, no trailing period, 4-10 words, "
    'as JSON: {"instruction": "..."}. It must be an ORDER ("lift the mug up '
    'toward the cabinet"), never a description ("the arm moves up"). Name the '
    "object and target when visible. Valid JSON only."
)


def _extract_clip(video: str, start: int, end: int, fps: int = 20) -> str:
    """Cut [start,end] frames from the episode mp4 to a temp clip."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
         "-vf", f"select='between(n,{start},{end})',setpts=N/{fps}/TB",
         "-an", tmp.name], check=True)
    return tmp.name


def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def caption_primitive(url: str, model: str, clip_path: str, kind: str,
                      subtask: str) -> str:
    hint = _KIND_HINT.get(kind, "")
    user = [
        {"type": "video_url",
         "video_url": {"url": f"data:video/mp4;base64,{_b64(clip_path)}"}},
        {"type": "text", "text":
            f"Parent subtask: {subtask}\nMotion segment: {hint}.\n"
            f"Give the imperative command for THIS segment only."},
    ]
    r = requests.post(f"{url}/chat/completions", json={
        "model": model, "temperature": 0.4, "max_tokens": 64,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}]}, timeout=120)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"].strip()
    try:
        return json.loads(txt)["instruction"].strip()
    except Exception:
        return txt.strip().strip('"')


def validate(url: str, model: str, clip_path: str, instruction: str) -> bool:
    """Round-trip: would the instructor accept this instruction for this motion?"""
    user = [
        {"type": "video_url",
         "video_url": {"url": f"data:video/mp4;base64,{_b64(clip_path)}"}},
        {"type": "text", "text":
            f'Does the command "{instruction}" correctly describe what the arm '
            f'does in this clip? Answer JSON {{"match": true/false}}.'}]
    r = requests.post(f"{url}/chat/completions", json={
        "model": model, "temperature": 0.0, "max_tokens": 16,
        "messages": [{"role": "user", "content": user}]}, timeout=120)
    try:
        return bool(json.loads(r.json()["choices"][0]["message"]["content"])["match"])
    except Exception:
        return True  # fail-open; validation is a filter, not a gate


def find_video(lerobot_dir: str, ep: int, cam: str = "robot0_agentview_left") -> str:
    import glob
    hits = glob.glob(f"{lerobot_dir}/videos/chunk-*/observation.images.{cam}/"
                     f"episode_{ep:06d}.mp4")
    return hits[0] if hits else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-dir", required=True)
    ap.add_argument("--segments", required=True, help="primitive_segmenter JSON")
    ap.add_argument("--url", default="http://localhost:8300/v1")
    ap.add_argument("--model", default="Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--slug", default="Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    segs = json.loads(Path(args.segments).read_text())
    out_path = Path(args.out or
                    f"{args.lerobot_dir}/meta/primitives_{args.slug}.jsonl")
    lines, n_drop = [], 0
    for epdata in segs:
        ep = epdata["episode"]
        video = find_video(args.lerobot_dir, ep)
        if not video:
            continue
        for i, p in enumerate(epdata["primitives"]):
            clip = _extract_clip(video, p["start"], p["end"])
            instr = caption_primitive(args.url, args.model, clip, p["kind"],
                                      epdata["subtask"])
            ok = (not args.validate) or validate(args.url, args.model, clip, instr)
            Path(clip).unlink(missing_ok=True)
            if not ok:
                n_drop += 1
                continue
            lines.append(json.dumps({
                "episode_index": ep, "primitive_idx": i, "kind": p["kind"],
                "start": p["start"], "end": p["end"], "instruction": instr}))
            print(f"  ep{ep} p{i} [{p['kind']:10s}] -> {instr!r}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {len(lines)} primitive captions ({n_drop} dropped by validate) "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
