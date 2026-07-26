#!/usr/bin/env python3
"""Dense annotation for robotic manipulation datasets.

Each dataset differs in three things:
  1. How to iterate over processing units (directory structure)
  2. How to find the video file(s) for an episode
  3. What task context to inject into prompts

Everything else — API calls, message building, prompt loading, annotations I/O — is shared.

Usage:
    python dense_annotator.py --dataset robocasa --vllm_base_url http://localhost:8000/v1 --vllm_model Qwen3-VL-30B-A3B-Instruct
    python dense_annotator.py --dataset robocasa --task CloseCabinet-DoubleDoor --n 1
    python dense_annotator.py --dataset robocasa --task CheesyBread,CoffeeSetupMug,CloseCabinet-DoubleDoor --n 1
"""

import argparse
import base64
import glob
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import openai


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Resolve the user's home base: /workspace inside the container, full path outside.
_BASE = (
    "/workspace"
    if os.path.isdir("/workspace")
    else "/path/to"
)

DEFAULT_DATA_PATHS = {
    "robocasa":     f"{_BASE}/data/robocasa/datasets/v1.0/pretrain/atomic_split",
    "robocasa_v01": f"{_BASE}/data/robocasa/datasets/v1.0/pretrain/robocasa_v01_subset/lerobot",
}
DEFAULT_ATOMIC_DATA_PATH = f"{_BASE}/data/robocasa/datasets/v1.0/pretrain/atomic"

_SPLIT_SUFFIX_RE = re.compile(r"-(DoubleDoor|SingleDoor|split)$")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplitInfo:
    """Identifies one processing unit: a single lerobot directory."""
    lerobot_dir: str
    display_name: str   # shown in log output
    task_name:   str    # task identifier (e.g. "CloseCabinet-DoubleDoor", "GR1")
    split_name:  str    # sub-task/primitive (e.g. "ApproachToObject_0"); "" for flat datasets


@dataclass
class SplitContext:
    """Task context computed once per SplitInfo, injected into every VLM prompt."""
    original_task_name: str
    original_descs:     list         # natural-language task descriptions (RoboCasa only)
    primitive_ctx:      Optional[dict]  # ground-truth for reasoning annotations (RoboCasa only)


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------

class DatasetAdapter:
    """Base class. Each subclass encapsulates one dataset's structural logic."""

    name: str = ""

    def iter_splits(
        self, data_path: Path, task_filter: str, split_filter: str,
        worker_id: int = 0, n_workers: int = 1,
    ) -> Iterator[SplitInfo]:
        raise NotImplementedError

    def find_videos(self, lerobot_dir: str, ep_idx: int) -> tuple:
        """Return (video_left_or_only, video_right_or_None). None means not found."""
        raise NotImplementedError

    def get_split_context(
        self, split_info: SplitInfo, atomic_data_path: str, decomp: dict
    ) -> SplitContext:
        raise NotImplementedError

    def get_episode_task_desc(self, episode: dict, split_info: SplitInfo) -> str:
        return (episode.get("tasks") or [split_info.task_name])[0]

    def get_episode_context(self, episode: dict, split_ctx: SplitContext) -> SplitContext:
        """Return a SplitContext for this specific episode. Default: return split_ctx unchanged."""
        return split_ctx

    def find_all_videos(self, lerobot_dir: str, ep_idx: int) -> dict:
        """Return dict of camera_name -> video_path for all available cameras."""
        left, right = self.find_videos(lerobot_dir, ep_idx)
        result = {}
        if left:
            result["left"] = left
        if right:
            result["right"] = right
        return result

    def episode_group_key(self, episode: dict) -> str:
        """Key used to count --n units. One unit = one original episode for flat datasets."""
        return str(episode["episode_index"])


# --- RoboCasa atomic_split ---

class RobocasaAdapter(DatasetAdapter):
    """
    Directory layout: {data_path}/{task}/{split}/{date}/lerobot/
    Videos: videos/chunk-*/observation.images.robot0_agentview_{left,right}/episode_*.mp4
    """
    name = "robocasa"

    @staticmethod
    def _split_key(d):
        m = re.search(r'_(\d+)$', d.name)
        return (int(m.group(1)) if m else 0, d.name)

    def iter_splits(self, data_path, task_filter, split_filter, worker_id=0, n_workers=1):  # noqa: ARG002
        task_dirs = sorted(d for d in data_path.iterdir() if d.is_dir())
        for task_idx, task_dir in enumerate(task_dirs):
            if task_filter and task_dir.name not in task_filter:
                continue
            if n_workers > 1 and task_idx % n_workers != worker_id:
                continue
            for child in sorted(task_dir.iterdir(), key=self._split_key):
                if not child.is_dir():
                    continue
                # Layout A: task/date/lerobot  (child is a date dir)
                lerobot_dir = child / "lerobot"
                if lerobot_dir.exists():
                    if split_filter:
                        continue  # no split name to match against
                    yield SplitInfo(
                        lerobot_dir=str(lerobot_dir),
                        display_name=f"{task_dir.name} / {child.name}",
                        task_name=task_dir.name,
                        split_name="",
                    )
                    continue
                # Layout B: task/split/date/lerobot  (child is a split dir)
                if split_filter and child.name != split_filter:
                    continue
                for date_dir in sorted(child.iterdir()):
                    lerobot_dir = date_dir / "lerobot"
                    if lerobot_dir.exists():
                        yield SplitInfo(
                            lerobot_dir=str(lerobot_dir),
                            display_name=f"{task_dir.name} / {child.name} / {date_dir.name}",
                            task_name=task_dir.name,
                            split_name=child.name,
                        )

    def find_videos(self, lerobot_dir, ep_idx):
        episode_str = f"episode_{ep_idx:06d}.mp4"
        base = os.path.join(lerobot_dir, "videos", "chunk-*")

        def first_match(camera):
            hits = glob.glob(os.path.join(base, f"observation.images.robot0_{camera}", episode_str))
            return hits[0] if hits else None

        return first_match("agentview_left"), first_match("agentview_right")

    def get_split_context(self, split_info, atomic_data_path, decomp):
        original_task_name = _SPLIT_SUFFIX_RE.sub("", split_info.task_name)
        return SplitContext(
            original_task_name=original_task_name,
            original_descs=_load_robocasa_task_descs(split_info.lerobot_dir),
            primitive_ctx=_get_primitive_ctx(split_info.task_name, split_info.split_name, decomp),
        )

    def find_all_videos(self, lerobot_dir, ep_idx):
        result = super().find_all_videos(lerobot_dir, ep_idx)
        episode_str = f"episode_{ep_idx:06d}.mp4"
        base = os.path.join(lerobot_dir, "videos", "chunk-*")
        hits = glob.glob(os.path.join(base, "observation.images.robot0_eye_in_hand", episode_str))
        if hits:
            result["wrist"] = hits[0]
        return result


# --- RoboCasa v0.1 flat subset ---

class RobocasaV01Adapter(RobocasaAdapter):
    """Flat lerobot layout for RoboCasa v0.1 subset. Same cameras as RoboCasa."""
    name = "robocasa_v01"

    def iter_splits(self, data_path, task_filter, split_filter, worker_id=0, n_workers=1):
        if (data_path / "meta").exists():
            yield SplitInfo(
                lerobot_dir=str(data_path),
                display_name="robocasa_v01_subset",
                task_name="RoboCasa",
                split_name="",
            )

    def get_split_context(self, split_info, atomic_data_path, decomp):
        return SplitContext(
            original_task_name="RoboCasa",
            original_descs=[],
            primitive_ctx=None,
        )

    def episode_group_key(self, episode):
        return str(episode["episode_index"])


# --- GR1_robot_split ---

class GR1Adapter(DatasetAdapter):
    """
    Directory layout: {data_path}/  (flat, single lerobot root)
    Videos: videos/chunk-{NNN}/observation.images.ego_view_freq20/episode_*.mp4
    """
    name = "gr1"

    def iter_splits(self, data_path, task_filter, split_filter, worker_id=0, n_workers=1):
        if (data_path / "meta").exists():
            yield SplitInfo(
                lerobot_dir=str(data_path),
                display_name="GR1_robot_split",
                task_name="GR1",
                split_name="",
            )

    def find_videos(self, lerobot_dir, ep_idx):
        chunk = ep_idx // 1000
        path = os.path.join(
            lerobot_dir, "videos", f"chunk-{chunk:03d}",
            "observation.images.ego_view_freq20", f"episode_{ep_idx:06d}.mp4",
        )
        return (path if os.path.exists(path) else None), None

    def get_split_context(self, split_info, atomic_data_path, decomp):
        return SplitContext(
            original_task_name="GR1 robot manipulation",
            original_descs=[],
            primitive_ctx=None,
        )

    def episode_group_key(self, episode: dict) -> str:
        return str(episode.get("orig_episode_index", episode["episode_index"]))


ADAPTERS = {
    "robocasa":     RobocasaAdapter(),
    "robocasa_v01": RobocasaV01Adapter(),
    "gr1":          GR1Adapter(),
}


# ---------------------------------------------------------------------------
# RoboCasa-specific helpers  (used only by RobocasaAdapter / SplitContext)
# ---------------------------------------------------------------------------

def _load_robocasa_task_descs(lerobot_dir: str) -> list:
    """Load unique natural-language task descriptions from original_episodes.jsonl."""
    path = os.path.join(lerobot_dir, "meta", "original_episodes.jsonl")
    descs = []
    for ep in load_jsonl(path):
        text = (ep.get("tasks") or [""])[0]
        if text:
            descs.append(text)
    return list(dict.fromkeys(descs))  # deduplicate, preserve order


def _parse_split_name(split_name: str) -> tuple:
    """'ApproachToObject_1' → ('ApproachToObject', 1);  'GraspObject' → ('GraspObject', 0)."""
    m = re.match(r"^(.+)_(\d+)$", split_name)
    return (m.group(1), int(m.group(2))) if m else (split_name, 0)


def _get_primitive_ctx(split_task_name: str, split_name: str, decomp: dict) -> Optional[dict]:
    tasks_decomp = decomp.get("tasks", {})
    primitives   = decomp.get("primitives", {})

    sequence = tasks_decomp.get(split_task_name) or tasks_decomp.get(
        _SPLIT_SUFFIX_RE.sub("", split_task_name)
    )
    if sequence is None:
        return None

    primitive, occurrence = _parse_split_name(split_name)
    position, count = None, 0
    for i, p in enumerate(sequence):
        if p == primitive:
            if count == occurrence:
                position = i
                break
            count += 1
    if position is None:
        return None

    prev_action = sequence[position - 1] if position > 0 else None
    next_action = sequence[position + 1] if position + 1 < len(sequence) else None
    return {
        "primitive_sequence":     sequence,
        "prev_action":            prev_action,
        "prev_action_desc":       primitives.get(prev_action, "") if prev_action else None,
        "current_primitive":      primitive,
        "current_primitive_desc": primitives.get(primitive, ""),
        "next_action":            next_action,
        "next_action_desc":       primitives.get(next_action, "") if next_action else None,
        "task_completion":        position == len(sequence) - 1,
    }


# ---------------------------------------------------------------------------
# Shared I/O helpers
# ---------------------------------------------------------------------------

def encode_video_base64(video_path: str) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def compute_window_indices(step: int, total_steps: int, window_size: int = 20, stride: int = 2) -> list:
    """Compute frame indices for a window starting at *step*.

    Returns up to ``window_size // stride`` indices, each < *total_steps*.
    """
    indices = []
    for i in range(0, window_size, stride):
        idx = step + i
        if idx >= total_steps:
            break
        indices.append(idx)
    return indices


def extract_frames(video_path: str, frame_indices: list) -> list:
    """Extract specific frames from a video file as numpy arrays (BGR)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames


def create_grid_frame(frames_dict: dict) -> np.ndarray:
    """Create a labeled 2×2 grid image from camera views.

    Layout::

        [LEFT VIEW]   [RIGHT VIEW]
        [WRIST VIEW]  [blank]
    """
    left = frames_dict["left"].copy()
    right = frames_dict["right"].copy()
    h, w = left.shape[:2]
    black = np.zeros((h, w, 3), dtype=np.uint8)
    wrist = frames_dict.get("wrist", black).copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, h / 500)
    thickness = max(1, int(h / 250))
    color = (255, 255, 255)

    for img, label in [(left, "LEFT VIEW"), (right, "RIGHT VIEW"), (wrist, "WRIST VIEW")]:
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (8, 8), (16 + tw, 16 + th), (0, 0, 0), -1)
        cv2.putText(img, label, (12, 12 + th), font, font_scale, color, thickness)

    top = np.hstack([left, right])
    bottom = np.hstack([wrist, black])
    return np.vstack([top, bottom])


def encode_frame_base64(frame: np.ndarray) -> str:
    """Encode a BGR numpy frame as base64 JPEG."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def make_side_by_side(left_path: str, right_path: str) -> str:
    """Stack two video clips horizontally into a temporary composite file.

    Labels 'LEFT VIEW' and 'RIGHT VIEW' are burned into the corners so the
    VLM can unambiguously identify which half is which camera angle.
    Returns the path to the temp file; the caller is responsible for deletion.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    # NOTE: view labels dropped (2026-07-23) — the imageio-ffmpeg static build on
    # this fleet has no drawtext filter, and prompts.json never references the
    # labels. Left half = left agentview, right half = right agentview.
    filter_graph = "[0][1]hstack[out]"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", left_path, "-i", right_path,
            "-filter_complex", filter_graph,
            "-map", "[out]",
            "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-an",
            tmp.name,
        ],
        check=True,
    )
    return tmp.name


def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_existing_annotations(annotations_path: str) -> dict:
    return {e["episode_index"]: e for e in load_jsonl(annotations_path)}


def save_annotations(annotations_path: str, annotations: dict):
    with open(annotations_path, "w") as f:
        for idx in sorted(annotations.keys()):
            f.write(json.dumps(annotations[idx]) + "\n")


def load_existing_fine_grain_annotations(annotations_path: str) -> dict:
    """Load fine-grain annotations keyed by ``(episode_index, step)``."""
    result = {}
    for entry in load_jsonl(annotations_path):
        key = (entry["episode_index"], entry["step"])
        result[key] = entry
    return result


def save_fine_grain_annotations(annotations_path: str, annotations: dict):
    """Save fine-grain annotations sorted by ``(episode_index, step)``."""
    with open(annotations_path, "w") as f:
        for key in sorted(annotations.keys()):
            f.write(json.dumps(annotations[key]) + "\n")


# ---------------------------------------------------------------------------
# VLM interface
# ---------------------------------------------------------------------------

def build_messages(
    video_path:           str,
    is_composite:         bool,
    task_desc:            str,
    split_ctx:            SplitContext,
    prompt_text:          str,
    constraint:           str = "",
    inject_primitive_ctx: bool = False,
    video_content_type:   str = "video_url",  # content block type for the video data URI
    no_context:           bool = False,
    json_key:             str = "annotation",
    prev_step_annotation:    str = "",
) -> list:
    """Build the system + user message list for one VLM call.

    video_path: path to the video file to send (single-view or L/R composite).
    is_composite: True when video_path is a side-by-side composite of two cameras.
    inject_primitive_ctx=True only for the 'reasoning' annotation type.
    video_content_type: content block type for the video data URI.
    """
    camera_desc = (
        "one video clip showing the same robot from two synchronized camera angles "
        "side-by-side (LEFT VIEW on the left half, RIGHT VIEW on the right half). "
        #"Both halves capture the identical scene simultaneously from different perspectives — "
        #"there is only one robot and one arm."
        if is_composite else
        "one video clip from a camera mounted on the agent"
    )

    system_parts = [
        f"You are an expert analyst of agent manipulation videos. "
        f"You will be shown {camera_desc}. "
        f'Respond with a JSON object containing a single key "{json_key}" whose value is the requested description. '
        f'Example: {{"{json_key}": "The robot arm reaches toward the cabinet door."}} '
        "Output valid JSON only — no markdown, no extra keys, no text outside the JSON object."
    ]
    if constraint:
        system_parts.append(f'The "{json_key}" value must be {constraint}.')

    def _fmt(name, desc):
        return f"{name} — {desc}" if desc else name

    # Task context block
    task_context = f"TASK: {split_ctx.original_task_name}\n"
    if not no_context:
        if split_ctx.original_descs:
            orig_block = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(split_ctx.original_descs[:5]))
            task_context += f"TASK DESCRIPTIONS:\n{orig_block}\n\n"
        task_context += f"CURRENT TASK: {task_desc}\n"

    # Ground-truth reasoning context (injected only for the 'reasoning' prompt)
    if not no_context and inject_primitive_ctx and split_ctx.primitive_ctx is not None:
        pc = split_ctx.primitive_ctx
        next_act     = pc["next_action"]
        next_act_str = _fmt(next_act, pc["next_action_desc"]) if next_act else "None (final step)"
        prev_act     = pc["prev_action"]
        prev_act_str = _fmt(prev_act, pc["prev_action_desc"]) if prev_act else "None (first step)"
        task_context += (
            f"\n[GROUND-TRUTH CONTEXT FOR REASONING]\n"
            f"Full action sequence: {' → '.join(pc['primitive_sequence'])}\n"
            f"Previous action: {prev_act_str}\n"
            f"Current action: {_fmt(pc['current_primitive'], pc['current_primitive_desc'])}\n"
            f"Next action: {next_act_str}\n"
            f"Task completion after this clip: {'Yes' if pc['task_completion'] else 'No'}\n"
        )

    if prev_step_annotation:
        task_context += (
            f"\n[PREVIOUS STEP'S REASONING]\n"
            f"{prev_step_annotation}\n"
        )

    task_context += f"\nINSTRUCTION: {prompt_text}"

    # Single video content block (composite or single-view)
    video_data_url = f"data:video/mp4;base64,{encode_video_base64(video_path)}"
    user_content = [
        {
            "type": video_content_type,
            video_content_type: {"url": video_data_url},
        },
        {"type": "text", "text": task_context},
    ]

    return [
        {"role": "system", "content": " ".join(system_parts)},
        {"role": "user",   "content": user_content},
    ]


def build_messages_fine_grain(
    grid_frames:          list,
    step:                 int,
    total_steps:          int,
    is_grid:              bool,
    task_desc:            str,
    split_ctx:            SplitContext,
    prompt_text:          str,
    constraint:           str = "",
    inject_primitive_ctx: bool = False,
    no_context:           bool = False,
    json_key:             str = "annotation",
    prev_step_annotation:    str = "",
) -> list:
    """Build system + user messages with multiple image frames for fine-grain annotation."""
    n_frames = len(grid_frames)

    if is_grid:
        camera_desc = (
            f"{n_frames} frames from a robot manipulation episode. "
            f"Each frame is a 2x2 grid showing four quadrants: "
            f"left agent view (top-left), right agent view (top-right), "
            f"wrist camera (bottom-left), and a blank quadrant (bottom-right). "
            f"This is step {step} of {total_steps} total steps."
        )
    else:
        camera_desc = (
            f"{n_frames} frames from a robot manipulation episode "
            f"captured from a camera mounted on the agent. "
            f"This is step {step} of {total_steps} total steps."
        )

    system_parts = [
        f"You are an expert analyst of agent manipulation videos. "
        f"You will be shown {camera_desc} "
        f'Respond with a JSON object containing a single key "{json_key}" whose value is the requested description. '
        f'Example: {{"{json_key}": "The robot arm reaches toward the cabinet door."}} '
        "Output valid JSON only — no markdown, no extra keys, no text outside the JSON object."
    ]
    if constraint:
        system_parts.append(f'The "{json_key}" value must be {constraint}.')

    def _fmt(name, desc):
        return f"{name} — {desc}" if desc else name

    task_context = f"TASK: {split_ctx.original_task_name}\n"
    if not no_context:
        if split_ctx.original_descs:
            orig_block = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(split_ctx.original_descs[:5]))
            task_context += f"TASK DESCRIPTIONS:\n{orig_block}\n\n"
        task_context += f"CURRENT TASK: {task_desc}\n"

    if not no_context and inject_primitive_ctx and split_ctx.primitive_ctx is not None:
        pc = split_ctx.primitive_ctx
        next_act     = pc["next_action"]
        next_act_str = _fmt(next_act, pc["next_action_desc"]) if next_act else "None (final step)"
        prev_act     = pc["prev_action"]
        prev_act_str = _fmt(prev_act, pc["prev_action_desc"]) if prev_act else "None (first step)"
        task_context += (
            f"\n[GROUND-TRUTH CONTEXT FOR REASONING]\n"
            f"Full action sequence: {' → '.join(pc['primitive_sequence'])}\n"
            f"Previous action: {prev_act_str}\n"
            f"Current action: {_fmt(pc['current_primitive'], pc['current_primitive_desc'])}\n"
            f"Next action: {next_act_str}\n"
            f"Task completion after this clip: {'Yes' if pc['task_completion'] else 'No'}\n"
        )

    if prev_step_annotation:
        task_context += (
            f"\n[PREVIOUS STEP'S REASONING]\n"
            f"{prev_step_annotation}\n"
        )

    task_context += f"\nINSTRUCTION: {prompt_text}"

    user_content = []
    for frame in grid_frames:
        data_url = f"data:image/jpeg;base64,{encode_frame_base64(frame)}"
        user_content.append({
            "type": "image_url",
            "image_url": {"url": data_url},
        })
    user_content.append({"type": "text", "text": task_context})

    return [
        {"role": "system", "content": " ".join(system_parts)},
        {"role": "user",   "content": user_content},
    ]


def call_api(client: openai.OpenAI, model: str, messages: list, max_tokens: int = 256) -> str:
    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_dataset(
    adapter:             DatasetAdapter,
    split_info:          SplitInfo,
    prompts_config:      dict,
    annotation_types:       list,
    client:              openai.OpenAI,
    model:               str,
    atomic_data_path:    str,
    decomp:              dict,
    rerun:               bool,
    max_tokens:          int,
    n_limit:             int,    # max episodes to annotation per split (-1 = unlimited)
    video_content_type:  str = "video_url",
    no_context:          bool = False,
    worker_id:           int = 0,
    n_workers:           int = 1,
    ep_shard:            bool = False,
):
    meta_dir      = os.path.join(split_info.lerobot_dir, "meta")
    episodes_file = os.path.join(meta_dir, "episodes.jsonl")
    model_slug    = model.split("/")[-1]
    suffix        = "_no_context" if no_context else ""
    shard_suffix  = f"_shard{worker_id}" if ep_shard and n_workers > 1 else ""
    annotations_file = os.path.join(meta_dir, f"captions_{model_slug}{suffix}{shard_suffix}.jsonl")

    if not os.path.exists(episodes_file):
        print(f"    [skip] no episodes.jsonl in {meta_dir}")
        return

    episodes  = load_jsonl(episodes_file)
    existing  = load_existing_annotations(annotations_file)
    # When using ep_shard, also load from the merged (non-shard) file to pick up
    # annotations from prior runs, so we don't redo already-completed annotation types.
    if ep_shard and n_workers > 1:
        merged_file = os.path.join(meta_dir, f"captions_{model_slug}{suffix}.jsonl")
        if merged_file != annotations_file and os.path.exists(merged_file):
            merged = load_existing_annotations(merged_file)
            for ep_idx, entry in merged.items():
                if ep_idx not in existing:
                    existing[ep_idx] = entry
                else:
                    # Merge annotation types from merged file into shard entry
                    for ct, val in entry.get("annotations", {}).items():
                        existing[ep_idx].setdefault("annotations", {}).setdefault(ct, val)
    split_ctx = adapter.get_split_context(split_info, atomic_data_path, decomp)

    if ("reasoning" in annotation_types
            and split_ctx.primitive_ctx is None
            and adapter.name == "robocasa"):
        print(f"    [warn] no decomposition found for "
              f"'{split_info.task_name}/{split_info.split_name}' — "
              "reasoning runs without ground-truth context")

    prompts        = prompts_config["prompts"]
    modified       = False
    count          = 0     # sub-episodes actually annotationed
    seen_groups    = set() # for --n group-level counting
    prev_reasoning = {}    # annotation_type -> text (previous episode's reasoning)

    for ep in episodes:
        # Episode-level sharding for flat datasets (GR1).
        # RoboCasa already shards at task level inside iter_splits.
        if n_workers > 1 and adapter.name != "robocasa":
            orig_ep = ep.get("orig_episode_index", ep["episode_index"])
            if orig_ep % n_workers != worker_id:
                continue

        group_key    = adapter.episode_group_key(ep)
        is_new_group = group_key not in seen_groups

        # Break when we'd start a new group beyond the limit
        if n_limit >= 0 and is_new_group and len(seen_groups) >= n_limit:
            break

        if is_new_group:
            seen_groups.add(group_key)

        ep_idx    = ep["episode_index"]
        task_desc = adapter.get_episode_task_desc(ep, split_info)
        ep_ctx    = adapter.get_episode_context(ep, split_ctx)

        video_left, video_right = adapter.find_videos(split_info.lerobot_dir, ep_idx)
        if not video_left:
            print(f"    [ep {ep_idx}] missing video, skipping")
            continue

        entry    = existing.get(ep_idx, {"episode_index": ep_idx, "annotations": {}})
        done_cts = set(entry.get("annotations", {}).keys())
        to_run   = [ct for ct in annotation_types if ct in prompts and (rerun or ct not in done_cts)]

        if not to_run:
            # Still update prev_reasoning from existing annotations
            for ct in annotation_types:
                if ct.startswith("reasoning") and ct in entry.get("annotations", {}):
                    prev_reasoning[ct] = entry["annotations"][ct]
            continue

        print(f"    [ep {ep_idx}] {task_desc!r:.60}  →  {to_run}")

        if "annotations" not in entry:
            entry["annotations"] = {}

        # Build composite side-by-side video once per episode, reuse for all annotation types
        composite_path = None
        if video_right:
            try:
                composite_path = make_side_by_side(video_left, video_right)
            except FileNotFoundError:
                print(f"    [warn] ffmpeg not found — using left view only for ep {ep_idx}")
                composite_path = None
        video_path   = composite_path if composite_path else video_left
        is_composite = composite_path is not None

        try:
            for ct in to_run:
                prompt_cfg = prompts[ct]
                try:
                    json_key = "reasoning" if ct == "reasoning" else "annotation"
                    prev_annotation = prev_reasoning.get(ct, "") if ct.startswith("reasoning") else ""
                    messages = build_messages(
                        video_path, is_composite,
                        task_desc, ep_ctx,
                        prompt_cfg["prompt"],
                        constraint=prompt_cfg.get("constraint", ""),
                        inject_primitive_ctx=(ct == "reasoning"),
                        video_content_type=video_content_type,
                        no_context=no_context,
                        json_key=json_key,
                        prev_step_annotation=prev_annotation,
                    )
                    raw = call_api(client, model, messages, max_tokens)
                    try:
                        annotation = json.loads(raw)[json_key]
                    except (json.JSONDecodeError, KeyError):
                        annotation = raw
                    entry["annotations"][ct] = annotation
                    print(f"      [{ct}] {annotation[:100]}...")
                except Exception as exc:
                    print(f"      [{ct}] ERROR: {exc}")
                    entry["annotations"][ct] = f"ERROR: {exc}"
        finally:
            if composite_path and os.path.exists(composite_path):
                os.unlink(composite_path)

        # Update prev_reasoning for next episode
        for ct in annotation_types:
            if ct.startswith("reasoning") and ct in entry.get("annotations", {}):
                prev_reasoning[ct] = entry["annotations"][ct]

        existing[ep_idx] = entry
        modified = True
        count   += 1
        save_annotations(annotations_file, existing)

    if modified:
        groups = len(seen_groups)
        if groups != count:
            print(f"    Saved → {annotations_file}  ({groups} original episodes, {count} sub-episodes processed)")
        else:
            print(f"    Saved → {annotations_file}  ({count} episodes processed)")


def process_dataset_fine_grain(
    adapter:             DatasetAdapter,
    split_info:          SplitInfo,
    prompts_config:      dict,
    annotation_types:       list,
    client:              openai.OpenAI,
    model:               str,
    atomic_data_path:    str,
    decomp:              dict,
    rerun:               bool,
    max_tokens:          int,
    n_limit:             int,
    window_size:         int = 20,
    stride:              int = 2,
    skip_step:           int = 1,
    no_context:          bool = False,
    worker_id:           int = 0,
    n_workers:           int = 1,
    ep_shard:            bool = False,
):
    """Fine-grain annotation: one VLM call per (episode, step) pair."""
    meta_dir      = os.path.join(split_info.lerobot_dir, "meta")
    episodes_file = os.path.join(meta_dir, "episodes.jsonl")
    model_slug    = model.split("/")[-1]
    suffix        = "_no_context" if no_context else ""
    shard_suffix  = f"_shard{worker_id}" if ep_shard and n_workers > 1 else ""
    annotations_file = os.path.join(meta_dir, f"captions_fine_grain_{model_slug}{suffix}{shard_suffix}.jsonl")

    if not os.path.exists(episodes_file):
        print(f"    [skip] no episodes.jsonl in {meta_dir}")
        return

    episodes  = load_jsonl(episodes_file)
    existing  = load_existing_fine_grain_annotations(annotations_file)
    split_ctx = adapter.get_split_context(split_info, atomic_data_path, decomp)

    if ("reasoning" in annotation_types
            and split_ctx.primitive_ctx is None
            and adapter.name == "robocasa"):
        print(f"    [warn] no decomposition found for "
              f"'{split_info.task_name}/{split_info.split_name}' — "
              "reasoning runs without ground-truth context")

    prompts       = prompts_config["prompts"]
    modified      = False
    step_count    = 0
    seen_groups   = set()

    for ep in episodes:
        # Episode-level sharding for flat datasets (GR1),
        # or any dataset when --ep_shard is enabled.
        if n_workers > 1 and (adapter.name != "robocasa" or ep_shard):
            orig_ep = ep.get("orig_episode_index", ep["episode_index"])
            if orig_ep % n_workers != worker_id:
                continue

        group_key    = adapter.episode_group_key(ep)
        is_new_group = group_key not in seen_groups
        if n_limit >= 0 and is_new_group and len(seen_groups) >= n_limit:
            break
        if is_new_group:
            seen_groups.add(group_key)

        ep_idx      = ep["episode_index"]
        total_steps = ep.get("length", 0)
        if total_steps == 0:
            print(f"    [ep {ep_idx}] no length field, skipping")
            continue

        task_desc = adapter.get_episode_task_desc(ep, split_info)
        ep_ctx    = adapter.get_episode_context(ep, split_ctx)

        # Find all camera videos for this episode
        all_videos = adapter.find_all_videos(split_info.lerobot_dir, ep_idx)
        if not all_videos:
            print(f"    [ep {ep_idx}] missing videos, skipping")
            continue

        is_grid = len(all_videos) > 1

        # Open all video captures once per episode
        caps = {}
        for cam_name, vpath in all_videos.items():
            cap = cv2.VideoCapture(vpath)
            if cap.isOpened():
                caps[cam_name] = cap
            else:
                print(f"    [ep {ep_idx}] cannot open {cam_name} video: {vpath}")
        if not caps:
            print(f"    [ep {ep_idx}] no readable videos, skipping")
            continue

        print(f"    [ep {ep_idx}] {task_desc!r:.60}  T={total_steps}  cameras={list(caps.keys())}")

        try:
            # Track previous step's reasoning annotations for temporal context
            prev_reasoning = {}  # annotation_type -> text

            for step in range(0, total_steps, skip_step):
                entry_key = (ep_idx, step)
                entry     = existing.get(entry_key, {"episode_index": ep_idx, "step": step, "annotations": {}})
                done_cts  = set(entry.get("annotations", {}).keys())
                to_run    = [ct for ct in annotation_types if ct in prompts and (rerun or ct not in done_cts)]

                if not to_run:
                    # Still update prev_reasoning from existing annotations
                    for ct in annotation_types:
                        if ct.startswith("reasoning") and ct in entry.get("annotations", {}):
                            prev_reasoning[ct] = entry["annotations"][ct]
                    continue

                frame_indices = compute_window_indices(step, total_steps, window_size, stride)

                # Extract frames from all cameras at the computed indices
                cam_frames = {}
                for cam_name, cap in caps.items():
                    frames = []
                    for idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(frame)
                    cam_frames[cam_name] = frames

                n_frames = min(len(f) for f in cam_frames.values()) if cam_frames else 0
                if n_frames == 0:
                    continue

                # Compose grid images (one per frame in the window)
                if is_grid:
                    images = []
                    for i in range(n_frames):
                        fdict = {cam: cam_frames[cam][i] for cam in cam_frames}
                        images.append(create_grid_frame(fdict))
                else:
                    cam_name = next(iter(cam_frames))
                    images = cam_frames[cam_name][:n_frames]

                if "annotations" not in entry:
                    entry["annotations"] = {}

                for ct in to_run:
                    prompt_cfg = prompts[ct]
                    try:
                        json_key = "reasoning" if ct == "reasoning" else "annotation"
                        prev_annotation = prev_reasoning.get(ct, "") if ct.startswith("reasoning") else ""
                        messages = build_messages_fine_grain(
                            images, step, total_steps, is_grid,
                            task_desc, ep_ctx,
                            prompt_cfg["prompt"],
                            constraint=prompt_cfg.get("constraint", ""),
                            inject_primitive_ctx=(ct == "reasoning"),
                            no_context=no_context,
                            json_key=json_key,
                            prev_step_annotation=prev_annotation,
                        )
                        raw = call_api(client, model, messages, max_tokens)
                        try:
                            annotation = json.loads(raw)[json_key]
                        except (json.JSONDecodeError, KeyError):
                            annotation = raw
                        entry["annotations"][ct] = annotation
                        if step % 20 == 0:
                            print(f"      [step {step}/{total_steps}] [{ct}] {annotation[:80]}...")
                    except Exception as exc:
                        print(f"      [step {step}] [{ct}] ERROR: {exc}")
                        entry["annotations"][ct] = f"ERROR: {exc}"

                # Update prev_reasoning for next step
                for ct in annotation_types:
                    if ct.startswith("reasoning") and ct in entry.get("annotations", {}):
                        prev_reasoning[ct] = entry["annotations"][ct]

                existing[entry_key] = entry
                modified    = True
                step_count += 1

                # Save periodically
                if step_count % 50 == 0:
                    save_fine_grain_annotations(annotations_file, existing)
        finally:
            for cap in caps.values():
                cap.release()

        # Save after each episode
        if modified:
            save_fine_grain_annotations(annotations_file, existing)

    if modified:
        print(f"    Saved → {annotations_file}  ({step_count} steps processed)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dense annotation for robotic manipulation datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset selection
    parser.add_argument(
        "--dataset", choices=list(ADAPTERS), default="robocasa",
        help="Which dataset to annotation",
    )
    parser.add_argument(
        "--data_path", default=None,
        help="Dataset root directory (defaults to the standard path for each --dataset)",
    )

    # RoboCasa-specific paths
    parser.add_argument(
        "--atomic_data_path", default=DEFAULT_ATOMIC_DATA_PATH,
        help="RoboCasa original atomic dataset root (for task descriptions; robocasa only)",
    )

    parser.add_argument(
        "--prompts_json",
        default=os.path.join(os.path.dirname(__file__), "prompts.json"),
    )

    # Scope filters
    parser.add_argument("--task",  default=None, help="Comma-separated task name(s) to process (robocasa only)")
    parser.add_argument("--split", default=None, help="Process only this split (robocasa only)")

    # Mode
    parser.add_argument(
        "--mode", choices=["default", "fine_grain"], default="default",
        help="Annotation mode: default (full episode) or fine_grain (per-step with frame windows)",
    )
    parser.add_argument("--window_size", type=int, default=20,
                        help="fine_grain: number of future steps in the window")
    parser.add_argument("--stride", type=int, default=2,
                        help="fine_grain: stride for frame sampling within the window")
    parser.add_argument("--skip_step", type=int, default=1,
                        help="fine_grain: annotation every N-th step (1 = every step, 4 = steps 0,4,8,...)")

    # Annotation control
    parser.add_argument(
        "--annotation_type", default=None,
        help="Comma-separated annotation types to generate (default: all in prompts.json)",
    )
    parser.add_argument("--rerun",      action="store_true", help="Overwrite existing annotations")
    parser.add_argument("--no-context", action="store_true",
                        help="Omit CURRENT TASK and GT reasoning context; saves to annotations_[model]_no_context.jsonl")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument(
        "--n", type=int, default=None, metavar="N",
        help="Process at most N episodes per split (useful for quick tests)",
    )

    # Backend (vLLM, OpenAI-compatible server)
    parser.add_argument("--vllm_base_url",  default="http://localhost:8000/v1")
    parser.add_argument("--vllm_api_key",   default="EMPTY")
    parser.add_argument("--vllm_model",     default="Qwen3-VL-30B-A3B-Instruct")

    # Parallel worker sharding
    parser.add_argument("--worker_id", type=int, default=0,
                        help="Index of this worker (0-based). Use with --n_workers for parallel runs.")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Total number of parallel workers. Splits work across workers.")
    parser.add_argument("--ep_shard", action="store_true",
                        help="Shard by episode instead of task directory (writes per-worker files, merge after)")

    args = parser.parse_args()

    # Load prompts and decomposition
    prompts_config    = json.load(open(args.prompts_json))
    all_annotation_types = list(prompts_config["prompts"].keys())
    annotation_types     = (
        [ct.strip() for ct in args.annotation_type.split(",")]
        if args.annotation_type else all_annotation_types
    )

    # No primitive-decomposition ground truth is shipped; the 'reasoning' aspect
    # is generated from the prompt + on-screen context alone.
    decomp = {}
    if "reasoning" in annotation_types:
        print("[info] 'reasoning' runs without primitive-decomposition ground-truth context")

    # Build API client (vLLM, OpenAI-compatible)
    client             = openai.OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
    model              = args.vllm_model
    video_content_type = "video_url"

    adapter   = ADAPTERS[args.dataset]
    data_path = Path(args.data_path or DEFAULT_DATA_PATHS[args.dataset])
    n_limit   = args.n if args.n is not None else -1

    print(f"Dataset      : {args.dataset}  ({data_path})")
    print(f"Model        : {model}")
    print(f"Mode         : {args.mode}")
    if args.mode == "fine_grain":
        print(f"Window       : size={args.window_size}  stride={args.stride}  skip_step={args.skip_step}")
    print(f"Annotation types: {annotation_types}")
    if args.no_context:
        print(f"Context      : disabled (no_context)")
    if args.n is not None:
        print(f"Episode limit: {args.n} per split")
    if args.n_workers > 1:
        print(f"Worker       : {args.worker_id} / {args.n_workers}")
    print()

    if not data_path.exists():
        print(f"[error] data_path does not exist: {data_path}")
        return

    task_filter = set(args.task.split(",")) if args.task else None
    # When --ep_shard is set, disable task-level sharding in iter_splits
    # and let episode-level sharding in process_dataset_fine_grain handle it.
    iter_worker_id = 0 if args.ep_shard else args.worker_id
    iter_n_workers = 1 if args.ep_shard else args.n_workers
    for split_info in adapter.iter_splits(
        data_path, task_filter, args.split,
        worker_id=iter_worker_id, n_workers=iter_n_workers,
    ):
        print(f"[{split_info.display_name}]")
        if args.mode == "fine_grain":
            process_dataset_fine_grain(
                adapter, split_info,
                prompts_config, annotation_types,
                client, model,
                args.atomic_data_path, decomp,
                args.rerun, args.max_tokens, n_limit,
                window_size=args.window_size, stride=args.stride,
                skip_step=args.skip_step,
                no_context=args.no_context,
                worker_id=args.worker_id,
                n_workers=args.n_workers,
                ep_shard=args.ep_shard,
            )
        else:
            process_dataset(
                adapter, split_info,
                prompts_config, annotation_types,
                client, model,
                args.atomic_data_path, decomp,
                args.rerun, args.max_tokens, n_limit,
                video_content_type=video_content_type,
                no_context=args.no_context,
                worker_id=args.worker_id,
                n_workers=args.n_workers,
                ep_shard=args.ep_shard,
            )


if __name__ == "__main__":
    main()
