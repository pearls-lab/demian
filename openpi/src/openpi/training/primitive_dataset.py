"""Mode-A primitive dataset: (segment, instruction) pairs where the primitive
instruction IS the prompt.

The commitment result (analysis Part 5) showed S1 ignores mid-motion
re-instruction because every training prompt spanned a whole atomic subtask.
This dataset teaches primitive-instruction -> action-chunk mapping directly:
each sample is a frame inside a motion primitive [start,end], with the action
chunk at that frame and the primitive's imperative instruction as `prompt`.

Reads meta/primitives_<slug>_clean.jsonl (from build_primitive_dataset.py +
the kinematic-consistency filter) and rides the same video-backed LeRobotDataset
as the multiroot loader — no repack.
"""

from __future__ import annotations

import json
import logging
import pathlib

import torch

import openpi.transforms as _transforms
from openpi.training.robocasa_lerobot_multiroot import RawRoboCasaAdapter

logger = logging.getLogger(__name__)


class _SetPrompt:
    """Force the prompt to the primitive instruction; clear reasoning_text."""

    def __init__(self, instruction: str):
        self._instr = instruction

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        data["prompt"] = self._instr
        data["reasoning_text"] = ""
        return data


class PrimitiveWindowDataset(torch.utils.data.Dataset):
    """Frames inside motion primitives, each labelled with the primitive prompt.

    Sample index: for every primitive [start,end] of every captioned episode, take
    up to `max_per_prim` start-frames f in [start, max(start, end-action_horizon)]
    (so the H-step action chunk stays within the primitive when possible); global
    frame = episode_from + f. Short primitives (< action_horizon) contribute their
    single start frame (the chunk then continues naturally past the boundary).
    """

    def __init__(self, root: pathlib.Path, action_horizon: int, slug: str,
                 max_per_prim: int = 4, frame_stride: int = 6):
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

        self._root = str(root)
        meta = lerobot_dataset.LeRobotDatasetMetadata(self._root)
        self._ds = lerobot_dataset.LeRobotDataset(
            self._root,
            delta_timestamps={"action": [t / meta.fps for t in range(action_horizon)]},
            video_backend="pyav",
        )
        # ~90/166 roots carry stale global task_index in their parquets against a
        # single-entry tasks.jsonl, so lerobot's internal item["task"] lookup
        # KeyErrors. Pad the table (same fix as the multiroot loader); we override
        # the prompt with the primitive instruction anyway, so "task" is unused.
        import collections

        self._ds.meta.tasks = collections.defaultdict(str, self._ds.meta.tasks)
        edi = self._ds.episode_data_index
        ep_from = {int(e): int(edi["from"][pos].item())
                   for pos, e in enumerate(self._ordered_episodes())}

        prim_file = root / "meta" / f"primitives_{slug}_clean.jsonl"
        if not prim_file.exists():
            prim_file = root / "meta" / f"primitives_{slug}.jsonl"
        self._samples: list[tuple[int, str]] = []
        if not prim_file.exists():
            logger.warning(f"no primitives file at {root}")
            return
        for line in prim_file.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ep = int(r["episode_index"])
            if ep not in ep_from:
                continue
            instr = r["instruction"].strip()
            if len(instr) < 4:
                continue
            start, end = int(r["start"]), int(r["end"])
            last = max(start, end - action_horizon)
            fs = list(range(start, last + 1, frame_stride))[:max_per_prim] or [start]
            base = ep_from[ep]
            for f in fs:
                self._samples.append((base + f, instr))

    def _ordered_episodes(self) -> list[int]:
        # LeRobotDataset without an `episodes` filter enumerates 0..N-1 in order;
        # episode_data_index rows align to that order.
        n = len(self._ds.episode_data_index["from"])
        return list(range(n))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, i: int) -> dict:
        gidx, instr = self._samples[i]
        item = dict(self._ds[gidx])
        item = RawRoboCasaAdapter()(item)
        return _SetPrompt(instr)(item)


def build_primitive_datasets(base_dirs: tuple[str, ...], action_horizon: int,
                             slug: str) -> list[PrimitiveWindowDataset]:
    """One PrimitiveWindowDataset per discovered captioned root."""
    from openpi.training.robocasa_lerobot_multiroot import discover_roots

    out = []
    total = 0
    for root in discover_roots(base_dirs):
        ds = PrimitiveWindowDataset(root, action_horizon, slug)
        if len(ds):
            out.append(ds)
            total += len(ds)
    logger.info(f"Primitive Mode-A: {len(out)} roots, {total} (segment,instruction) samples")
    return out
