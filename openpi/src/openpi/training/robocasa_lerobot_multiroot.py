"""Multi-root raw-LeRobot RoboCasa dataset with mix5 caption sampling.

Trains directly off GR00T-LeRobot v2.1 roots (mp4 video + 12-dim actions) —
no packed/npz preprocessing — so Method-1 augmentation roots
(``composite_atomic_split_aug``) concat with their source roots
(``composite_atomic_split``) exactly as in the original Method-1 recipe: uniform
frame shuffle over the concatenation, mix ratio = frame-count ratio.

Per-root pipeline:
  LeRobotDataset (delta_timestamps -> stacked 12-dim action chunks)
    -> PromptFromLeRobotTask          (task_index -> per-subtask instruction)
    -> RawRoboCasaAdapter             (action[..., 5:12]; synthesize next_state)
    -> ReasoningMix5                  (episode_index -> caption sampled from
                                       meta/captions_<slug>.jsonl; "none" = "")

Captions are read straight from the dense_annotator output
(``meta/captions_<model_slug>.jsonl``: {"episode_index", "annotations": {aspect:
text}}); nothing is baked into episodes.jsonl. Roots without a captions file
contribute empty reasoning_text (logged) — acceptable for the aug clips only
until their transferred captions land, not for the source roots.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import random

import numpy as np
import torch

import openpi.transforms as _transforms

logger = logging.getLogger(__name__)

_STATE_DIM = 16
_RAW_ACTION_DIM = 12
_EFFECTIVE_SLICE = slice(5, 12)  # base(0:4)+mode(4) are zero/const in no-nav data


def discover_roots(base_dirs: tuple[str, ...]) -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    for base in base_dirs:
        base_path = pathlib.Path(base).expanduser()
        found, skipped = [], []
        for p in sorted(base_path.glob("**/lerobot")):
            if ((p / "data" / "chunk-000").is_dir()
                    and (p / "meta" / "info.json").is_file()
                    and (p / "meta" / "episodes.jsonl").is_file()):
                found.append(p)
            else:
                skipped.append(p)
        if not found:
            raise FileNotFoundError(f"No complete lerobot roots under {base_path}")
        for p in skipped:  # e.g. CuttingToolSelection/* ship without meta/info.json
            logger.warning(f"Skipping incomplete lerobot root: {p}")
        roots.extend(found)
    return roots


class PromptFromEpisodes:
    """Prompt from episodes.jsonl's per-episode task string.

    ~90/263 split roots carry stale global task_index values in their parquets
    against single-entry tasks.jsonl files, so task-table lookups are unsafe.
    episodes.jsonl `tasks` is always correct (the same source the original
    Method-1 run used via prompt_from_episode)."""

    def __init__(self, root: pathlib.Path):
        self._prompts: dict[int, str] = {}
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
            if line.strip():
                ep = json.loads(line)
                tasks = ep.get("tasks") or [""]
                self._prompts[int(ep["episode_index"])] = str(tasks[0])

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        data["prompt"] = self._prompts.get(int(data["episode_index"]), "")
        return data


@dataclasses.dataclass(frozen=True)
class RawRoboCasaAdapter(_transforms.DataTransformFn):
    """Adapt a raw GR00T-LeRobot item to the processed-dataset schema."""

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        action = np.asarray(data["action"])
        if action.shape[-1] == _RAW_ACTION_DIM:
            data["action"] = action[..., _EFFECTIVE_SLICE]
        state = np.asarray(data["observation.state"])
        if "observation.next_state" not in data:
            data["observation.next_state"] = np.zeros(_STATE_DIM, dtype=np.float32)
            data["observation.next_state_mask"] = np.zeros(1, dtype=np.float32)
        del state
        return data


@dataclasses.dataclass(frozen=True)
class AnchorTarget(_transforms.DataTransformFn):
    """I1b future-state anchor target, computed in-batch (no global-index .npy).

    From a long action window (fetched via extended delta_timestamps), compute the
    cumulative EEF-position displacement at each future horizon and a validity mask
    (0 past the episode end), then truncate the action back to the flow chunk so the
    rest of the pipeline is unchanged. Runs after RawRoboCasaAdapter (action already
    sliced to the 7-dim effective vector; pos = action[:, pos_slice]). Delivered as
    data['ee_physics_target'] = [n_horizons, 4] = (3 raw cumulative-disp + 1 mask);
    z-scored downstream in the model with registered per-(horizon, dim) stats.
    """

    horizons: tuple[int, ...]
    pos_slice: tuple[int, int]
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        action = np.asarray(data["action"], dtype=np.float32)  # [W, 7], W >= max(horizons)+1
        w = action.shape[0]
        is_pad = np.asarray(data.get("action_is_pad", np.zeros(w, dtype=bool))).reshape(-1)
        lo, hi = self.pos_slice
        # action[t] is the EEF Δpos applied at step t; net displacement after h steps
        # (actions[0..h-1]) = cumsum(action[:, pos])[h-1].
        cum = np.cumsum(action[:, lo:hi], axis=0)  # [W, 3]
        tgt = np.zeros((len(self.horizons), 4), dtype=np.float32)
        for i, h in enumerate(self.horizons):
            if h <= w and not bool(is_pad[h - 1]):
                tgt[i, :3] = cum[h - 1]
                tgt[i, 3] = 1.0
        data["ee_physics_target"] = tgt
        data["action"] = action[: self.action_horizon]  # restore flow chunk length
        data.pop("action_is_pad", None)
        return data


class ReasoningMix5:
    """Per-access caption sampling from the dense-annotator output.

    sample_keys is a comma-separated list; the literal key "none" contributes an
    empty string (task-description-only sample). Aspects missing for an episode
    simply drop out of that episode's pool.
    """

    def __init__(self, root: pathlib.Path, model_slug: str, sample_keys: str):
        self._keys = [k.strip() for k in sample_keys.split(",") if k.strip()]
        self._pools: dict[int, list[str]] = {}
        caps_file = root / "meta" / f"captions_{model_slug}.jsonl"
        n_caps = 0
        if caps_file.exists():
            for line in caps_file.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ann = rec.get("annotations", {})
                pool = ["" if k == "none" else ann.get(k, "") for k in self._keys]
                pool = [p for k, p in zip(self._keys, pool) if k == "none" or p]
                self._pools[int(rec["episode_index"])] = pool
                n_caps += 1
        if n_caps == 0:
            logger.warning(f"No captions at {caps_file} — reasoning_text will be empty")

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        ep = int(data["episode_index"])
        pool = self._pools.get(ep)
        data["reasoning_text"] = random.choice(pool) if pool else ""
        return data


def build_multiroot_dataset(
    base_dirs: tuple[str, ...],
    action_horizon: int,
    action_sequence_keys: tuple[str, ...],
    caption_model_slug: str,
    caption_sample_keys: str | None,
    primitive_slug: str | None = None,
    primitive_base_dirs: tuple[str, ...] = (),
    primitive_repeat: int = 1,
    primitive_only: bool = False,
    anchor_horizons: tuple[int, ...] = (),
    anchor_pos_slice: tuple[int, int] = (0, 3),
) -> torch.utils.data.ConcatDataset:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    from openpi.training.data_loader import TransformedDataset

    # Pure-primitive pretraining stage: skip the atomic-split task frames, train
    # on primitive (segment, instruction) samples only.
    if primitive_only:
        if not primitive_slug:
            raise ValueError("primitive_only=True requires primitive_slug")
        from openpi.training.primitive_dataset import build_primitive_datasets

        prim_dirs = primitive_base_dirs or base_dirs
        prim = build_primitive_datasets(tuple(prim_dirs), action_horizon, primitive_slug)
        parts_only = []
        for _ in range(max(1, primitive_repeat)):
            parts_only.extend(prim)
        logger.info(f"Primitive-ONLY pretrain: {len(prim)} roots, "
                    f"{sum(len(p) for p in prim)} distinct x{primitive_repeat}")
        return torch.utils.data.ConcatDataset(parts_only)

    # I1b anchor: fetch a longer action window so the far horizons are available;
    # AnchorTarget computes the target then truncates action back to action_horizon.
    anchor_fetch = max(anchor_horizons) + 1 if anchor_horizons else 0
    action_fetch = max(action_horizon, anchor_fetch)

    roots = discover_roots(base_dirs)
    parts: list = []
    total_frames = 0
    missing_caps = 0
    for root in roots:
        meta = lerobot_dataset.LeRobotDatasetMetadata(str(root))
        ds = lerobot_dataset.LeRobotDataset(
            str(root),
            delta_timestamps={
                key: [t / meta.fps for t in range(action_fetch)]
                for key in action_sequence_keys
            },
            video_backend="pyav",  # no system ffmpeg on these boxes; torchcodec fails
        )
        # Pad the task table so lerobot's internal item["task"] lookup cannot
        # KeyError on the stale task_index values (see PromptFromEpisodes).
        import collections

        ds.meta.tasks = collections.defaultdict(str, ds.meta.tasks)
        tfs: list = [
            PromptFromEpisodes(root),
            RawRoboCasaAdapter(),
        ]
        if anchor_horizons:
            tfs.append(AnchorTarget(tuple(anchor_horizons), tuple(anchor_pos_slice), action_horizon))
        if caption_sample_keys:
            mix = ReasoningMix5(root, caption_model_slug, caption_sample_keys)
            if not mix._pools:
                missing_caps += 1
            tfs.append(mix)
        parts.append(TransformedDataset(ds, tfs))
        total_frames += meta.total_frames
    logger.info(
        f"Multi-root RoboCasa dataset: {len(parts)} roots, {total_frames} frames"
        + (f", {missing_caps} roots WITHOUT captions" if missing_caps else "")
    )
    # Mode-A: append primitive (segment, instruction) samples to the concat.
    if primitive_slug:
        from openpi.training.primitive_dataset import build_primitive_datasets

        prim_dirs = primitive_base_dirs or base_dirs
        prim = build_primitive_datasets(tuple(prim_dirs), action_horizon, primitive_slug)
        n_prim = sum(len(p) for p in prim)
        # ConcatDataset samples proportional to length, so repeat the primitive
        # parts to lift them from a negligible fraction (~1%) to a trainable
        # share of the mix (~15-20%). Distinct samples are unchanged; each is
        # just drawn more often per epoch.
        for _ in range(max(1, primitive_repeat)):
            parts.extend(prim)
        frac = n_prim * primitive_repeat / max(1, total_frames + n_prim * primitive_repeat)
        logger.info(f"Mode-A primitives appended: {len(prim)} roots, {n_prim} distinct "
                    f"x{primitive_repeat} = {n_prim*primitive_repeat} samples "
                    f"({100*frac:.0f}% of mix) -> concat of {len(parts)} parts")
    return torch.utils.data.ConcatDataset(parts)
