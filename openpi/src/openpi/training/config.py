"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.rlbench_policy as rlbench_policy
import openpi.policies.robocasa_policy as robocasa_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# NOTE (DeMiAn release): parameter-freeze filters were nnx (JAX) based and are
# unused by the PyTorch trainer (scripts/train_pytorch.py). `Filter` is kept as an
# opaque alias only for API/type compatibility of TrainConfig.freeze_filter.
Filter: TypeAlias = Any

_RLBENCH_RAW_PHYSICS_SCALE = (
    0.08, 0.08, 0.08,  # cumulative displacement xyz, meters over the action chunk
    0.35, 0.35, 0.35,  # cumulative rotation xyz, radians over the action chunk
    0.50, 0.50, 0.50,  # gripper one-hot 3-way
    0.01,              # mean z velocity/delta, meters per step
    0.30,              # straight-line / arc-length
    0.01,              # mean translation step, meters
)


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Caption key for per-step fine-grained captions (e.g. "reasoning-short").
    # When set, RoboCasaDataset will use per-step captions instead of episode-level reasoning.
    caption_key: str | None = None
    # Stride (in frames) between fine-grained caption steps.
    # Maps frame_idx to step via floor-division: step = (frame_idx // fg_stride) * fg_stride.
    fg_stride: int | None = None

    # Multi-root raw-LeRobot mode (Method-1 aug training; robocasa_lerobot_multiroot.py).
    # When non-empty, the dataset is the concat of every */lerobot root under these
    # base dirs (mp4-backed, no packed/npz needed); repo_id is only used as a label.
    dataset_base_dirs: Sequence[str] = ()
    # dense_annotator output file: meta/captions_<slug>.jsonl in each root.
    caption_model_slug: str = "Qwen3-VL-30B-A3B-Instruct"
    # mix5-style per-access sampling, e.g. "none,current_action,environment-simple,
    # agent_arm_state,reasoning-short" ("none" = task-description only).
    caption_sample_keys: str | None = None
    # Mode-A: when set, append primitive (segment, instruction) samples read from
    # meta/primitives_<slug>_clean.jsonl in each root (primitive_dataset.py).
    primitive_slug: str | None = None
    primitive_repeat: int = 1
    primitive_only: bool = False

    # I1b anchor target (multiroot mode). When non-empty, the loader fetches a longer
    # action window and attaches observation.ee_physics_target for the anchor aux head.
    # Populated from the model config in LeRobotRoboCasaDataConfig.create.
    anchor_horizons: tuple[int, ...] = ()
    anchor_pos_slice: tuple[int, int] = (0, 3)

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            # NOTE (DeMiAn release): PI0_FAST tokenizer path removed (FAST models not included).


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            # Alias `next_state` to `state` for the next_state_aux loss — both
            # are drawn from the same distribution (frames from the same episodes),
            # so re-using the state's per-dim mean/std is correct (and avoids the
            # need to regenerate norm stats).
            if "state" in norm_stats and "next_state" not in norm_stats:
                norm_stats["next_state"] = norm_stats["state"]
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRLBenchDataConfig(DataConfigFactory):
    """Data config for RLBench datasets converted to LeRobot format.

    Expected raw LeRobot keys:
      - image: front camera RGB
      - wrist_image: wrist camera RGB
      - state: 7D EE pose/gripper state
      - actions: 7D EE delta rotvec/gripper action
      - task: language prompt
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[rlbench_policy.RLBenchInputs(model_type=model_config.model_type)],
            outputs=[rlbench_policy.RLBenchOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRoboCasaDataConfig(DataConfigFactory):
    """Data config for fine-tuning on the merged RoboCasa manipulation dataset.

    Expects a merged LeRobot dataset with:
      - 3 cameras: robot0_agentview_left, robot0_eye_in_hand, robot0_agentview_right
      - 7-dim actions: ee_position(3) + ee_rotation(3) + gripper(1)
      - 16-dim state
      - Task descriptions as prompts
    """

    # Include reasoning_text in the data pipeline (for LM training).
    include_reasoning: bool = False

    # Caption key for per-step fine-grained captions (e.g. "reasoning-short").
    # When set, RoboCasaDataset will use per-step captions instead of episode-level reasoning.
    caption_key: str | None = None
    # Stride (in frames) between fine-grained caption steps.
    fg_stride: int | None = None

    # Multi-root raw-LeRobot mode (see DataConfig.dataset_base_dirs).
    dataset_base_dirs: Sequence[str] = ()
    caption_sample_keys: str | None = None
    primitive_slug: str | None = None
    primitive_repeat: int = 1
    primitive_only: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_keys = {
            "observation/agentview_left": "observation.images.robot0_agentview_left",
            "observation/eye_in_hand": "observation.images.robot0_eye_in_hand",
            "observation/agentview_right": "observation.images.robot0_agentview_right",
            "observation/state": "observation.state",
            "observation/next_state": "observation.next_state",
            "observation/next_state_mask": "observation.next_state_mask",
            "actions": "action",
            "prompt": "prompt",
        }
        if self.include_reasoning:
            repack_keys["reasoning_text"] = "reasoning_text"
        # I1b anchor: enable when the model config carries anchor_aux_weight > 0. The
        # loader-side AnchorTarget produces observation.ee_physics_target; keep it
        # through the repack so it reaches Observation.from_dict.
        anchor_horizons: tuple[int, ...] = ()
        if getattr(model_config, "anchor_aux_weight", 0.0) > 0.0:
            anchor_horizons = tuple(getattr(model_config, "anchor_horizons_frames", ()))
            repack_keys["ee_physics_target"] = "ee_physics_target"

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_keys)]
        )

        data_transforms = _transforms.Group(
            inputs=[robocasa_policy.RoboCasaInputs(model_type=model_config.model_type)],
            outputs=[robocasa_policy.RoboCasaOutputs()],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            caption_key=self.caption_key,
            fg_stride=self.fg_stride,
            dataset_base_dirs=tuple(self.dataset_base_dirs),
            caption_sample_keys=self.caption_sample_keys,
            primitive_slug=self.primitive_slug,
            primitive_repeat=self.primitive_repeat,
            primitive_only=self.primitive_only,
            anchor_horizons=anchor_horizons,
            anchor_pos_slice=tuple(getattr(model_config, "anchor_pos_slice", (0, 3))),
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=lambda: None)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self):
        """Trainable-parameter filter was nnx (JAX) based and is unused by the
        PyTorch trainer, which sets requires_grad directly. Returns None."""
        return None

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
# NOTE (DeMiAn release): the upstream example registry (Aloha/Libero/DROID/
# pi0-FAST/RoboArena/PolaRiS configs) has been removed — none are used by this
# paper, and several depend on the dropped JAX FAST models. The MolmoBot DeMiAn-VLA
# configs are defined in code/molmobot_pi0/config_openpi.py and register themselves
# into _CONFIGS_DICT below. The RoboCasa365 DeMiAn-VLA configs (paper Table 1) are
# registered here, since they are openpi-native (LeRobotRoboCasaDataConfig +
# RoboCasaDataset). Dataset/checkpoint paths are placeholders — override at submit
# time via `--data.repo-id`, `--data.assets.assets-dir`, `--pytorch-weight-path`
# (see scripts/slurm/train_demian_vla_robocasa.slurm). The annotation aspect is
# selected with `--data.caption-key` (none / current_action / environment-simple /
# agent_arm_state / reasoning-short) and `--data.fg-stride`.
_CONFIGS: list["TrainConfig"] = [
    # Baseline (task-only) — fine-tune pi05_base on the RoboCasa 365 atomic split.
    TrainConfig(
        name="pi05_robocasa",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotRoboCasaDataConfig(
            repo_id="robocasa_lerobot",  # override via --data.repo-id
            assets=AssetsConfig(asset_id="robocasa", assets_dir="robocasa_lerobot/assets"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=None,  # override via --pytorch-weight-path
        num_train_steps=30_000,
        save_interval=2000,
        keep_period=5000,
    ),
    # DeMiAn-VLA RoboCasa variant: annotation auxiliary LM loss (lambda_LM=0.1).
    # Aspect selected at submit time via --data.caption-key (+ --data.fg-stride).
    TrainConfig(
        name="pi05_robocasa_lm",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            max_token_len=384, lm_loss_weight=0.1,
        ),
        data=LeRobotRoboCasaDataConfig(
            repo_id="robocasa_lerobot",  # override via --data.repo-id
            assets=AssetsConfig(asset_id="robocasa", assets_dir="robocasa_lerobot/assets"),
            base_config=DataConfig(prompt_from_task=True),
            include_reasoning=True,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=None,  # override via --pytorch-weight-path
        num_train_steps=30_000,
        save_interval=2000,
        keep_period=5000,
    ),
    # ------------------------------------------------------------------
    # Language x initial-state-augmentation study (2026-07): fine-tune the
    # released DeMiAn-VLA on the composite atomic split, +/- Method-1
    # front-buffer augmentation, with mix5 caption sampling. Trains directly
    # off the raw mp4 lerobot roots (robocasa_lerobot_multiroot.py). The two
    # configs differ ONLY in dataset_base_dirs — matched budget/init/captions.
    # ------------------------------------------------------------------
    TrainConfig(
        name="pi05_robocasa_lm_composite",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            max_token_len=384, lm_loss_weight=0.1,
        ),
        data=LeRobotRoboCasaDataConfig(
            repo_id="robocasa_composite_multiroot",
            assets=AssetsConfig(
                asset_id="robocasa_composite_split",
                assets_dir="/home/bkim/demian/openpi/assets/robocasa_composite",
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_reasoning=True,
            dataset_base_dirs=(
                "/home/bkim/orchest_data/robocasa_v1.0/composite_atomic_split",
            ),
            caption_sample_keys="none,current_action,environment-simple,agent_arm_state,reasoning-short",
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/home/bkim/ckpts/demian/DeMiAn-VLA/openpi",
        num_train_steps=20_000,
        save_interval=2000,
        keep_period=5000,
        num_workers=16,
    ),
    # ------------------------------------------------------------------
    # CURRICULUM Stage 1 — primitive PRETRAIN from openpi-0.5 base.
    #   pi05_base -> [primitive] -> atomic -> composite (stages 2,3 continue
    #   from this checkpoint). Trains on primitive (segment, instruction) pairs
    #   ONLY (primitive_only), so the base model learns the low-level
    #   primitive-instruction -> action vocabulary before task specialization.
    #   Checkpoint every 2k so we can read the trend on atomic + composite.
    # ------------------------------------------------------------------
    TrainConfig(
        name="pi05_robocasa_prim_pretrain",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            max_token_len=384, lm_loss_weight=0.1,
        ),
        data=LeRobotRoboCasaDataConfig(
            repo_id="robocasa_prim_pretrain",
            assets=AssetsConfig(
                asset_id="robocasa_composite_split",
                assets_dir="/home/bkim/demian/openpi/assets/robocasa_composite",
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_reasoning=True,
            dataset_base_dirs=(
                "/home/bkim/orchest_data/robocasa_v1.0/composite_atomic_split",
            ),
            primitive_slug="Qwen3-VL-30B-A3B-Instruct",
            primitive_only=True,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/home/bkim/ckpts/pi05_base_pytorch",
        num_train_steps=12_000,
        save_interval=2000,
        keep_period=2000,
        num_workers=16,
    ),
    # ------------------------------------------------------------------
    # Mode-A PRIMITIVE arm: DeMiAn-VLA init + composite_atomic_split (mix5
    # captions) + PRIMITIVE (segment, instruction) pairs where the primitive
    # imperative is the prompt. The training-side attack on the commitment
    # bottleneck (analysis Part 5): teaches primitive-instruction -> action so
    # an S2 instructor can steer mid-subtask. = the ctrl config + primitive_slug.
    # Judge on the primitive-following probe + instructor-mode SR, NOT oracle SR.
    # ------------------------------------------------------------------
    TrainConfig(
        name="pi05_robocasa_lm_prim",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            max_token_len=384, lm_loss_weight=0.1,
        ),
        data=LeRobotRoboCasaDataConfig(
            repo_id="robocasa_composite_multiroot_prim",
            assets=AssetsConfig(
                asset_id="robocasa_composite_split",
                assets_dir="/home/bkim/demian/openpi/assets/robocasa_composite",
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_reasoning=True,
            dataset_base_dirs=(
                "/home/bkim/orchest_data/robocasa_v1.0/composite_atomic_split",
            ),
            caption_sample_keys="none,current_action,environment-simple,agent_arm_state,reasoning-short",
            primitive_slug="Qwen3-VL-30B-A3B-Instruct",
            primitive_repeat=25,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/home/bkim/ckpts/demian/DeMiAn-VLA/openpi",
        num_train_steps=20_000,
        save_interval=2000,
        keep_period=5000,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_robocasa_lm_composite_aug",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            max_token_len=384, lm_loss_weight=0.1,
        ),
        data=LeRobotRoboCasaDataConfig(
            repo_id="robocasa_composite_multiroot_aug",
            assets=AssetsConfig(
                asset_id="robocasa_composite_split",
                assets_dir="/home/bkim/demian/openpi/assets/robocasa_composite",
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_reasoning=True,
            dataset_base_dirs=(
                "/home/bkim/orchest_data/robocasa_v1.0/composite_atomic_split",
                "/home/bkim/orchest_data/robocasa_v1.0/composite_atomic_split_aug_v2",
            ),
            caption_sample_keys="none,current_action,environment-simple,agent_arm_state,reasoning-short",
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/home/bkim/ckpts/demian/DeMiAn-VLA/openpi",
        num_train_steps=20_000,
        save_interval=2000,
        keep_period=5000,
        num_workers=16,
    ),
    # ------------------------------------------------------------------
    # SOTA-chase arm: the aug arm + BOTH proven auxiliary losses stacked.
    #   - I1a (deriv_aux 0.1): derivative/Sobolev on the flow head's implied
    #     clean action — the promoted, zero-param, no-inference-change win.
    #   - physics_aux (0.1): in-batch future-state summary off the action
    #     expert (I1b's within-chunk portion; aux-target delivery = the SAFE
    #     mode per phase4-report, +7.3pp atomic). Both are "predict/match"
    #     losses, not output constraints, so mode-collapse risk is low; the
    #     time-weight on I1a keeps it benign on the flow head.
    # The still-running aug arm (no aux) is the free ablation anchor:
    # this arm minus that one isolates (I1a + physics_aux).
    # ------------------------------------------------------------------
    TrainConfig(
        name="pi05_robocasa_lm_composite_aug_i1",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            max_token_len=384, lm_loss_weight=0.1,
            deriv_aux_weight=0.1, deriv_action_slice=(0, 6),
            physics_aux_weight=0.1,
        ),
        data=LeRobotRoboCasaDataConfig(
            repo_id="robocasa_composite_multiroot_aug_i1",
            assets=AssetsConfig(
                asset_id="robocasa_composite_split",
                assets_dir="/home/bkim/demian/openpi/assets/robocasa_composite",
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_reasoning=True,
            dataset_base_dirs=(
                "/home/bkim/orchest_data/robocasa_v1.0/composite_atomic_split",
                "/home/bkim/orchest_data/robocasa_v1.0/composite_atomic_split_aug_v2",
            ),
            caption_sample_keys="none,current_action,environment-simple,agent_arm_state,reasoning-short",
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/home/bkim/ckpts/demian/DeMiAn-VLA/openpi",
        num_train_steps=20_000,
        save_interval=2000,
        keep_period=5000,
        num_workers=16,
    ),
    # ==================================================================
    # CURRICULUM with I1b future-state anchor aux (hierarchy's composite-proven
    # lever, +6.5pp -dynamic p=0.00073). anchor_aux_weight>0 -> the loader fetches
    # a 161-frame action window and attaches observation.ee_physics_target; the
    # anchor head off pooled suffix latents regresses cumulative EEF-pos
    # displacement at {5,10,20,40,80,160}. Never read at inference. Curriculum:
    # Stage 1 primitive pretrain (no anchor; segments too short) is a SEPARATE
    # config already running -> its final ckpt inits Stage 2 (atomic) -> Stage 3
    # (composite). anchor_target_mean/std left empty until compute_anchor_stats.py.
    # ==================================================================
    TrainConfig(
        name="pi05_robocasa_atomic_i1b",  # Stage 2: atomic (init from Stage-1 primitive)
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            max_token_len=384, lm_loss_weight=0.1,
            anchor_aux_weight=0.1,
            # z-stats from compute_anchor_stats.py (composite; atomic ~ same action space).
            anchor_target_mean=(-0.00644, 0.04620, -0.04763, -0.09077, -0.00220, -0.02313, -0.27981, -0.17047, 0.03344, 0.00190, -0.80571, -0.42820, 1.88683, -1.44289, -1.78842, 1.70492, -2.21208, -2.06273),
            anchor_target_std=(1.50391, 1.59244, 1.48343, 2.95065, 3.16610, 2.96658, 5.54637, 5.89488, 5.51407, 9.88085, 10.99075, 9.85736, 13.65520, 18.87670, 14.42240, 20.08450, 27.36267, 20.46720),
        ),
        data=LeRobotRoboCasaDataConfig(
            repo_id="robocasa_atomic_i1b",
            assets=AssetsConfig(
                # Reuse composite norm_stats across all curriculum stages so the action
                # normalization is consistent from primitive -> atomic -> composite
                # (same 16-dim state / 7-dim EEF action space; atomic range ⊂ composite).
                asset_id="robocasa_composite_split",
                assets_dir="/home/bkim/demian/openpi/assets/robocasa_composite",
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_reasoning=True,
            dataset_base_dirs=(
                "/home/bkim/orchest_data/robocasa_v1.0/robocasa_atomic_navfiltered",
            ),
            caption_sample_keys="none,current_action,environment-simple,agent_arm_state,reasoning-short",
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/home/bkim/ckpts/curriculum/stage1_prim",  # Stage-1 final; set by the gate
        num_train_steps=30_000,
        save_interval=5000,
        keep_period=10000,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_robocasa_composite_i1b",  # Stage 3: composite (init from Stage-2 atomic)
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            max_token_len=384, lm_loss_weight=0.1,
            anchor_aux_weight=0.1,
            # z-stats from compute_anchor_stats.py over composite_atomic_split.
            anchor_target_mean=(-0.00644, 0.04620, -0.04763, -0.09077, -0.00220, -0.02313, -0.27981, -0.17047, 0.03344, 0.00190, -0.80571, -0.42820, 1.88683, -1.44289, -1.78842, 1.70492, -2.21208, -2.06273),
            anchor_target_std=(1.50391, 1.59244, 1.48343, 2.95065, 3.16610, 2.96658, 5.54637, 5.89488, 5.51407, 9.88085, 10.99075, 9.85736, 13.65520, 18.87670, 14.42240, 20.08450, 27.36267, 20.46720),
        ),
        data=LeRobotRoboCasaDataConfig(
            repo_id="robocasa_composite_i1b",
            assets=AssetsConfig(
                asset_id="robocasa_composite_split",
                assets_dir="/home/bkim/demian/openpi/assets/robocasa_composite",
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_reasoning=True,
            dataset_base_dirs=(
                "/home/bkim/orchest_data/robocasa_v1.0/composite_atomic_split",
            ),
            caption_sample_keys="none,current_action,environment-simple,agent_arm_state,reasoning-short",
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/home/bkim/ckpts/curriculum/stage2_atomic",  # Stage-2 final; set by the gate
        num_train_steps=20_000,
        save_interval=2000,
        keep_period=5000,
        num_workers=16,
    ),
    # ==================================================================
    # COMBINED "everything" curriculum (jeju-2): every positive lever stacked —
    # curriculum (pi05_base -> primitive -> atomic -> composite) + I1a (deriv/Sobolev,
    # deriv_aux 0.1) on ALL stages + I1b (future-state anchor 0.1) on atomic+composite.
    # Runs alongside the I1b-only curriculum (jeju-3/jeju-0) and the I1b-standalone arm
    # (ssc-3) -> an ablation ladder: I1b-standalone / curriculum+I1b / curriculum+I1a+I1b.
    # ==================================================================
    TrainConfig(
        name="pi05_robocasa_prim_i1a",  # Stage 1 combined: primitive pretrain + I1a
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            max_token_len=384, lm_loss_weight=0.1,
            deriv_aux_weight=0.1, deriv_action_slice=(0, 6),
        ),
        data=LeRobotRoboCasaDataConfig(
            repo_id="robocasa_prim_i1a",
            assets=AssetsConfig(
                asset_id="robocasa_composite_split",
                assets_dir="/home/bkim/demian/openpi/assets/robocasa_composite",
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_reasoning=True,
            dataset_base_dirs=(
                "/home/bkim/orchest_data/robocasa_v1.0/composite_atomic_split",
            ),
            primitive_slug="Qwen3-VL-30B-A3B-Instruct",
            primitive_only=True,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/home/bkim/ckpts/pi05_base_pytorch",
        num_train_steps=12_000,
        save_interval=2000,
        keep_period=2000,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_robocasa_atomic_i1ab",  # Stage 2 combined: atomic + I1a + I1b
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            max_token_len=384, lm_loss_weight=0.1,
            deriv_aux_weight=0.1, deriv_action_slice=(0, 6),
            anchor_aux_weight=0.1,
            anchor_target_mean=(-0.00644, 0.04620, -0.04763, -0.09077, -0.00220, -0.02313, -0.27981, -0.17047, 0.03344, 0.00190, -0.80571, -0.42820, 1.88683, -1.44289, -1.78842, 1.70492, -2.21208, -2.06273),
            anchor_target_std=(1.50391, 1.59244, 1.48343, 2.95065, 3.16610, 2.96658, 5.54637, 5.89488, 5.51407, 9.88085, 10.99075, 9.85736, 13.65520, 18.87670, 14.42240, 20.08450, 27.36267, 20.46720),
        ),
        data=LeRobotRoboCasaDataConfig(
            repo_id="robocasa_atomic_i1ab",
            assets=AssetsConfig(
                asset_id="robocasa_composite_split",
                assets_dir="/home/bkim/demian/openpi/assets/robocasa_composite",
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_reasoning=True,
            dataset_base_dirs=(
                "/home/bkim/orchest_data/robocasa_v1.0/robocasa_atomic_navfiltered",
            ),
            caption_sample_keys="none,current_action,environment-simple,agent_arm_state,reasoning-short",
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/home/bkim/ckpts/curriculum_i1ab/stage1",  # Stage-1 final; set by the gate
        num_train_steps=30_000,
        save_interval=5000,
        keep_period=10000,
        num_workers=16,
    ),
    TrainConfig(
        name="pi05_robocasa_composite_i1ab",  # Stage 3 combined: composite + I1a + I1b
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            max_token_len=384, lm_loss_weight=0.1,
            deriv_aux_weight=0.1, deriv_action_slice=(0, 6),
            anchor_aux_weight=0.1,
            anchor_target_mean=(-0.00644, 0.04620, -0.04763, -0.09077, -0.00220, -0.02313, -0.27981, -0.17047, 0.03344, 0.00190, -0.80571, -0.42820, 1.88683, -1.44289, -1.78842, 1.70492, -2.21208, -2.06273),
            anchor_target_std=(1.50391, 1.59244, 1.48343, 2.95065, 3.16610, 2.96658, 5.54637, 5.89488, 5.51407, 9.88085, 10.99075, 9.85736, 13.65520, 18.87670, 14.42240, 20.08450, 27.36267, 20.46720),
        ),
        data=LeRobotRoboCasaDataConfig(
            repo_id="robocasa_composite_i1ab",
            assets=AssetsConfig(
                asset_id="robocasa_composite_split",
                assets_dir="/home/bkim/demian/openpi/assets/robocasa_composite",
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_reasoning=True,
            dataset_base_dirs=(
                "/home/bkim/orchest_data/robocasa_v1.0/composite_atomic_split",
            ),
            caption_sample_keys="none,current_action,environment-simple,agent_arm_state,reasoning-short",
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/home/bkim/ckpts/curriculum_i1ab/stage2",  # Stage-2 final; set by the gate
        num_train_steps=20_000,
        save_interval=2000,
        keep_period=5000,
        num_workers=16,
    ),
]


if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
