import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboCasaInputs(transforms.DataTransformFn):
    """Converts RoboCasa observations to the model input format.

    RoboCasa has 3 cameras and a 7-dim arm action space (after stripping base_motion + control_mode):
      - action[0:3] = end_effector_position
      - action[3:6] = end_effector_rotation
      - action[6]   = gripper_close

    State is 16-dim: base_pos(3) + base_rot(4) + ee_pos_rel(3) + ee_rot_rel(4) + gripper_qpos(2).
    """

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        # Map RoboCasa cameras to model image slots.
        base_image = _parse_image(data["observation/agentview_left"])
        wrist_image = _parse_image(data["observation/eye_in_hand"])
        right_image = _parse_image(data["observation/agentview_right"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # I1b anchor target (train only): [n_horizons, 4] = (3 disp + 1 mask).
        if "ee_physics_target" in data:
            inputs["ee_physics_target"] = data["ee_physics_target"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Pass reasoning text separately for masked LM training.
        # Always pass the key (even if empty) so TokenizePrompt creates a consistent reasoning_token_mask.
        if "reasoning_text" in data:
            inputs["reasoning_text"] = data["reasoning_text"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboCasaOutputs(transforms.DataTransformFn):
    """Converts model outputs back to RoboCasa action space (7-dim)."""

    def __call__(self, data: dict) -> dict:
        # Return only first 7 action dims (strip model padding).
        return {"actions": np.asarray(data["actions"][:, :7])}
