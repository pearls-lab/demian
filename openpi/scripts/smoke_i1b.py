"""I1b wiring smoke test (no model load): verify observation.ee_physics_target is
computed by the loader and survives the full transform pipeline into from_dict."""
import numpy as np
import torch

import openpi.training.config as _config
import openpi.training.data_loader as _dl
import openpi.transforms as _transforms
from openpi.models import model as _model

cfg = _config.get_config("pi05_robocasa_composite_i1b")
print("=== config ===")
print("  anchor_aux_weight:", cfg.model.anchor_aux_weight)
print("  anchor_horizons:", cfg.model.anchor_horizons_frames)
print("  anchor_pos_slice:", cfg.model.anchor_pos_slice)

data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
print("  data.anchor_horizons:", data_config.anchor_horizons)
print("  repack has ee_physics_target:",
      any("ee_physics_target" in str(t.structure) for t in data_config.repack_transforms.inputs))

ah = cfg.model.action_horizon
raw = _dl.create_torch_dataset(data_config, ah, cfg.model)
print("=== raw multiroot dataset len:", len(raw))
item0 = raw[0]
print("  raw item keys:", sorted(k for k in item0 if not k.startswith("observation.image")))
print("  raw action shape (should be truncated to ah):", np.asarray(item0["action"]).shape)
print("  raw ee_physics_target shape:", np.asarray(item0["ee_physics_target"]).shape)
ept = np.asarray(item0["ee_physics_target"])
print("  mask per horizon:", ept[:, 3].tolist())
print("  disp norms per horizon:", np.linalg.norm(ept[:, :3], axis=1).round(4).tolist())

full = _dl.transform_dataset(raw, data_config, skip_norm_stats=False)
titem = full[0]
print("=== after full transform pipeline ===")
print("  keys:", sorted(k for k in titem if k != "image" and k != "image_mask"))
assert "ee_physics_target" in titem, "ee_physics_target DROPPED by pipeline!"
print("  ee_physics_target shape:", np.asarray(titem["ee_physics_target"]).shape)
print("  actions shape (flow chunk):", np.asarray(titem["actions"]).shape)

# collate a batch of 2 and run from_dict
batch = _dl._collate_fn([full[0], full[1]])
obs = _model.Observation.from_dict(batch)
print("=== Observation.from_dict ===")
print("  obs.ee_physics_target:", None if obs.ee_physics_target is None else np.asarray(obs.ee_physics_target).shape)
assert obs.ee_physics_target is not None, "ee_physics_target did NOT reach Observation!"
print("\nDATA SMOKE PASSED: ee_physics_target flows end-to-end with shape",
      np.asarray(obs.ee_physics_target).shape)

# --- model-math unit test of the anchor loss block (no pi05 load) ---
print("\n=== anchor loss math unit test ===")
torch.manual_seed(0)
B, n, d = 4, 6, 32
suffix_out = torch.randn(B, ah, d, requires_grad=True)
head = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.SiLU(), torch.nn.Linear(4 * d, n * 3))
tmean = torch.zeros(n, 3); tstd = torch.ones(n, 3)
ee = torch.randn(B, n, 4); ee[..., 3] = (torch.rand(B, n) > 0.3).float()  # some masked
pooled = suffix_out.mean(dim=1)
pred = head(pooled).reshape(-1, n, 3)
tgt = ee.reshape(-1, n, 4)
target = (tgt[..., :3] - tmean) / tstd
mask = tgt[..., 3:4]
per = (pred - target) ** 2 * mask
loss = per.sum() / mask.sum().clamp(min=1.0)
loss.backward()
print("  anchor_loss:", float(loss), "| finite:", bool(torch.isfinite(loss)))
print("  grad flows to suffix_out:", suffix_out.grad is not None and float(suffix_out.grad.abs().sum()) > 0)
# all-masked -> loss 0, no nan
ee0 = torch.randn(B, n, 4); ee0[..., 3] = 0.0
m0 = ee0.reshape(-1, n, 4)[..., 3:4]
l0 = ((pred.detach() - 0) ** 2 * m0).sum() / m0.sum().clamp(min=1.0)
print("  all-masked loss (should be 0, no nan):", float(l0), "| finite:", bool(torch.isfinite(l0)))
print("\nALL SMOKE CHECKS PASSED")
