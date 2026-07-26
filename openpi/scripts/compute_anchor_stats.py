"""Compute per-(horizon, dim) z-score stats of the I1b anchor displacement target,
over valid (mask=1) samples, and print flat tuples to paste into the config."""
import sys
import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _dl

cfg_name = sys.argv[1] if len(sys.argv) > 1 else "pi05_robocasa_composite_i1b"
n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 400
cfg = _config.get_config(cfg_name)
data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
raw = _dl.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
H = list(cfg.model.anchor_horizons_frames)
n = len(H)
rng = np.random.default_rng(0)
idxs = rng.integers(0, len(raw), size=n_samples)

disp = [[] for _ in range(n)]  # per-horizon list of [3] valid displacements
for j, i in enumerate(idxs):
    ept = np.asarray(raw[int(i)]["ee_physics_target"])  # [n,4]
    for h in range(n):
        if ept[h, 3] > 0.5:
            disp[h].append(ept[h, :3])
    if (j + 1) % 100 == 0:
        print(f"  {j+1}/{n_samples} sampled", flush=True)

mean = np.zeros((n, 3), dtype=np.float64)
std = np.ones((n, 3), dtype=np.float64)
for h in range(n):
    if disp[h]:
        arr = np.stack(disp[h])
        mean[h] = arr.mean(0)
        std[h] = arr.std(0).clip(min=1e-3)
    print(f"  horizon {H[h]:4d}: n_valid={len(disp[h]):4d}  mean={mean[h].round(3)}  std={std[h].round(3)}")

print("\nanchor_target_mean=(" + ", ".join(f"{v:.5f}" for v in mean.flatten()) + "),")
print("anchor_target_std=(" + ", ".join(f"{v:.5f}" for v in std.flatten()) + "),")
