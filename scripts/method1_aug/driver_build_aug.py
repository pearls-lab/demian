#!/usr/bin/env python3
"""Select 2nd+ phases (per-task argmin frame-0 eef-pos std = canonical-reset 'first' phase is skipped)
and build front-buffer aug clips for the rest, in parallel."""
import json, os, glob, sys, numpy as np, pandas as pd, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

S = "/home/bkim/orchest_data/robocasa_v1.0/composite_atomic_split"
OUT = "/home/bkim/orchest_data/robocasa_v1.0/composite_atomic_split_aug_v2"
BUILDER = "/home/bkim/demian/scripts/method1_aug/build_frontbuffer_aug.py"
PY = "/home/bkim/openpi-venv/bin/python"
K, H, M, STRIDE = 32, 16, 4, 4
NPROC = 24

def eef_std(lr, n=15):
    pqs = sorted(glob.glob(f"{lr}/data/**/*.parquet", recursive=True))[:n]
    f0 = []
    for pq in pqs:
        df = pd.read_parquet(pq, columns=["observation.state", "frame_index"])
        r = df[df.frame_index == 0]
        if len(r): f0.append(np.asarray(r["observation.state"].iloc[0])[7:10])
    return float(np.linalg.norm(np.stack(f0).std(0))) if f0 else 1e9

def select_pairs():
    pairs = []
    for task in sorted(os.listdir(S)):
        td = os.path.join(S, task)
        if not os.path.isdir(td): continue
        phases = [p for p in os.listdir(td) if os.path.isdir(os.path.join(td, p))]
        lrs = {p: glob.glob(os.path.join(td, p, "**", "lerobot"), recursive=True) for p in phases}
        phases = [p for p in phases if lrs[p]]
        if len(phases) < 2: continue
        stds = {p: eef_std(lrs[p][0]) for p in phases}
        first = min(stds, key=stds.get)
        for p in phases:
            if p == first: continue
            pairs.append((lrs[p][0], os.path.join(OUT, task, p, "lerobot")))
    return pairs

def build_one(src_dst):
    src, dst = src_dst
    if os.path.exists(f"{dst}/meta/info.json"):
        return (dst, "SKIP(exists)")
    os.makedirs(dst, exist_ok=True)
    r = subprocess.run([PY, BUILDER, "--src", src, "--dst", dst,
                        "--K", str(K), "--H", str(H), "--M", str(M), "--stride", str(STRIDE)],
                       capture_output=True, text=True)
    return (dst, (r.stdout.strip().splitlines() or [r.stderr.strip()[-200:]])[-1])

if __name__ == "__main__":
    pairs = select_pairs()
    print(f"=== {len(pairs)} 2nd+ phases to build (K={K} H={H} M={M}) ===", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=NPROC) as ex:
        futs = {ex.submit(build_one, p): p for p in pairs}
        for f in as_completed(futs):
            dst, msg = f.result(); done += 1
            print(f"[{done}/{len(pairs)}] {dst.split('/v1.0/')[-1]:<70} {msg}", flush=True)
    print("=== ALL AUG BUILDS DONE ===", flush=True)
