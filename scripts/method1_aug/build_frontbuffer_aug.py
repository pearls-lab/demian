#!/usr/bin/env python3
"""Method-1 (B) builder: materialize front-buffer mini-clips of a 2nd+ subtask LeRobot phase dataset.
Each source episode -> M clips of length K+H (front-buffer K with valid action chunks + H lookahead),
at start offsets o=m*stride (1B). Image aug is applied at LOAD time (augmax), not baked in here.
Output is a GR00T-LeRobot v2.1 dataset matching the source schema (parquet + 3 mp4 + meta)."""
import json, os, argparse, glob
import numpy as np, pandas as pd
import imageio.v2 as imageio

def col_stats(series):
    v0 = series.iloc[0]
    if isinstance(v0, np.ndarray):
        a = np.stack(series.values).astype(np.float64)
    else:
        a = np.asarray(series.values, dtype=np.float64)[:, None]
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
            "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [int(a.shape[0])]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help=".../<Task>/<Phase>/<date>/lerobot")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()
    L_clip = a.K + a.H
    src, dst = a.src, a.dst
    info = json.load(open(f"{src}/meta/info.json"))
    fps = info["fps"]
    vid_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    chunks_size = info.get("chunks_size", 1000)
    src_eps = [json.loads(l) for l in open(f"{src}/meta/episodes.jsonl")]

    os.makedirs(f"{dst}/meta", exist_ok=True)
    for fn in ["tasks.jsonl", "modality.json", "embodiment.json"]:
        p = f"{src}/meta/{fn}"
        if os.path.exists(p):
            open(f"{dst}/meta/{fn}", "w").write(open(p).read())

    new_eps, new_stats = [], []
    gidx = new_ep = total_frames = 0
    for se in src_eps:
        sei = se["episode_index"]
        schunk = sei // chunks_size
        df = pd.read_parquet(f"{src}/data/chunk-{schunk:03d}/episode_{sei:06d}.parquet")
        L = len(df)
        vids = {vk: imageio.mimread(
                    f"{src}/videos/chunk-{schunk:03d}/{vk}/episode_{sei:06d}.mp4", memtest=False)
                for vk in vid_keys}
        for m in range(a.M):
            o = min(m * a.stride, max(0, L - L_clip))
            e = min(o + L_clip, L)
            n = e - o
            if n < a.K:  # episode shorter than one front buffer: skip extra variants
                if m > 0:
                    continue
                o, e, n = 0, L, L
            sub = df.iloc[o:e].copy().reset_index(drop=True)
            sub["frame_index"] = np.arange(n, dtype=np.int64)
            sub["episode_index"] = np.int64(new_ep)
            sub["index"] = (gidx + np.arange(n)).astype(np.int64)
            sub["timestamp"] = (np.arange(n) / fps).astype(np.float32)
            dchunk = new_ep // chunks_size
            os.makedirs(f"{dst}/data/chunk-{dchunk:03d}", exist_ok=True)
            sub.to_parquet(f"{dst}/data/chunk-{dchunk:03d}/episode_{new_ep:06d}.parquet")
            for vk in vid_keys:
                vd = f"{dst}/videos/chunk-{dchunk:03d}/{vk}"
                os.makedirs(vd, exist_ok=True)
                w = imageio.get_writer(f"{vd}/episode_{new_ep:06d}.mp4", fps=fps, codec="libx264",
                                       pixelformat="yuv420p", macro_block_size=1, ffmpeg_log_level="error")
                for fr in vids[vk][o:e]:
                    w.append_data(np.asarray(fr))
                w.close()
            new_eps.append({"episode_index": new_ep, "tasks": se["tasks"], "length": n,
                            "source_episode_index": int(sei), "source_offset": int(o),
                            "source_length": int(L)})
            new_stats.append({"episode_index": new_ep,
                              "stats": {c: col_stats(sub[c]) for c in df.columns}})
            gidx += n; total_frames += n; new_ep += 1

    with open(f"{dst}/meta/episodes.jsonl", "w") as f:
        for e in new_eps: f.write(json.dumps(e) + "\n")
    with open(f"{dst}/meta/episodes_stats.jsonl", "w") as f:
        for e in new_stats: f.write(json.dumps(e) + "\n")
    info2 = dict(info)
    info2.update(total_episodes=new_ep, total_frames=total_frames,
                 total_videos=new_ep * len(vid_keys),
                 total_chunks=(new_ep + chunks_size - 1) // chunks_size,
                 splits={"train": f"0:{new_ep}"})
    json.dump(info2, open(f"{dst}/meta/info.json", "w"), indent=4)
    print(f"DONE {dst}: {new_ep} clips, {total_frames} frames (src {len(src_eps)} eps)")

if __name__ == "__main__":
    main()
