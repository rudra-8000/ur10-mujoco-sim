#!/usr/bin/env python3
"""Smooth colour gain map for the top camera: the real board is not uniformly coloured (lighting gradient / white balance across the image).
Per-pixel median over many real frames (robot mostly removed) vs the same for the sim, both heavily blurred; gain = real / sim.  -> appearance/top_gainmap.npy
python fit_top_gainmap.py"""
import copy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_view as SV
SV._setup_gl("egl")
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
import build_scene as B
cfg0 = B.load_config(); cfg0["cameras"]["top"].pop("appearance", None)
EPS = [("circular_hole_v21_clean", 0), ("circular_hole_v21_clean", 20), ("circular_hole_v21_clean", 60), ("square_hole_v21_clean", 60), ("square_hole_v21_clean", 13), ("square_hole_v21_clean", 100)]
R, S = [], []
for ds, ep in EPS:
    n = len(pd.read_parquet(B.DATASETS / ds / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"))
    for fr in np.linspace(0.05, 0.9, 8) * n:
        cfg = copy.deepcopy(cfg0); cfg["robot"]["source"] = "dataset"; cfg["robot"]["dataset"] = dict(name=ds, episode=ep, frame=int(fr)); cfg["render"]["settle_steps"] = 50
        sc = B.Scene(cfg); S.append(sc.render(640, 480)["top"]); sc.close(); R.append(B.real_frames(cfg, 640, 480)[0])
    print(ds, ep, flush=True)
Rm, Sm = np.median(np.stack(R), 0), np.median(np.stack(S), 0)
sm = lambda x: np.stack([gaussian_filter(x[..., c], 30, mode="nearest") for c in range(3)], -1)
gain = np.clip(sm(Rm) / np.maximum(sm(Sm), 1.0), 0.4, 2.5)
small = gain[::20, ::20]                               # 24 x 32 x 3
np.save(Path(__file__).resolve().parent / "appearance" / "top_gainmap.npy", small.astype(np.float32))
print("gain map", small.shape, "min", small.min((0, 1)).round(2), "max", small.max((0, 1)).round(2), "centre", small[12, 16].round(2))
