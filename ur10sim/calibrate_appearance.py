#!/usr/bin/env python3
"""Fit how the sim cameras should look like the real ones: Gaussian blur, per-channel colour map (quantile matching) and sensor noise, per camera.

  python calibrate_appearance.py [--frames 6]     -> appearance/{top,wrist}.json ; scene_config.yaml cameras.<name>.appearance points at them
Pairs: dataset frames (sim scene driven by the recorded joints of the same frame, real frame from the video).  Content differs a little between
sim and real (peg carried or not ...), so the colour map is fitted on quantiles (no pixel correspondence needed), blur on mean gradient energy
and noise on a Laplacian noise estimate.  Existing color_gain / appearance settings are ignored while fitting."""
import argparse, copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_view as SV
SV._setup_gl("egl")
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, convolve
import build_scene as B

EPS = [("circular_hole_v21_clean", 0), ("circular_hole_v21_clean", 5), ("square_hole_v21_clean", 60), ("square_hole_v21_clean", 13)]
CROP = {"top": (20, 440, 120, 520), "wrist": (0, 480, 0, 640)}


def grad_energy(g):
    gx, gy = np.gradient(gaussian_filter(g, 0.7)); return float(np.mean(np.hypot(gx, gy)))


def noise_sigma(g):            # Immerkaer fast noise variance estimate
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    return float(np.sqrt(np.pi / 2) / 6 * np.mean(np.abs(convolve(g, k))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", type=int, default=6); ap.add_argument("--blur-top", type=float, default=None); ap.add_argument("--blur-wrist", type=float, default=None); a = ap.parse_args()
    cfg0 = B.load_config(); W, H = 640, 480
    for cam in ("top", "wrist"):
        cfg0["cameras"][cam].pop("color_gain", None); cfg0["cameras"][cam].pop("appearance", None)
    sims = {"top": [], "wrist": []}; reals = {"top": [], "wrist": []}
    for ds, ep in EPS:
        n = len(pd.read_parquet(B.DATASETS / ds / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"))
        for fr in np.linspace(0.08, 0.85, a.frames).astype(float) * n:
            cfg = copy.deepcopy(cfg0); cfg["robot"]["source"] = "dataset"; cfg["robot"]["dataset"] = dict(name=ds, episode=ep, frame=int(fr))
            cfg["render"]["settle_steps"] = 300
            for k in ("circle", "square"): cfg["holes"][k]["enabled"] = (k == ("square" if "square" in ds else "circle"))
            sc = B.Scene(cfg); im = sc.render(W, H); sc.close(); rl = B.real_frames(cfg, W, H)
            for cam, r in (("top", rl[0]), ("wrist", rl[1])):
                sims[cam].append(im[cam]); reals[cam].append(r)
        print(ds, ep, "done", flush=True)
    q = np.linspace(0, 100, 33)
    out_cfg = {}
    for cam in ("top", "wrist"):
        y0, y1, x0, x1 = CROP[cam]
        S = np.stack([s[y0:y1, x0:x1] for s in sims[cam]]).astype(np.float32); R = np.stack([r[y0:y1, x0:x1] for r in reals[cam]]).astype(np.float32)
        gs = lambda X: X.mean(-1)
        def fit_lut(Sx):
            xs = np.percentile(Sx.reshape(-1, 3), q, axis=0)            # (33,3) sim quantiles
            ys = np.percentile(R.reshape(-1, 3), q, axis=0)
            lx = np.arange(0, 256, 8.0); ly = []
            for c in range(3):
                xc, uc = np.unique(xs[:, c], return_index=True); ly.append(np.interp(lx, xc, ys[uc, c]))
            return lx, ly
        gr, nr = np.mean([grad_energy(g) for g in gs(R)]), np.mean([noise_sigma(g) for g in gs(R)])
        best = None
        fixed = {"top": a.blur_top, "wrist": a.blur_wrist}[cam]
        for sigma in ([fixed] if fixed is not None else np.arange(0.0, 4.01, 0.25)):
            Sb = np.stack([gaussian_filter(x, (sigma, sigma, 0)) if sigma > 0 else x for x in S])
            lx, ly = fit_lut(Sb)
            Sm = np.stack([np.stack([np.interp(x[..., c], lx, ly[c]) for c in range(3)], -1) for x in Sb])
            gsim = np.mean([grad_energy(g) for g in gs(Sm)]); err = abs(gsim - gr)
            if best is None or err < best[0]: best = (err, sigma, lx, ly, Sm, gsim)
        err, sigma, lx, ly, Sm, gsim = best
        ns = np.mean([noise_sigma(g) for g in gs(Sm)]); add = float(np.sqrt(max(0.0, nr ** 2 - ns ** 2)))
        print(f"{cam}: blur sigma {sigma:.2f} px (gradient energy sim {gsim:.2f} vs real {gr:.2f}), noise real {nr:.2f} sim {ns:.2f} -> add {add:.2f}; "
              f"mean rgb sim {Sm.mean((0,1,2)).round(0)} real {R.mean((0,1,2)).round(0)}")
        (Path(__file__).resolve().parent / "appearance").mkdir(exist_ok=True)
        (Path(__file__).resolve().parent / "appearance" / f"{cam}.json").write_text(json.dumps(
            dict(blur_sigma=float(sigma), noise_std=add, lut_x=lx.tolist(), lut_y=[list(map(float, l)) for l in ly],
                 n_pairs=len(S), note="fitted by calibrate_appearance.py on dataset frames"), indent=0))
main()
