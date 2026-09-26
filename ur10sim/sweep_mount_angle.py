#!/usr/bin/env python3
"""Montage of the sim wrist image for several mount angles (cameras.wrist.mount.angle_deg) next to the real frame.
  python ur10sim/sweep_mount_angle.py [-10 0 10 20 30]      -> out/mount_angle_sweep.png"""
import copy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_view as SV
SV._setup_gl("egl")
import numpy as np
from PIL import Image
import build_scene as B

def main():
    angles = [float(a) for a in sys.argv[1:]] or [-10, 0, 10, 20, 30]
    cfg0 = B.load_config(); W, H = cfg0["render"]["width"], cfg0["render"]["height"]
    tiles = []
    for a in angles:
        cfg = copy.deepcopy(cfg0); cfg["cameras"]["wrist"]["mount"]["angle_deg"] = a
        sc = B.Scene(cfg); F = B.wrist_mount_frames(cfg["cameras"]["wrist"])
        ax = F["meta"]["cad_lens_axis_deg_from_x_g"] + a
        tiles.append(SV._label(sc.render(W, H)["wrist"], f"mount angle {a:+.0f} deg (lens axis {ax:.1f} deg off x_g)")); sc.close()
    real = B.real_frames(cfg0, W, H)
    if real is not None: tiles.append(SV._label(real[1], "REAL"))
    rows = [np.concatenate(tiles[i:i + 3] + [np.zeros_like(tiles[0])] * (3 - len(tiles[i:i + 3])), axis=1) for i in range(0, len(tiles), 3)]
    out = np.concatenate(rows, axis=0)
    p = B.REPO / cfg0["render"]["out_dir"] / "mount_angle_sweep.png"; Image.fromarray(out).save(p); print("wrote", p)
main()
