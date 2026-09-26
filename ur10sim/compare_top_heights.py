#!/usr/bin/env python3
"""Test the two top-camera height hypotheses (0.78 m vs 1.07 m above the table): fit the pose for each (fit_top_camera.py), render the sim
top view, and write out/top_height_test.png = [real | 0.78 m blend | 1.07 m blend] plus a board-edge overlap number.
Usage: python ur10sim/compare_top_heights.py [--fov 65]"""
import argparse, copy, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
sys.unraisablehook = lambda *a, **k: None
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from PIL import Image, ImageDraw
import build_scene as B
import fit_top_camera as F

ap = argparse.ArgumentParser(); ap.add_argument("--fov", type=float, default=65.0); a = ap.parse_args()
F.FOV_H = a.fov; F.F = 320.0 / np.tan(np.radians(a.fov) / 2)
cfg = B.load_config(); sc = B.Scene(cfg)
real = B.real_frames(cfg, 640, 480)[0]
def edge_mask(img):      # bright wood (board) vs darker background/robot-free border: crude board mask from brightness + colour
    f = img.astype(float); return (f[..., 0] > f[..., 2] + 12) & (f.mean(-1) > 90)
rm = edge_mask(real)
panels = [np.asarray(ImageDraw.Draw(Image.fromarray(real.copy())) and Image.fromarray(real))]
rows = []
for h in (0.78, 1.07):
    r = F.fit(h); x, y, th, yaw, roll = r.x
    pos = np.array([x, y, h + F.TZ]); R = F.cam_R(th, yaw, roll); fwd = -R[:, 2]
    c = copy.deepcopy(cfg); t = c["cameras"]["top"]
    t["pos"] = [float(v) for v in pos]; t["look_at"] = [float(v) for v in pos + fwd * (F.TZ - pos[2]) / fwd[2]]; t["up_hint"] = [float(v) for v in R[:, 1]]
    t["roll_deg"] = 0; t["fov_h_deg"] = a.fov
    sc.update_cameras(c); sim = sc.render(640, 480)["top"]
    sm = (sim[..., 0].astype(int) > sim[..., 2] + 40)
    iou = (sm & rm).sum() / max(1, (sm | rm).sum())
    blend = ((sim.astype(float) + real.astype(float)) / 2).astype(np.uint8)
    im = Image.fromarray(blend); ImageDraw.Draw(im).rectangle([0, 0, 470, 16], fill=(0, 0, 0))
    ImageDraw.Draw(im).text((4, 2), f"h={h:.2f} m fit cost {r.cost:.1f}  pos ({x:.2f},{y:.2f}) tilt {np.degrees(th):.0f} deg  board IoU {iou:.2f}", fill=(255, 255, 255))
    panels.append(np.asarray(im)); rows.append((h, r.cost, iou))
    print(f"h={h:.2f}: fit cost {r.cost:.1f}, camera ({x:.3f},{y:.3f},{h+F.TZ:.3f}), tilt {np.degrees(th):.1f} deg, board-mask IoU vs real {iou:.3f}")
Image.fromarray(np.concatenate(panels, 1)).save(HERE / "out" / "top_height_test.png"); print("wrote", HERE / "out" / "top_height_test.png")
