#!/usr/bin/env python3
"""Sweep the wrist-camera tilt (and optionally the look-at point) and report, per setting, the fraction of image pixels covered by
the gripper's Tip_Support (rubber pads), Removable_Tip, other gripper parts, the peg -- to find a view that sees the middle of the tips
with only the Tip_Supports + object in frame and the Removable_Tip bases barely visible.
Usage: python ur10sim/tune_wrist_camera.py [--tilts -20 -10 0 10 20] [--config scene_config.yaml]"""
import argparse, copy, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
sys.unraisablehook = lambda *a, **k: None
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, mujoco
import build_scene as B

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None); ap.add_argument("--tilts", type=float, nargs="+", default=[-30, -20, -10, 0, 10, 20, 30])
ap.add_argument("--look_x", type=float, nargs="+", default=None, help="also sweep the look-at x (m along the fingers) instead of tilt")
a = ap.parse_args()
cfg = B.load_config(a.config)
sc = B.Scene(cfg); m = sc.model
r = mujoco.Renderer(m, 480, 640)
mesh_of = {}
for g in range(m.ngeom):
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_of[g] = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) or ""
bodyname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
def classify(g):
    n = mesh_of.get(g, "")
    if n.startswith("Tip_Support"): return "tip_support"
    if n.startswith("Removable_Tip"): return "removable_tip"
    if n.startswith("Distal"): return "distal"
    if bodyname(g) == "peg": return "peg"
    if bodyname(g).startswith(("pincopen", "cam", "dr_", "ir_", "or_", "tip_")): return "gripper_other"
    if g == 0 or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and bodyname(g) == "world": return "table"
    return "other"
cls = {g: classify(g) for g in range(m.ngeom)}
PAL = {"tip_support": (1, 0, 0), "removable_tip": (0, 1, 0), "distal": (0, 0, 1), "gripper_other": (1, 1, 0), "peg": (1, 0, 1), "table": (0.5, 0.5, 0.5), "other": (0.2, 0.2, 0.2)}
orig_rgba = m.geom_rgba.copy(); orig_mat = m.geom_matid.copy()
for g in range(m.ngeom):
    m.geom_rgba[g] = (*PAL[cls[g]], 1.0); m.geom_matid[g] = -1
# classify by hue (lighting changes brightness, not hue)
print("tilt/look   " + "  ".join(f"{k:>13}" for k in ("tip_support", "removable_tip", "distal", "gripper_other", "peg")) + "   rest")
sweep = [("tilt", t) for t in a.tilts] if a.look_x is None else [("look_x", x) for x in a.look_x]
for kind, val in sweep:
    c = copy.deepcopy(cfg)
    if kind == "tilt": c["cameras"]["wrist"]["tilt_deg"] = val
    else: c["cameras"]["wrist"]["look_at"][0] = val
    sc.update_cameras(c)
    r.update_scene(sc.data, camera="wrist")
    raw = r.render()
    if kind == "tilt" and val == a.tilts[len(a.tilts) // 2]:
        from PIL import Image; Image.fromarray(raw).save(Path(__file__).resolve().parent / "out" / "wrist_classes.png")
    img = raw.astype(float) / 255.0; tot = img.shape[0] * img.shape[1]
    mx = img.max(-1, keepdims=True); n = img / np.maximum(mx, 1e-6); R, G, Bc = n[..., 0], n[..., 1], n[..., 2]
    hi = mx[..., 0] > 0.08
    fr = {"tip_support": (hi & (R > .95) & (G < .4) & (Bc < .4)).sum() / tot,
          "removable_tip": (hi & (G > .95) & (R < .4) & (Bc < .4)).sum() / tot,
          "distal": (hi & (Bc > .95) & (R < .4) & (G < .4)).sum() / tot,
          "gripper_other": (hi & (R > .8) & (G > .8) & (Bc < .4)).sum() / tot,
          "peg": (hi & (R > .8) & (Bc > .8) & (G < .4)).sum() / tot}
    rest = 1 - sum(fr[k] for k in ("tip_support", "removable_tip", "distal", "gripper_other", "peg"))
    print(f"{kind}={val:6.1f}  " + "  ".join(f"{100 * fr[k]:12.1f}%" for k in ("tip_support", "removable_tip", "distal", "gripper_other", "peg")) + f"  {100 * rest:5.1f}%")
