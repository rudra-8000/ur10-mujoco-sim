#!/usr/bin/env python3
"""Render the standalone PincOpen gripper model from canonical views at open / grasp / nearly-closed, to compare with the CAD.
Output: ur10sim/out/gripper_views.png  (rows: frac 0, 0.55, 0.9;  columns: linkage plane [looking along the hinge axis], front [along the
fingers], top [along the jaw direction]).  Usage: python ur10sim/render_gripper.py"""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
sys.unraisablehook = lambda *a, **k: None
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, mujoco
import xml.etree.ElementTree as ET
import pincopen as PC
from PIL import Image

root = ET.parse(PC.MODEL_XML).getroot()
asset = root.find("asset")
for m in asset: m.set("file", str(PC.MODEL_XML.parent / "meshes" / m.get("file")))
root.find("compiler").attrib.pop("meshdir", None)
wb = root.find("worldbody")
ET.SubElement(wb, "light", pos="0.1 -0.6 0.4", dir="0 0.8 -0.5", diffuse="0.9 0.9 0.9")
ET.SubElement(wb, "light", pos="0.1 0.6 0.4", dir="0 -0.8 -0.5", diffuse="0.4 0.4 0.4")
vis = ET.Element("visual"); ET.SubElement(vis, "headlight", ambient="0.4 0.4 0.4", diffuse="0.5 0.5 0.5", specular="0 0 0"); root.insert(1, vis)
# cameras: (name, pos, target, up)
views = {"linkage plane (along hinge axis)": ([0.05, -0.55, 0.0], [0.05, 0, 0.0], [0, 0, 1]),
         "front (along fingers)": ([0.75, 0.0, 0.0], [0.0, 0.0, 0.0], [0, 1, 0]),
         "top (along jaw axis)": ([0.05, 0.0, 0.6], [0.05, 0.0, 0.0], [1, 0, 0])}
for n, (p, t, up) in views.items():
    z = np.array(p) - np.array(t); z /= np.linalg.norm(z); x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    ET.SubElement(wb, "camera", name=n.split()[0], pos=" ".join(map(str, p)), xyaxes=" ".join(map(str, [*x, *y])), fovy="30")
m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")); d = mujoco.MjData(m); m.opt.gravity[:] = 0
r = mujoco.Renderer(m, 360, 480)
rows = []
for frac in (0.0, 0.55, 0.9):
    mujoco.mj_resetData(m, d); tgt = float(np.clip(PC.cam_angle_for_frac(frac), m.actuator_ctrlrange[0][0], m.actuator_ctrlrange[0][1]))
    for k in range(4000):
        d.ctrl[0] = tgt * min(1.0, k / 2500); mujoco.mj_step(m, d)
    ims = []
    for n in views:
        r.update_scene(d, camera=n.split()[0]); ims.append(r.render().copy())
    rows.append(np.concatenate(ims, 1))
out = HERE / "out"; out.mkdir(exist_ok=True)
Image.fromarray(np.concatenate(rows, 0)).save(out / "gripper_views.png"); print("wrote", out / "gripper_views.png")
