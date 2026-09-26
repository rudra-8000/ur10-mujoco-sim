#!/usr/bin/env python3
"""Minimal use of the scene as a library: move the (kinematic) arm between two joint configurations, close the gripper, save the cameras.

  python ur10sim/examples/drive_joints.py            -> out/drive_joints.png (top over wrist camera at 4 moments)

The arm is kinematic: every physics step we write the joint positions AND velocities (so contacts see the motion); the gripper is a real
position-controlled MuJoCo actuator (ctrl = cam angle from the recorded gripper fraction 0 = open .. ~0.57 = holding).
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import scene_view
scene_view._setup_gl("egl")              # or "osmesa" / "glfw" if EGL is not available
import numpy as np
import mujoco
from PIL import Image
import build_scene as B
import pincopen as PC

cfg = B.load_config(sys.argv[1] if len(sys.argv) > 1 else None)
sc = B.Scene(cfg)                        # builds the MJCF, settles the physics
m, d = sc.model, sc.data
q0, q1 = np.radians([0, -90, 90, -90, -90, 90]), np.radians([30, -100, 100, -80, -90, 60])
strip = []
N = 200                                                        # physics steps (1 ms each) per keyframe
ts = np.linspace(0, 1, 9)
for t0, t1 in zip(ts[:-1], ts[1:]):
    for k in range(N):
        t = t0 + (t1 - t0) * (k + 1) / N                       # smooth interpolation: never jump the kinematic arm between steps
        q = q0 + (q1 - q0) * t
        d.qvel[sc.arm_dof] = (q - d.qpos[sc.arm_adr]) / m.opt.timestep
        d.qpos[sc.arm_adr] = q
        d.ctrl[0] = float(np.clip(PC.cam_angle_for_frac(0.55 * t), *sc.ctrl_range))     # gripper closes with t
        mujoco.mj_step(m, d)
    im = sc.render(cfg["render"]["width"], cfg["render"]["height"])
    strip.append(np.concatenate([im["top"], im["wrist"]], axis=0))
out = HERE / cfg["render"]["out_dir"]; out.mkdir(exist_ok=True)
Image.fromarray(np.concatenate(strip[1::2], axis=1)).resize((1600, 480)).save(out / "drive_joints.png")
print("wrote", out / "drive_joints.png")
