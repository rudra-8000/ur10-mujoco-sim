#!/usr/bin/env python3
"""Check the MuJoCo UR10 (official Universal Robots description, assets/ur10) + gripper mount against ur10_kinematics.py (the FK used for the datasets): flange pose for random joints,
and the TCP (145 mm point at the pad centre) against forward_kinematics() with the TCP offset."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import build_scene as B
import ur10_kinematics as K

cfg = B.load_config(); cfg["render"]["settle_steps"] = 1
sc = B.Scene(cfg); m, d = sc.model, sc.data
rng = np.random.default_rng(0)
site = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
worst_f = worst_t = 0.0
for _ in range(200):
    q = rng.uniform(-np.pi, np.pi, 6)
    d.qpos[sc.arm_adr] = q; mujoco.mj_kinematics(m, d)
    T = np.eye(4)
    for i in range(6): T = T @ K._dh_transform(K._A[i], K._D[i], K._ALPHA[i], q[i])
    fl = d.site_xpos[site("flange")]; Rf = d.site_xmat[site("flange")].reshape(3, 3)
    worst_f = max(worst_f, np.abs(fl - T[:3, 3]).max(), np.abs(Rf - T[:3, :3]).max())
    pos, _ = K.forward_kinematics(q)           # includes the 145 mm TCP offset along flange +z
    worst_t = max(worst_t, np.abs(d.site_xpos[site("tcp")] - pos).max())
print(f"flange pose vs ur10_kinematics: max abs err {worst_f:.2e}; TCP site (145 mm along the flange axis) vs FK with 145 mm offset: max err {worst_t*1e3:.3f} mm")
print("PASS" if worst_f < 1e-9 and worst_t < 1e-6 else "CHECK")
sys.exit(0 if worst_f < 1e-9 and worst_t < 1e-6 else 1)
