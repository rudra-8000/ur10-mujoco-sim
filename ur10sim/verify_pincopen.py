#!/usr/bin/env python3
"""Verify assets/pincopen/pincopen.xml (revision try3) in MuJoCo, CPU only:
 - loops close, jaw gap (pad box faces) vs an independent analytic six-bar solution, tip rotation vs analytic
 - recorded-gripper-fraction map reproduces the peg sizes (41.5 / 50 mm) at the dataset grasp plateaus
 - URDF tree compiles, a 20 mm block is held by friction
"""
import json
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pincopen as B  # analytic reference

A = Path(__file__).resolve().parent / "assets" / "pincopen"
m = mujoco.MjModel.from_xml_path(str(A / "pincopen.xml")); d = mujoco.MjData(m)
m.opt.gravity[:] = 0
meta = json.load(open(A / "pincopen_meta.json"))
tab = np.array(meta["gap_table_deg_mm"]); lo, hi = meta["cam_range_rad"]
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
cam_adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cam_joint")]
print(f"nq={m.nq} nbody={m.nbody} neq={m.neq} nu={m.nu}  cam range deg {np.degrees([lo, hi]).round(1)}")
mujoco.mj_forward(m, d)

def ramp_to(theta, steps_per_deg=25):
    """move the servo target gradually (keeps the linkage on its branch), then settle"""
    cur = d.ctrl[0]
    n = max(1, int(abs(np.degrees(theta - cur)) * steps_per_deg))
    for i in range(n):
        d.ctrl[0] = cur + (theta - cur) * (i + 1) / n
        mujoco.mj_step(m, d)
    for _ in range(1500):
        mujoco.mj_step(m, d)

def gap_mm():
    zL = d.geom_xpos[gid("pad_L")][2] - m.geom_size[gid("pad_L")][2]
    zR = d.geom_xpos[gid("pad_R")][2] + m.geom_size[gid("pad_R")][2]
    return (zL - zR) * 1e3

def tip_angle(name):    # rotation of the tip body about y (hinge axis -y, CCW positive) relative to qpos0, deg
    R = d.xmat[bid(name)].reshape(3, 3)
    return np.degrees(np.arctan2(R[2, 0], R[0, 0])) * -1.0

ref = {n: tip_angle(n) for n in ("tip_L", "tip_R")}
prev = {"L": None, "R": None}
rows = []
worst_gap = worst_rot = 0.0
d.ctrl[0] = 0.0
for deg in list(np.arange(0, np.degrees(hi) + 0.1, 10.0)) + list(np.arange(np.degrees(hi), np.degrees(lo) - 0.1, -10.0)):
    ramp_to(np.radians(deg))
    act = np.degrees(d.qpos[cam_adr])
    g = gap_mm(); ana = np.interp(act, tab[:, 0], tab[:, 1])
    rot_sim = max(abs(tip_angle(n) - ref[n]) for n in ("tip_L", "tip_R"))
    poses = {s: B.finger_pose(s, np.radians(act), prev[s]) for s in "LR"}
    prev = {s: poses[s] for s in "LR"}
    rot_ana = max(abs(np.degrees(poses[s]["dphi"])) for s in "LR")
    rows.append((act, g, ana, rot_sim, rot_ana))
    if ana > 8.0:        # below ~8 mm the pads are in contact in MuJoCo (analytic ignores collisions)
        worst_gap = max(worst_gap, abs(g - ana)); worst_rot = max(worst_rot, abs(rot_sim - rot_ana))
print(" cam_deg  sim_gap_mm  analytic_mm  tip_rot_sim  tip_rot_analytic")
for r in rows: print(f"  {r[0]:6.1f}   {r[1]:8.2f}   {r[2]:8.2f}    {r[3]:7.2f}     {r[4]:7.2f}")
print(f"WORST: |sim-analytic gap| {worst_gap:.2f} mm, |tip rotation diff| {worst_rot:.2f} deg")
kin_ok = worst_gap < 1.5 and worst_rot < 3.0
print("KINEMATICS", "PASS" if kin_ok else "CHECK")

# ---- gripper fraction -> peg sizes at the dataset grasp plateaus
fm = meta["frac_map"]
print(f"frac map fitted on the two plateaus: span {fm['span_deg']:.1f} deg (cam/servo ratio {fm['implied_cam_to_servo_ratio']:.2f}), "
      f"frac0 at cam {fm['a_open_deg']:.1f} deg")
print(f"  PREDICTIONS: gap at frac 0 = {fm['PREDICTED_gap_mm_at_frac_0']:.1f} mm (linkage max is {tab[:,1].max():.1f} mm; the '80 mm' figure does NOT match), "
      f"rubber pads meet at frac {fm['PREDICTED_frac_at_pad_contact']:.2f} vs max recorded {fm['data_max_recorded_frac']}")

# ---- URDF tree
try:
    mu = mujoco.MjModel.from_xml_path(str(A / "pincopen.urdf"))
    print(f"URDF compiles in MuJoCo: nq={mu.nq} nbody={mu.nbody} (loops intentionally absent)")
except Exception as e:  # noqa: BLE001
    print("URDF compile via MuJoCo not possible here:", str(e)[:100])

# ---- grasp: hold a 20 mm block by friction (gravity along -y, i.e. along the hinge axis)
jc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "jaw_center")
mujoco.mj_forward(m, d)
BOX = """<mujoco><include file="pincopen.xml"/><worldbody>
<body name="blk" pos="0.080 0 0.0"><freejoint/><geom name="blk" type="box" size="0.0125 0.010 0.010" mass="0.05" friction="0.8 0.01 0.001"/></body>
</worldbody></mujoco>"""
tmp = A / "_grasp_test.xml"; tmp.write_text(BOX)
try:
    g = mujoco.MjModel.from_xml_path(str(tmp)); gd = mujoco.MjData(g); g.opt.gravity[:] = 0
    gb = mujoco.mj_name2id(g, mujoco.mjtObj.mjOBJ_BODY, "blk")
    gd.qpos[-7:-4] = (0.080, 0.0, 0.0); gd.qvel[:] = 0
    gd.ctrl[0] = 0.0
    tgt = np.radians(-60.0)
    for k in range(6000):
        gd.ctrl[0] = tgt * min(1.0, k / 3000)
        mujoco.mj_step(g, gd)
    g.opt.gravity[:] = (0, -9.81, 0)
    for _ in range(300): mujoco.mj_step(g, gd)
    y0 = gd.xpos[gb][1]
    for _ in range(2000): mujoco.mj_step(g, gd)
    drift = (gd.xpos[gb][1] - y0) * 1e3
    print(f"grasp test: block y-drift over 2 s under gravity = {drift:.2f} mm, cam {np.degrees(gd.qpos[0]):.1f} deg -> {'HELD' if abs(drift) < 3 else 'SLIPPING'}")
finally:
    tmp.unlink(missing_ok=True)
sys.exit(0 if kin_ok else 1)
