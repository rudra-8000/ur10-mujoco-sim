#!/usr/bin/env python3
"""Drop-insertion checks for assets/peg_hole (CPU): right end into each hole (with lateral offsets) must seat,
wrong end must stay on the rim."""
import sys
from pathlib import Path
import numpy as np
import mujoco

A = Path(__file__).resolve().parent / "assets" / "peg_hole"
m = mujoco.MjModel.from_xml_path(str(A / "peg_hole_test.xml")); d = mujoco.MjData(m)
pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "peg")
Q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "peg_free")]   # peg qpos address (holes may be free too)
sq_end_dn = np.array([1, 0, 0, 0.0])                    # square end down (peg +z up)
circ_end_dn = np.array([0, 1, 0, 0.0])                  # 180 deg about x: circular end down
LOW = {"sq": 0.10325, "circ": 0.09675}                  # distance from centroid to each end face (m)
cases = []
for hole, cx in (("square", -0.15), ("circle", 0.15)):
    for end, quat in (("sq", sq_end_dn), ("circ", circ_end_dn)):
        right = (hole == "square" and end == "sq") or (hole == "circle" and end == "circ")
        for off in ((0, 0), (0.0015, 0), (0.0015, -0.0015), (-0.002, 0.0)) if right else ((0, 0), (0.001, 0.001)):
            cases.append((hole, end, right, cx, off, quat))
fails = 0
for hole, end, right, cx, off, quat in cases:
    mujoco.mj_resetData(m, d)
    z0 = 0.041 + 0.030 + LOW[end]
    d.qpos[Q:Q + 7] = [cx + off[0], off[1], z0, *quat]
    for _ in range(2500): mujoco.mj_step(m, d)
    z = d.qpos[Q + 2]; near = np.hypot(d.qpos[Q] - cx, d.qpos[Q + 1]) < 0.01
    seated = near and z < LOW[end] + 0.006          # over the hole AND bottom face within 6 mm of the floor
    ok = seated == right
    fails += 0 if ok else 1
    print(f"{end:>4}-end down over {hole:>6} hole, offset {off[0]*1e3:+.1f},{off[1]*1e3:+.1f} mm -> bottom height above floor "
          f"{(z - LOW[end])*1e3:6.1f} mm  {'SEATED' if seated else ('on rim' if near and z > 0.1 else 'toppled off the rim')}  expected {'seat' if right else 'reject'}  {'ok' if ok else 'MISMATCH'}")
print("PASS" if fails == 0 else f"{fails} MISMATCH(ES)")
sys.exit(1 if fails else 0)
