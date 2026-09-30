"""Regression check: the PincOpen linkage must not flip onto its other assembly branch (R finger tip ~180deg off).
Cycles the gripper fully closed/open with raw step commands; open must return to the SAME pose every time, closed must be L/R symmetric.
Run: python check_gripper_branch.py   (exits non-zero on failure)"""
import numpy as np, mujoco as mj
import pincopen as PC

m = mj.MjModel.from_xml_path(str(PC.MODEL_XML)); d = mj.MjData(m)
q = lambda n: d.qpos[m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)]]
lo, hi = m.actuator_ctrlrange[0]
d.qpos[m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "cam_joint")]] = hi
opens = []
for _ in range(5):
    for tgt in (lo, hi):                       # command far beyond reach on purpose: the cam limit must be what stops it
        d.ctrl[0] = tgt
        for _ in range(700): mj.mj_step(m, d)
    opens.append((q("tip_R_joint"), q("tip_L_joint")))
    assert abs(opens[-1][0] - opens[0][0]) < 0.02 and abs(opens[-1][1] - opens[0][1]) < 0.02, f"open pose drifted (branch flip?): {opens}"
d.ctrl[0] = lo
for _ in range(700): mj.mj_step(m, d)
assert abs(q("tip_R_joint") + q("tip_L_joint")) < 0.3, f"closed pose not L/R symmetric: R={q('tip_R_joint'):.2f} L={q('tip_L_joint'):.2f}"
print("ok: no branch flip; open tipR/L =", np.round(opens[-1], 2), "closed tipR/L =", round(q("tip_R_joint"), 2), round(q("tip_L_joint"), 2))
