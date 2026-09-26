"""Helpers for the PincOpen gripper model (see build_pincopen.py / assets/pincopen/pincopen_meta.json).

frac = the recorded gripper value (0 = open, 1 = closed).  cam angle is the MuJoCo `cam_joint` position (rad), measured
from the CAD export pose.  Jaw gap is the distance between the two pad faces in mm.
"""
import json
from pathlib import Path

import numpy as np

_META = json.loads((Path(__file__).resolve().parent / "assets" / "pincopen" / "pincopen_meta.json").read_text())
_FM = _META["frac_map"]
MODEL_XML = Path(__file__).resolve().parent / "assets" / "pincopen" / "pincopen.xml"


def cam_angle_for_frac(frac: float) -> float:
    """cam_joint target (rad) for a gripper fraction in [0, 1]."""
    return float(np.radians(np.interp(np.clip(frac, 0, 1), _FM["frac"], _FM["cam_deg"])))


def gap_mm_for_frac(frac: float) -> float:
    return float(np.interp(np.clip(frac, 0, 1), _FM["frac"], _FM["gap_mm"]))


def frac_for_gap_mm(gap_mm: float) -> float:
    """inverse of gap_mm_for_frac (gap decreases with frac)."""
    gaps, fr = np.array(_FM["gap_mm"])[::-1], np.array(_FM["frac"])[::-1]
    return float(np.interp(gap_mm, gaps, fr))


def flange_T_gripper():
    """4x4 transform of the gripper frame G expressed in the UR flange frame (see build_pincopen.mount_info; ASSUMED sign/offsets)."""
    m = _META["mount"]
    T = np.eye(4)
    T[:3, :3] = np.array(m["R_flange_gripper"])
    T[:3, 3] = m["p_flange_gripper_m"]
    return T
