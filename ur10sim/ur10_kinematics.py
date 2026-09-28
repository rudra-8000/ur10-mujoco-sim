#!/usr/bin/env python
"""UR10 forward kinematics + gripper angle->width conversion.

Used offline by convert_joints_to_ee.py to turn recorded joint-angle +
normalized-gripper trajectories into end-effector pose + physical gripper
width trajectories, for training a Cartesian-action pi0.5 policy instead of
a joint-angle one.

FORWARD KINEMATICS
------------------
Computed from the standard, publicly-published UR10 (CB-series, non-e-series)
Denavit-Hartenberg parameters (the same parameters underlying most open-source
UR kinematics implementations, e.g. ROS-Industrial's ur_description and the
ur_kinematics package). Convention: standard DH, T_i = Rz(theta_i) Tz(d_i)
Tx(a_i) Rx(alpha_i).

This is NOT the same as querying the real controller's calibrated kinematics
(`rtde_receive.getActualTCPPose()`, used live elsewhere in this project, e.g.
examples/ur10_gello/serl_client_ur10.py) — that call corrects for this
specific robot's small manufacturing deviations via its calibration.conf and
is only available from a connected controller. Nominal DH parameters are
typically accurate to within a few mm of the real robot, which is more than
enough for behavior-cloning training (the vision-conditioned policy only
needs internal self-consistency between camera frames and pose, not absolute
world-frame accuracy) — but if sub-mm real-world accuracy is ever needed
(e.g. cross-checking against a physical measurement), pull this robot's
calibration.conf from the controller and adjust the parameters below, or
compute FK live via RTDE/URSim instead of this module.

Sanity-check before trusting this at scale: run
    python ur10_agent/conversion/ur10_kinematics.py --selftest
and compare the printed home-pose position against a measurement (or a
URSim/live-robot getActualTCPPose() reading) taken at the same joint pose
["0, -90, 90, -90, -90, 90" deg, the documented UR10 home in
examples/ur10_gello/ur10_teleoperate.py's default_ur_home_action()].

GRIPPER ANGLE -> WIDTH
-----------------------
The "PincOpen" gripper (lerobot's pincopen_gripper.py) only exposes a
Dynamixel servo angle (285 deg = fully open, 166 deg = fully closed).
GRIPPER_OPEN_MM / GRIPPER_CLOSED_MM below are the measured physical jaw gap
at those two angles (user-measured 2026-09-14: ~80mm open, 0mm closed —
not from a datasheet). If these are ever re-measured, update the two
constants below; every place gripper width is used derives from them.

TCP OFFSET
----------
forward_kinematics() returns the wrist FLANGE pose by default (end of the
standard 6-DOF DH chain), not the gripper's actual tool-center-point (TCP,
i.e. roughly where the fingertips meet). The real UR controller's
getActualTCPPose() reports whatever TCP is configured in the robot's
installation settings (a translation, and possibly rotation, from the
flange) — neither this repo nor lerobot ever calls setTcp()/set_tcp() in
code, so that configuration (if any) only lives on the physical controller.
TCP_OFFSET_MM below (145mm, user-recalled 2026-09-14, unverified against
the teach pendant's Installation -> TCP Configuration) is applied as a
translation along the flange's local +Z axis (the standard "tool extends
straight out along Z" mounting convention) — this is a real, if-unverified,
number, so it is applied by default here.

APPLY_TCP_OFFSET is a toggle: flip it to False (or set TCP_OFFSET_MM = 0)
to fall back to raw flange pose, e.g. if 145mm/this axis turns out to be
wrong, or if a live serving path ends up commanding the flange frame
directly (TCP unset / identity on the controller) rather than a
TCP-relative frame — in which case the dataset and the live command frame
must match, so exactly one of "bake the offset into training data" or
"apply it at the serving layer" should be active, never both.
"""

from __future__ import annotations

import os

# Force single-threaded BLAS BEFORE numpy is imported (threading libs read
# these at load time). Found 2026-09-15: running the per-row FK loop over a
# large dataset (combined_multihole_clean, 215k frames) intermittently
# segfaulted or corrupted array shapes (a clean "0-dimensional array"
# IndexError one run, raw SIGSEGV two others) on the exact same,
# independently-verified-clean source data -- i.e. non-deterministic
# behavior on deterministic input, the signature of a BLAS/threading race
# rather than a logic bug. Forcing single-threaded BLAS made the failure
# disappear across repeated retries. Root cause not fully confirmed (no
# root/kernel-log access on this machine to inspect further), but this is a
# real, reproducible fix -- keep it.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse

import numpy as np

# ── UR10 (CB-series) standard DH parameters (meters, radians) ──────────────
# a_i, d_i, alpha_i for i = 1..6. theta_i is the live joint angle (input).
_A = np.array([0.0, -0.612, -0.5723, 0.0, 0.0, 0.0])
_D = np.array([0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922])
_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

# ── Gripper calibration (measured — see module docstring) ──────────────────
GRIPPER_OPEN_ANGLE_DEG = 285.0   # matches pincopen_gripper.py open_angle
GRIPPER_CLOSED_ANGLE_DEG = 166.0  # matches pincopen_gripper.py close_angle
GRIPPER_OPEN_MM = 80.0    # measured jaw gap at open_angle (2026-09-14)
GRIPPER_CLOSED_MM = 0.0   # measured jaw gap at close_angle (2026-09-14)

# ── TCP offset (unverified, toggleable — see module docstring) ─────────────
APPLY_TCP_OFFSET = True
TCP_OFFSET_MM = 145.0  # translation along the flange's local +Z axis


def _dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _rotmat_to_rotvec(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle), UR/RTDE convention.

    Standard matrix-log / Rodrigues inversion. Handles the near-0 and
    near-pi degenerate cases explicitly for numerical stability.
    """
    cos_angle = np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-8:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # Near-pi: off-diagonal formula below divides by ~0; use the
        # symmetric part of (R - I)/2 to recover the axis instead.
        rot_plus_i = rot + np.eye(3)
        # Largest-diagonal column of (R+I) is numerically most stable.
        k = int(np.argmax(np.diag(rot_plus_i)))
        axis = rot_plus_i[:, k]
        axis = axis / np.linalg.norm(axis)
        return axis * angle
    axis = np.array([
        rot[2, 1] - rot[1, 2],
        rot[0, 2] - rot[2, 0],
        rot[1, 0] - rot[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def forward_kinematics(joint_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """UR10 FK: 6 joint angles (rad) -> (position_xyz_m[3], rotvec[3]).

    Position is in the robot base frame, meters, and includes TCP_OFFSET_MM
    (translated along the flange's local +Z axis) when APPLY_TCP_OFFSET is
    True — see module docstring; set APPLY_TCP_OFFSET=False for raw flange
    position instead. Orientation is a rotation vector (axis * angle,
    radians) in the same convention as UR's getActualTCPPose():
    [x, y, z, rx, ry, rz] — unaffected by the offset (translation only, no
    rotational offset assumed for a straight tool mount).
    """
    joint_rad = np.asarray(joint_rad, dtype=np.float64).reshape(6)
    t = np.eye(4)
    for i in range(6):
        t = t @ _dh_transform(_A[i], _D[i], _ALPHA[i], joint_rad[i])
    rot = t[:3, :3]
    pos = t[:3, 3].copy()
    if APPLY_TCP_OFFSET:
        pos = pos + rot @ np.array([0.0, 0.0, TCP_OFFSET_MM / 1000.0])
    rotvec = _rotmat_to_rotvec(rot)
    return pos, rotvec


def joints_gripper_to_ee(state_or_action: np.ndarray) -> np.ndarray:
    """[j0..j5, gripper_frac(0..1)] (7,) -> [x,y,z,rx,ry,rz,gripper_mm] (7,)."""
    vec = np.asarray(state_or_action, dtype=np.float64).reshape(7)
    pos, rotvec = forward_kinematics(vec[:6])
    gripper_mm = gripper_frac_to_mm(vec[6])
    return np.concatenate([pos, rotvec, [gripper_mm]]).astype(np.float32)


def gripper_frac_to_mm(frac: float) -> float:
    """Normalized [0,1] gripper command (0=open,1=closed) -> physical mm.

    Linear interpolation between the two PLACEHOLDER extremes above (see
    module docstring) — replace GRIPPER_OPEN_MM/GRIPPER_CLOSED_MM with real
    caliper measurements before trusting the mm scale for deployment.
    """
    frac = float(np.clip(frac, 0.0, 1.0))
    return GRIPPER_OPEN_MM - frac * (GRIPPER_OPEN_MM - GRIPPER_CLOSED_MM)


def gripper_mm_to_frac(mm: float) -> float:
    """Inverse of gripper_frac_to_mm — for converting a predicted mm target
    back into the [0,1] command the real gripper controller expects."""
    span = GRIPPER_OPEN_MM - GRIPPER_CLOSED_MM
    if abs(span) < 1e-9:
        return 0.0
    frac = (GRIPPER_OPEN_MM - float(mm)) / span
    return float(np.clip(frac, 0.0, 1.0))


# ── INVERSE KINEMATICS ───────────────────────────────────────────────────────
# Analytic (closed-form) UR-family 6R IK, for the live EE-pose serving path
# (examples/ur10/client_ur10.py, ur10_agent/scripts/run_client_small.py):
# the robot only ever accepts joint-space commands (servoJ, see CLAUDE.md
# §5 — no Cartesian/servoL command is used anywhere in this stack), so a
# policy trained on [x,y,z,rx,ry,rz,gripper_mm] needs this to produce
# something the robot can actually execute.
#
# This is the standard published UR5/UR10 closed-form algorithm (e.g.
# Hawkins' "Analytic Inverse Kinematics for the Universal Robots UR-5/UR-10
# Arms" tech report; the same structure underlies most open-source
# ur_kinematics ports) — NOT a generic numerical solver. It only applies
# because this module's DH parameters/convention (see forward_kinematics)
# match the family the algorithm assumes (spherical-wrist-less UR structure
# with d1/a2/a3/d4/d5/d6 offsets, alpha = [pi/2,0,0,pi/2,-pi/2,0]).
#
# Validated by numeric round-trip only (FK -> IK -> compare), since there is
# no live robot/URSim connection available from this workspace — see
# --ik-selftest. Before ever trusting this on the physical robot, additD
# a spot check: send a known joint pose, read getActualTCPPose(), confirm
# inverse_kinematics() recovers (approximately) the original joints.


def _rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: rotation vector (axis*angle) -> 3x3 rotation matrix.
    Inverse of _rotmat_to_rotvec."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rotvec / theta
    kx = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


def _dh_i(i: int, theta: float) -> np.ndarray:
    """DH transform for joint index i (0-indexed, i.e. A_{i+1} in 1-indexed
    algorithm notation: _dh_i(0, th1) == A1 == T01, etc.)."""
    return _dh_transform(_A[i], _D[i], _ALPHA[i], theta)


def _ik_branches_flange(t06: np.ndarray) -> list[np.ndarray]:
    """All (up to 8) analytic IK solutions for a FLANGE-frame target T06.

    Returns a list of joint_rad[6] arrays (may be fewer than 8 if some
    branches hit a domain error, e.g. an out-of-workspace target — those
    are skipped rather than returned as NaN/garbage).
    """
    d1, d4, d5, d6 = _D[0], _D[3], _D[4], _D[5]
    a2, a3 = _A[1], _A[2]
    del d1, d5  # not needed directly (folded into the DH transforms below)

    p06 = t06[:3, 3]
    r06 = t06[:3, :3]
    p05 = p06 - d6 * r06[:, 2]

    branches: list[np.ndarray] = []
    r_xy = float(np.hypot(p05[0], p05[1]))
    if r_xy < 1e-9:
        return branches  # directly above/below the base column — singular, skip
    psi = float(np.arctan2(p05[1], p05[0]))
    ratio = d4 / r_xy
    if abs(ratio) > 1.0:
        return branches  # target's wrist point is inside the un-reachable cylinder
    phi = float(np.arccos(np.clip(ratio, -1.0, 1.0)))

    for theta1 in (psi + phi + np.pi / 2.0, psi - phi + np.pi / 2.0):
        t01 = _dh_i(0, theta1)
        t16 = np.linalg.inv(t01) @ t06
        val = (t16[2, 3] - d4) / d6
        if abs(val) > 1.0:
            continue
        val = float(np.clip(val, -1.0, 1.0))
        t61 = np.linalg.inv(t16)
        for theta5 in (np.arccos(val), -np.arccos(val)):
            if abs(np.sin(theta5)) < 1e-8:
                continue  # wrist singularity (theta5≈0/pi): theta6 undefined, skip branch
            # NOTE: theta6 comes from T61 = inv(T16), not T16 itself -- easy to
            # get backwards (found + fixed via the round-trip self-test below).
            theta6 = float(np.arctan2(-t61[1, 2] / np.sin(theta5), t61[0, 2] / np.sin(theta5)))
            t45 = _dh_i(4, theta5)
            t56 = _dh_i(5, theta6)
            t14 = t16 @ np.linalg.inv(t45 @ t56)
            p13 = (t14 @ np.array([0.0, -d4, 0.0, 1.0]))[:3]
            r13 = float(np.linalg.norm(p13))
            cos3 = (r13 ** 2 - a2 ** 2 - a3 ** 2) / (2.0 * a2 * a3)
            if abs(cos3) > 1.0:
                continue  # elbow target out of reach for this branch
            cos3 = float(np.clip(cos3, -1.0, 1.0))
            for theta3 in (np.arccos(cos3), -np.arccos(cos3)):
                theta2 = -np.arctan2(p13[1], -p13[0]) + np.arcsin(
                    np.clip(a3 * np.sin(theta3) / max(r13, 1e-9), -1.0, 1.0)
                )
                t12 = _dh_i(1, theta2)
                t23 = _dh_i(2, theta3)
                t13 = t12 @ t23
                t34 = np.linalg.inv(t13) @ t14
                theta4 = float(np.arctan2(t34[1, 0], t34[0, 0]))
                branches.append(np.array([theta1, theta2, theta3, theta4, theta5, theta6]))
    return branches


def _angle_diff_norm(q: np.ndarray, q_ref: np.ndarray) -> float:
    """Sum of |shortest angular distance| per joint — for picking the branch
    closest to a reference/current joint config (continuity across control
    steps, avoiding a valid-but-wild alternate elbow/wrist flip)."""
    d = (q - q_ref + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(np.abs(d)))


def inverse_kinematics(
    pos_m: np.ndarray,
    rotvec: np.ndarray,
    q_ref: np.ndarray | None = None,
    apply_tcp_offset: bool | None = None,
    tcp_offset_mm: float | None = None,
) -> tuple[np.ndarray, dict]:
    """(position_xyz_m[3], rotvec[3]) -> joint_rad[6] + diagnostics.

    Mirrors forward_kinematics()'s TCP-offset convention: if
    apply_tcp_offset (default: module-level APPLY_TCP_OFFSET), the input
    pose is treated as the TCP/gripper-tip target (matching how the
    training datasets were built) and the offset is subtracted before
    solving IK for the flange — i.e. this function's default behavior
    exactly inverts joints_gripper_to_ee's position/orientation half.

    q_ref: current/previous joint_rad[6], used only to pick the branch
    closest to it out of the up to 8 analytic solutions (for continuity
    between control steps). If None, the first valid branch is returned.

    Returns (joint_rad[6], info) where info has:
      n_candidates   — how many of the 8 analytic branches were valid
      pos_err_m      — FK(chosen)-vs-target position error (meters)
      rotvec_err_rad — FK(chosen)-vs-target rotation error (radians)
    Raises ValueError if no branch is valid (target outside the reachable
    workspace) — callers on the real robot should catch this and NOT move
    rather than guess.
    """
    apply_offset = APPLY_TCP_OFFSET if apply_tcp_offset is None else apply_tcp_offset
    offset_mm = TCP_OFFSET_MM if tcp_offset_mm is None else tcp_offset_mm

    pos_m = np.asarray(pos_m, dtype=np.float64).reshape(3)
    r = _rotvec_to_rotmat(rotvec)
    flange_pos = pos_m - (r @ np.array([0.0, 0.0, offset_mm / 1000.0])) if apply_offset else pos_m

    t06 = np.eye(4)
    t06[:3, :3] = r
    t06[:3, 3] = flange_pos

    branches = _ik_branches_flange(t06)
    if not branches:
        raise ValueError(
            f"UR10 IK: no valid analytic solution for pos={pos_m} rotvec={rotvec} "
            f"(flange target={flange_pos}) — likely outside the reachable workspace"
        )

    if q_ref is not None:
        q_ref = np.asarray(q_ref, dtype=np.float64).reshape(6)
        branches.sort(key=lambda q: _angle_diff_norm(q, q_ref))
    chosen = branches[0]

    # Recompute flange FK for the chosen branch directly (NOT via
    # forward_kinematics(), which reads the module-level APPLY_TCP_OFFSET/
    # TCP_OFFSET_MM globals — this function must honor its own local
    # apply_offset/offset_mm arguments instead, so the two can't silently
    # diverge when a caller overrides them).
    t_chosen = np.eye(4)
    for i in range(6):
        t_chosen = t_chosen @ _dh_i(i, chosen[i])
    chosen_rot = t_chosen[:3, :3]
    chosen_flange_pos = t_chosen[:3, 3]
    chosen_tcp_pos = (
        chosen_flange_pos + chosen_rot @ np.array([0.0, 0.0, offset_mm / 1000.0])
        if apply_offset
        else chosen_flange_pos
    )
    pos_err = float(np.linalg.norm(chosen_tcp_pos - pos_m))
    rot_err_mat = r.T @ chosen_rot  # R_target^T @ R_chosen -> identity if exact match
    rotvec_err = float(np.linalg.norm(_rotmat_to_rotvec(rot_err_mat)))
    info = {
        "n_candidates": len(branches),
        "pos_err_m": pos_err,
        "rotvec_err_rad": rotvec_err,
    }
    return chosen, info


def _selftest() -> None:
    print("=== UR10 FK self-test ===")

    # 1. Orthonormality of the rotation matrix across random configs — this
    #    catches implementation bugs in the DH chain independent of any
    #    external ground truth.
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(200):
        q = rng.uniform(-np.pi, np.pi, size=6)
        t = np.eye(4)
        for i in range(6):
            t = t @ _dh_transform(_A[i], _D[i], _ALPHA[i], q[i])
        r = t[:3, :3]
        err = np.max(np.abs(r @ r.T - np.eye(3)))
        det_err = abs(np.linalg.det(r) - 1.0)
        max_err = max(max_err, err, det_err)
    print(f"[{'PASS' if max_err < 1e-8 else 'FAIL'}] rotation-matrix orthonormality/det "
          f"over 200 random configs, max err={max_err:.2e}")

    # 2. Round-trip rotvec -> known angle for a pure single-axis rotation.
    theta = 1.234
    r_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    rv = _rotmat_to_rotvec(r_z)
    err = abs(np.linalg.norm(rv) - theta)
    print(f"[{'PASS' if err < 1e-9 else 'FAIL'}] rotvec magnitude round-trip "
          f"(expected {theta:.6f}, got {np.linalg.norm(rv):.6f})")

    # 3. Home pose used throughout this project
    #    (examples/ur10_gello/ur10_teleoperate.py default_ur_home_action):
    #    joints (0, -90, 90, -90, -90, 90) degrees.
    home_deg = np.array([0.0, -90.0, 90.0, -90.0, -90.0, 90.0])
    home_rad = np.deg2rad(home_deg)
    pos, rotvec = forward_kinematics(home_rad)
    q = home_rad
    t = np.eye(4)
    for i in range(6):
        t = t @ _dh_transform(_A[i], _D[i], _ALPHA[i], q[i])
    flange_pos = t[:3, 3]
    print(f"[INFO] home pose (0,-90,90,-90,-90,90 deg) -> "
          f"pos={np.round(pos, 4)} m, rotvec={np.round(rotvec, 4)} rad "
          f"(APPLY_TCP_OFFSET={APPLY_TCP_OFFSET}, {TCP_OFFSET_MM}mm)")
    print(f"       raw flange pos (offset=0) = {np.round(flange_pos, 4)} m; "
          f"offset moved the point by {np.round(pos - flange_pos, 4)} m")
    print("       Sanity check: at this home pose the tool is flipped ~180deg "
          "about the base X axis, so a +Z-local offset should move the point "
          "DOWN in world Z (gripper hanging below the flange, pointing at a "
          "table) -- confirm the delta above has a negative Z component.")
    print("       Please spot-check this against a physical/URSim "
          "getActualTCPPose() reading at the same joint pose before "
          "trusting this at scale (see module docstring).")

    # 4. Gripper conversion sanity.
    for frac in (0.0, 0.5, 1.0):
        mm = gripper_frac_to_mm(frac)
        back = gripper_mm_to_frac(mm)
        print(f"[INFO] gripper frac={frac:.2f} -> {mm:.2f} mm -> back to frac={back:.2f} "
              f"(measured calibration, open={GRIPPER_OPEN_MM}mm closed={GRIPPER_CLOSED_MM}mm)")


def _ik_selftest() -> None:
    print("=== UR10 analytic IK round-trip self-test ===")
    print("(no live robot/URSim available in this workspace -- this is a numeric")
    print(" self-consistency check against this module's OWN forward_kinematics,")
    print(" not independent ground truth. Spot-check against a real")
    print(" getActualTCPPose() reading before trusting this on hardware.)")

    rng = np.random.default_rng(1)
    n = 500
    n_no_solution = 0
    n_wrong_branch = 0
    worst_pos_err = 0.0
    worst_rot_err = 0.0
    worst_joint_err = 0.0

    for trial in range(n):
        q_true = rng.uniform(-np.pi, np.pi, size=6)
        pos, rotvec = forward_kinematics(q_true)
        try:
            q_hat, info = inverse_kinematics(pos, rotvec, q_ref=q_true)
        except ValueError:
            n_no_solution += 1
            continue
        worst_pos_err = max(worst_pos_err, info["pos_err_m"])
        worst_rot_err = max(worst_rot_err, info["rotvec_err_rad"])
        joint_err = _angle_diff_norm(q_hat, q_true)
        worst_joint_err = max(worst_joint_err, joint_err)
        if joint_err > 1e-4:
            n_wrong_branch += 1
            if n_wrong_branch <= 3:
                print(f"  [trial {trial}] closest branch didn't match original joints: "
                      f"true={np.round(q_true, 4)} chosen={np.round(q_hat, 4)} "
                      f"(sum|diff|={joint_err:.6f} rad, but pos_err={info['pos_err_m']:.2e} m "
                      f"rot_err={info['rotvec_err_rad']:.2e} rad -- i.e. a DIFFERENT valid "
                      f"branch reaches the same pose; only a problem if q_ref-based "
                      f"branch selection is what's being tested)")

    print(f"[{'PASS' if n_no_solution == 0 else 'WARN'}] {n - n_no_solution}/{n} random configs "
          f"had >=1 valid analytic solution ({n_no_solution} reported no solution)")
    print(f"[{'PASS' if worst_pos_err < 1e-6 else 'FAIL'}] worst FK(IK(pose))-vs-pose position "
          f"error over all solved trials: {worst_pos_err:.2e} m")
    print(f"[{'PASS' if worst_rot_err < 1e-6 else 'FAIL'}] worst FK(IK(pose))-vs-pose rotation "
          f"error over all solved trials: {worst_rot_err:.2e} rad")
    print(f"[INFO] worst q_ref-branch joint-angle mismatch (sum |diff| over 6 joints): "
          f"{worst_joint_err:.2e} rad ({n_wrong_branch}/{n - n_no_solution} trials picked a "
          f"different-but-equally-valid branch than the original -- expected occasionally "
          f"near wrist/shoulder singularities, harmless as long as position/rotation error above is tiny)")

    # TCP-offset-aware round trip: same test, but through the full
    # joints_gripper_to_ee-equivalent (position+orientation only) with the
    # default TCP offset applied, mirroring the actual training/serving path.
    print("\n--- with TCP offset applied (matches training data convention) ---")
    worst_pos_err2 = 0.0
    for _ in range(200):
        q_true = rng.uniform(-np.pi, np.pi, size=6)
        pos, rotvec = forward_kinematics(q_true)  # module default: APPLY_TCP_OFFSET=True
        q_hat, info = inverse_kinematics(pos, rotvec, q_ref=q_true, apply_tcp_offset=True)
        worst_pos_err2 = max(worst_pos_err2, info["pos_err_m"])
    print(f"[{'PASS' if worst_pos_err2 < 1e-6 else 'FAIL'}] worst TCP-space position error: "
          f"{worst_pos_err2:.2e} m (200 trials)")

    # Reachability sanity: a target far outside the arm's reach should
    # raise ValueError, not silently return a bogus joint config.
    try:
        inverse_kinematics(np.array([100.0, 100.0, 100.0]), np.array([0.0, 0.0, 0.0]))
        print("[FAIL] an obviously unreachable target did not raise ValueError")
    except ValueError:
        print("[PASS] an obviously unreachable target correctly raised ValueError")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true", help="run internal FK consistency checks and print the home pose")
    p.add_argument("--ik-selftest", action="store_true", help="run IK round-trip numeric self-tests (500 random configs)")
    args = p.parse_args()
    if args.selftest:
        _selftest()
    if args.ik_selftest:
        _ik_selftest()
    if not args.selftest and not args.ik_selftest:
        p.print_help()
