#!/usr/bin/env python3
"""Fit the top-camera pose to pixel measurements from a real cam_high frame, for a fixed camera height above the table.

Measurements (frame 0/400 of circular_hole_v21_clean ep 0, 640x480) and known geometry:
  - hole block top-face centre  (0.007, -0.872, 0.042)  at px (331, 192)
  - far board edge (y = -(0.11 + 1.15) = -1.26, z = table)  at image row ~15
  - left board edge is a straight line parallel to y: px (15, 60) and (30, 180)
  - right board edge at row 180: px (632, 180); board width 1.25 m  => left/right edge points at row 180 are 1.25 m apart in x
  - hole block width 70 mm appears 34 px wide
Camera intrinsics: 640x480, fov_h 69 deg (D415 RGB nominal).  Unknowns per hypothesis: x, y, pitch, yaw, roll  (height fixed).
Usage: python ur10sim/fit_top_camera.py [--heights 0.78 1.07]
"""
import argparse, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scene as B

FOV_H = 69.0
F = 320.0 / np.tan(np.radians(FOV_H) / 2)
TZ = -0.002


def cam_R(theta, yaw, roll):
    """camera looking toward +y and down by `theta` (rad from vertical), image-up = -z-ish (camera upside down, base at image bottom)"""
    fwd = np.array([0, np.sin(theta), -np.cos(theta)])
    tgt = fwd
    R = B.camera_R({"pos": [0, 0, 0], "look_at": list(tgt), "up_hint": [0, 0, -1]}, wrist=False)
    return B.rot_z(yaw) @ R @ B.rot_z(roll)


def project(pos, R, pts):
    pc = (pts - pos) @ R                      # columns of R = camera axes in world -> p_cam = R^T (p - pos)
    return np.stack([320 + F * pc[:, 0] / -pc[:, 2], 240 - F * pc[:, 1] / -pc[:, 2]], 1)


def backproject(pos, R, px, z):
    d = R @ np.array([(px[0] - 320) / F, -(px[1] - 240) / F, -1.0])
    t = (z - pos[2]) / d[2]
    return pos + t * d


def residuals(p, h):
    x, y, th, yaw, roll = p
    pos = np.array([x, y, h + TZ])
    R = cam_R(th, yaw, roll)
    hole = project(pos, R, np.array([[0.007, -0.872, 0.042]]))[0]
    far = project(pos, R, np.array([[x, -1.26, TZ], [x + 0.3, -1.26, TZ], [x - 0.3, -1.26, TZ]]))
    l1, l2 = backproject(pos, R, (15, 60), TZ), backproject(pos, R, (30, 180), TZ)
    r = backproject(pos, R, (632, 180), TZ)
    plate = project(pos, R, np.array([[0.007 - 0.035, -0.872, 0.042], [0.007 + 0.035, -0.872, 0.042]]))
    res = [(hole[0] - 331) / 3, (hole[1] - 192) / 3,
           (far[0, 1] - 15) / 4, (far[1, 1] - 15) / 6, (far[2, 1] - 15) / 6,                   # far edge horizontal-ish at row ~15
           (l1[0] - l2[0]) / 0.01,                                                             # left edge parallel to y (same world x)
           ((l2[0] - r[0]) - 1.25) / 0.02,                                                     # (sign: image right = -x) board width 1.25 m
           (abs(plate[1, 0] - plate[0, 0]) - 34) / 2,
           yaw / 0.06, roll / 0.06]      # priors: camera is (almost) not twisted about the vertical / its optical axis
    return np.array(res)


def fit(h):
    best = None
    for th0 in (0.2, 0.4, 0.6):
        for yaw0 in (-0.1, 0.0, 0.1):
            for roll0 in (np.pi - np.pi + 0.0, ):
                r = least_squares(residuals, [0.03, -1.1, th0, yaw0, roll0], args=(h,), bounds=([-0.5, -2.0, 0.05, -0.6, -0.6], [0.5, -0.2, 1.4, 0.6, 0.6]))
                if best is None or r.cost < best.cost:
                    best = r
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--heights", type=float, nargs="+", default=[0.78, 1.07])
    ap.add_argument("--fovs", type=float, nargs="+", default=[69.0], help="horizontal FOV values to try (D415 RGB nominal 69)"); a = ap.parse_args()
    for h, fv in [(h, fv) for h in a.heights for fv in a.fovs]:
        FOV_H = fv; F = 320.0 / np.tan(np.radians(FOV_H) / 2)
        print(f"--- fov_h {fv:.0f}")
        r = fit(h)
        x, y, th, yaw, roll = r.x
        print(f"height {h:.2f} m: cost {r.cost:.2f}  pos ({x:.3f}, {y:.3f}, {h + TZ:.3f})  pitch from vertical {np.degrees(th):.1f} deg, yaw {np.degrees(yaw):.1f}, roll {np.degrees(roll):.1f}")
        print("   residuals (sigma units: hole x,y | far edge x3 | left-edge parallel | width | plate px | yaw, roll priors):", r.fun.round(2))
        pos = np.array([x, y, h + TZ]); R = cam_R(th, yaw, roll)
        print("   look direction", (-R[:, 2]).round(3), " board left edge world x", backproject(pos, R, (30, 180), TZ)[0].round(3), " right edge x", backproject(pos, R, (632, 180), TZ)[0].round(3))
