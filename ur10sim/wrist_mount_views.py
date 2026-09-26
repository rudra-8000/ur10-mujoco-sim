#!/usr/bin/env python3
"""Show the two ways the D435i can be screwed onto the CAD camera mount (2 holes -> 2 orientations) and what each lens sees.

  python ur10sim/wrist_mount_views.py            # -> ur10sim/out/wrist_mount_orientations.png

Row per orientation (rgb_toward_z = +1 / -1): close-up of the mount + camera with the lenses labelled, then the sim wrist image from the
RGB lens, the left IR imager and the right IR imager, next to the real wrist frame of the configured dataset frame (if any).
Mount angle etc. come from scene_config.yaml (cameras.wrist.mount).
"""
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene_view as SV  # noqa: E402

SV._setup_gl("egl")
import numpy as np  # noqa: E402
import mujoco  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import build_scene as B  # noqa: E402


def closeup(scene, F, W=640, H=480):
    m, d = scene.model, scene.data
    gid = scene.bid("d435i")
    Rb, pb = d.xmat[gid].reshape(3, 3), d.xpos[gid]
    look = pb + Rb @ np.array([0.0, 0.0, -0.012])
    lens_dir = Rb[:, 2]                                  # viewing direction of the D435i in the world
    off = lens_dir * 0.55 + np.array([0.0, 0.0, 0.25]) + Rb[:, 0] * 0.0
    off /= np.linalg.norm(off)
    fwd = -off
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = look
    cam.distance = 0.30
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    cam.elevation = float(np.degrees(np.arcsin(fwd[2])))
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, camera=cam)
    img = r.render().copy()
    # project the lenses
    camp = look - cam.distance * fwd
    up0 = np.array([0, 0, 1.0]); right = np.cross(fwd, up0); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    f = 0.5 * H / np.tan(np.radians(45) / 2)
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    for name, p in F["dmeta"]["lenses"].items():
        w = pb + Rb @ np.array(p); v = w - camp
        x, y, z = v @ right, v @ up, v @ fwd
        u, vv = W / 2 + f * x / z, H / 2 - f * y / z
        dr.ellipse([u - 4, vv - 4, u + 4, vv + 4], outline=(255, 255, 0))
        dr.text((u + 6, vv - 6), name, fill=(255, 255, 0))
    return np.asarray(im)


def main():
    cfg0 = B.load_config()
    W, H = cfg0["render"]["width"], cfg0["render"]["height"]
    real = B.real_frames(cfg0, W, H)
    rows = []
    for sgn in (1, -1):
        cols = []
        for k, lens in enumerate(("rgb", "ir_left", "ir_right")):
            cfg = copy.deepcopy(cfg0)
            cfg["cameras"]["wrist"]["mount"].update(enabled=True, rgb_toward_z=sgn, lens=lens)
            sc = B.Scene(cfg)
            F = B.wrist_mount_frames(cfg["cameras"]["wrist"])
            if k == 0:
                cols.append(SV._label(closeup(sc, F), f"orientation rgb_toward_z={sgn:+d}"))
            cols.append(SV._label(sc.render(W, H)["wrist"], f"wrist view, lens={lens}"))
            sc.close()
        if real is not None:
            cols.append(SV._label(real[1], "REAL cam_right_wrist"))
        rows.append(np.concatenate(cols, axis=1))
    out = np.concatenate(rows, axis=0)
    p = B.REPO / cfg0["render"]["out_dir"] / "wrist_mount_orientations.png"
    Image.fromarray(out).save(p)
    print("wrote", p, out.shape)


main()
