#!/usr/bin/env python3
"""Place the try3 camera mount (CAMERA MOUNT_V2 / V6_REINFORCED, STLs in *print* orientation) into the gripper frame G and
work out its pivot, its two D435i screw holes and the outward normal of the hole plate.

How the mount pose is obtained: the try2 print_3 assembly export contains the mount (V2-2) in assembly coordinates, in the tilted
state it had in that assembly (CAD tilt ~32 deg).  The dense print-orientation mesh is registered onto it by ICP (rms 0.25 mm) and
the same transform is used for V6 (identical outer geometry).  The screw-hole / pivot-hole positions are read off the try2 print_2
export of the same part (vertical plate; hole circles fitted to the mesh) and carried over through a second registration.

Result (assets/wrist_camera/wrist_camera_meta.json + meshes/camera_mount_*.stl, vertices relative to the pivot, in G, metres):
  pivot_G           point on the mount's pivot-hole axis
  pivot_axis_G      unit axis (z of G: the jaw / pinch direction)
  hole_mid_G        midpoint of the two D435i screw holes on the outer face of the plate (CAD pose)
  hole_dir_G        unit vector along the two holes (the camera long axis)
  normal_G          outward normal of the plate (the direction the lenses look at mount_angle 0)
"""
import json
from pathlib import Path
import numpy as np
import register_mount as RM
from build_pincopen import load_stl, write_stl, to_G

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets" / "wrist_camera"
S = RM.S
RM.sample.__defaults__ = (20000, 0)


def fit_circle(pts):
    A = np.c_[2 * pts, np.ones(len(pts))]; b = (pts ** 2).sum(1)
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    return c[:2], float(np.sqrt(c[2] + (c[:2] ** 2).sum()))


def main():
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)
    asm3 = load_stl(S + "try2/print_3/TRY2_ASS - CAMERA MOUNT_V2-2.STL")
    p2 = load_stl(S + "try3/print/CAMERA MOUNT_V2.STL")
    e1, R1, t1 = RM.register(p2, asm3)          # print -> print_3 assembly (tilted)
    print(f"registration rms to the print_3 assembly part: {e1:.3f} mm")

    # ---- features in the print frame (mm), V6 (dense enough, same outer geometry).  Print orientation: long axis z, pivot disc on the
    #      z=0..7.5 slab (axis z), thin plate on the -x side with its two D435i screw holes (through, ~x direction), spaced along z.
    v = np.unique(load_stl(S + "try3/print/CAMERA MOUNT_V2.STL").reshape(-1, 3).round(3), axis=0)
    sel = v[(v[:, 2] < 7.4) & (np.hypot(v[:, 0] - 168.0, v[:, 1] - 62.4) < 3.0)]
    pc, pr = fit_circle(sel[:, :2])
    print("pivot hole (x,y) mm:", pc.round(2), "radius", round(pr, 2), f"({len(sel)} verts)")
    # outer plate face: for each y-bin the min x -> line x = a + b*y
    ca = v[(v[:, 1] < 56) & (v[:, 0] < 156)]; cb = v[(v[:, 1] > 72) & (v[:, 0] < 158.5)]       # the two outer-face corners (x, y)
    pa, pb = ca[np.argmin(ca[:, 0]), :2], cb[np.argmin(cb[:, 0]), :2]
    b_ = (pb[0] - pa[0]) / (pb[1] - pa[1]); a_ = pa[0] - b_ * pa[1]
    nrm = np.array([-1.0, b_, 0.0]); nrm /= np.linalg.norm(nrm)             # outward (-x side); face line x = a + b y
    print(f"outer face corners {pa} {pb}: tilt {np.degrees(np.arctan(b_)):.2f} deg, normal {nrm.round(4)}")
    hs = []
    for z0 in (12.6, 57.6):
        s2 = v[(np.hypot(v[:, 1] - 63.2, v[:, 2] - z0) < 2.4) & (v[:, 0] < 160)]
        c, r = fit_circle(s2[:, 1:]); hs.append((c, r, len(s2)))
    print("screw holes (y,z) mm:", [(h[0].round(2).tolist(), round(h[1], 2), h[2]) for h in hs], "spacing", round(np.linalg.norm(hs[0][0] - hs[1][0]), 2))
    yh = (hs[0][0][0] + hs[1][0][0]) / 2; zh = (hs[0][0][1] + hs[1][0][1]) / 2
    mid_p = np.array([a_ + b_ * yh, yh, zh])                  # on the outer face, between the two holes
    piv_p = np.array([pc[0], pc[1], 3.75])
    r1_ = hs[0][1]
    R1 = np.asarray(R1); Rg = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
    def P(p): return to_G(R1 @ p + t1)
    def D(d): return Rg @ (R1 @ d)
    piv_G, mid_G = P(piv_p), P(mid_p)
    n_G = D(nrm); n_G[2] = 0; n_G /= np.linalg.norm(n_G); u_G = np.array([0, 0, -1.0]); ax_G = np.array([0, 0, -1.0])   # snapped: print z = -z_g
    print("pivot_G [mm]", (piv_G * 1e3).round(2), " hole_mid_G [mm]", (mid_G * 1e3).round(2))
    print("normal_G", n_G.round(3), "hole_dir_G", u_G.round(3), "pivot_axis_G", ax_G.round(3))
    ang = np.degrees(np.arctan2(n_G[1], n_G[0]))
    print(f"CAD pose: lens axis is {ang:.1f} deg from +x_g (fingers) in the x-y plane of G")
    h1, h2 = hs[0][0], hs[1][0]

    meta = {"pivot_G": piv_G.tolist(), "pivot_axis_G": ax_G.tolist(), "hole_mid_G": mid_G.tolist(), "hole_dir_G": u_G.tolist(),
            "normal_G": n_G.tolist(), "hole_spacing_mm": float(np.linalg.norm(h1 - h2)), "hole_radius_mm": float(hs[0][1]),
            "plate_outer_face_to_pivot_mm": float(np.linalg.norm(mid_G - piv_G) * 1e3), "cad_lens_axis_deg_from_x_g": float(ang),
            "registration_rms_mm": [e1], "versions": {}}
    for ver in ("V2", "V6_REINFORCED"):
        t = load_stl(S + f"try3/print/CAMERA MOUNT_{ver}.STL")
        tG = to_G(t.reshape(-1, 3) @ R1.T + t1).reshape(-1, 3, 3) - piv_G
        write_stl(OUT / "meshes" / f"camera_mount_{ver}.stl", tG)
        meta["versions"][ver] = {"tris": int(len(t))}
    (OUT / "wrist_camera_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
