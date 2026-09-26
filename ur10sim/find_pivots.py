#!/usr/bin/env python3
"""Reproduce PIV_A in build_pincopen.py: detect circular through-holes (pivot pins) in the linkage-part meshes exported in
ASSEMBLY coordinates (CAD/PincOpen_Gripper/solidworks/try2/print_3), by RANSAC circle fits on the cylinder walls (triangles whose
normal is ~perpendicular to the hinge axis = assembly z).  Prints (x_a, y_a, r, n_points) in mm.  A pivot = holes of two different
links that coincide (e.g. Driving_Rod-1 far hole == Internal_Rod-2 near hole at (75.5, 103.4))."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pincopen import SRC, PREFIX, load_stl  # noqa: E402


def circles(w, rmin=0.8, rmax=3.4, min_in=9, tol=0.06, seed=1):
    n = np.cross(w[:, 1] - w[:, 0], w[:, 2] - w[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-18
    P = np.unique(np.round(w[np.abs(n[:, 2]) < 0.2][:, :, [0, 1]].reshape(-1, 2), 3), axis=0)
    rng = np.random.default_rng(seed); out = []
    for _ in range(40):
        if len(P) < 8: break
        best = None
        for _ in range(2500):
            s = P[rng.choice(len(P), 3, replace=False)]
            try: sol = np.linalg.solve(np.c_[2 * s, np.ones(3)], (s ** 2).sum(1))
            except np.linalg.LinAlgError: continue
            c = sol[:2]; r2 = sol[2] + c @ c
            if r2 <= 0 or not (rmin < np.sqrt(r2) < rmax): continue
            inl = np.abs(np.linalg.norm(P - c, axis=1) - np.sqrt(r2)) < tol
            if best is None or inl.sum() > best[0]: best = (inl.sum(), inl)
        if best is None or best[0] < min_in: break
        Q = P[best[1]]; sol = np.linalg.lstsq(np.c_[2 * Q, np.ones(len(Q))], (Q ** 2).sum(1), rcond=None)[0]; c = sol[:2]
        out.append((round(c[0], 1), round(c[1], 1), round(np.sqrt(sol[2] + c @ c), 2), int(best[0]))); P = P[~best[1]]
    return out


if __name__ == "__main__":
    for n in ["Cam-1", "Driving_Rod-1", "Driving_Rod-2", "Internal_Rod-1", "Internal_Rod-2", "External_Rod-1", "External_Rod-2", "Distal_Rod-1", "Distal_Rod-2"]:
        print(n, circles(load_stl(SRC / f"{PREFIX}{n}.STL"))[:8], flush=True)
