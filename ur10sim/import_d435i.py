#!/usr/bin/env python3
"""Import the Intel RealSense D435i visual model from MuJoCo Menagerie (Apache-2.0, kept next to the meshes as LICENSE) into
assets/d435i/, decimated by vertex clustering (the source OBJs are 45 MB).  Model frame (as in Menagerie):
origin = centre of the FRONT face, +z = out of the lenses (viewing direction), x = long axis (RGB lens at +x), y = device up when seen from the
front with the RGB lens on the right (so, in the image of a lens, right = -x, up = +y).
Writes assets/d435i/{meshes/*.stl, d435i_meta.json}."""
import json, shutil
from pathlib import Path
import numpy as np
from build_pincopen import write_stl
import os
SRC = Path(os.environ.get("MENAGERIE_D435I", "third_party/mujoco_menagerie/realsense_d435i"))   # a checkout of google-deepmind/mujoco_menagerie/realsense_d435i
OUT = Path(__file__).resolve().parent / "assets" / "d435i"
NAMES = ["IR_Lens", "IR_Emitter_Lens", "IR_Rim", "IR_Lens", "Cameras_Gray", "Black_Acrylic", "Black_Acrylic", "RGB_Pupil", "Metal_Casing"]
GRID = [0.1e-3, 0.1e-3, 0.15e-3, 0.1e-3, 0.4e-3, 0.3e-3, 0.15e-3, 0.08e-3, 0.5e-3]

def load_obj(p):
    V, F = [], []
    for l in open(p):
        if l.startswith("v "): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "): F.append([int(t.split("/")[0]) - 1 for t in l.split()[1:]])
    F = [[f[0], f[i], f[i + 1]] for f in F for i in range(1, len(f) - 1)]
    return np.array(V), np.array(F)

def decimate(V, F, g):
    key = np.round(V / g).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    C = np.zeros((inv.max() + 1, 3)); n = np.bincount(inv).astype(float)
    for k in range(3): C[:, k] = np.bincount(inv, V[:, k]) / n
    F2 = inv[F]; ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    return C[F2[ok]]

def main():
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / "LICENSE", OUT / "LICENSE_Apache2_MuJoCo_Menagerie")
    meta = {"source": "MuJoCo Menagerie realsense_d435i (Apache-2.0)", "parts": []}
    for i in range(9):
        V, F = load_obj(SRC / "assets" / f"d435i_{i}.obj")
        tris = decimate(V, F, GRID[i]); write_stl(OUT / "meshes" / f"d435i_{i}.stl", tris)
        meta["parts"].append({"mesh": f"d435i_{i}", "material": NAMES[i], "tris": int(len(tris)), "src_tris": int(len(F))})
        print(i, NAMES[i], len(F), "->", len(tris))
    # lens centres in the model frame (m): imager/pupil glass measured from the meshes (see README)
    meta["lenses"] = {"ir_left": [-0.0325, 0, -0.0012], "emitter": [-0.0115, 0, -0.0039], "ir_right": [0.0175, 0, -0.0012], "rgb": [0.0325, 0, -0.0036]}
    meta["size_m"] = [0.090, 0.025, 0.025]; meta["rgb_fov_deg"] = [69, 42]
    (OUT / "d435i_meta.json").write_text(json.dumps(meta, indent=1))
main() if __name__ == "__main__" else None
