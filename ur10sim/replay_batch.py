#!/usr/bin/env python3
"""Run replay_episode.py over several episodes of a dataset (peg yaw chosen from the sign of the grasp x: the recorded grasp is on the
peg end named by the task -- square task grasps the ROUND end, circle task the SQUARE end) and summarise where the peg's lower end
ends up relative to the hole block at release.   python replay_batch.py --dataset square_hole_v21_clean --n 12"""
import argparse, re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ur10_kinematics as K

ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--n", type=int, default=12)
ap.add_argument("--peg-x", type=float, default=0.01); ap.add_argument("--peg-y", type=float, default=-0.662); a = ap.parse_args()
sq = "square" in a.dataset; import build_scene as _B; D = _B.DATASETS / a.dataset / "data" / "chunk-000"
eps = np.linspace(0, len(list(D.glob("*.parquet"))) - 1, a.n).astype(int); rows = []
for e in eps:
    S = np.stack(pd.read_parquet(D / f"episode_{e:06d}.parquet")["observation.state"]); g = S[:, 6]
    i0 = 80 + int(np.argmax(g[80:] > 0.3)); gx = K.forward_kinematics(S[i0 + 8, :6])[0][0]
    yaw = (0 if gx > 0 else 180) if sq else (180 if gx > 0 else 0)
    out = subprocess.run([sys.executable, str(HERE / "replay_episode.py"), "--dataset", a.dataset, "--episode", str(e), "--peg-x", str(a.peg_x), "--peg-y", str(a.peg_y),
                          "--peg-yaw", str(yaw), "--stride", "1", "--shots", "1"], capture_output=True, text=True).stdout
    m = re.search(r"\(([\d.]+) deg from vertical\), lower end = (\w+ end).*\n.*lower-end xy offset \[\s*([-\d.]+)\s+([-\d.]+)\] mm, TCP xy offset \[\s*([-\d.]+)\s+([-\d.]+)\] mm, lower end height above the table ([-\d.]+) mm", out)
    if not m: print(e, "no result"); continue
    tilt, end, dx, dy, tx, ty, h = m.groups(); rows.append([float(x) for x in (e, gx, yaw, tilt, dx, dy, tx, ty, h)])
    print(f"ep {e:3d} grasp x {gx:+.3f} yaw {yaw:3d}: tilt {tilt:>4s} deg, {end} down, lower end vs hole ({dx:>5s},{dy:>5s}) mm, TCP vs hole ({tx:>5s},{ty:>5s}) mm, tip height {h} mm")
R = np.array(rows)
print(f"\nmedian lower-end offset x {np.median(R[:, 4]):+.0f} y {np.median(R[:, 5]):+.0f} mm (std {R[:, 4].std():.0f}, {R[:, 5].std():.0f});  TCP offset x {np.median(R[:, 6]):+.0f} y {np.median(R[:, 7]):+.0f} mm;  tip height {np.median(R[:, 8]):.0f} mm")
