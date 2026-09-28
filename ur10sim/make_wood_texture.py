#!/usr/bin/env python3
"""Procedural plywood-like texture for the board (streaks along y, a few darker/lighter veneer patches).  python make_wood_texture.py [amplitude]
-> assets/table/wood.png (1024x1024, mean colour ~ mid grey-beige; the camera appearance map (calibrate_appearance.py) fixes the final colours)."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
amp = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
rng = np.random.default_rng(3)
N = 1024
def streak(sx, sy):                       # long along rows (image y) -> vertical grain lines
    z = gaussian_filter(rng.normal(size=(N, N)), (sy, sx), mode="wrap"); return z / z.std()
v = 0.55 * streak(1.2, 60) + 0.35 * streak(3.0, 120) + 0.25 * streak(0.6, 25)
patch = gaussian_filter(rng.normal(size=(N, N)), 90, mode="wrap"); patch /= patch.std()
lum = 1.0 + amp * v + 0.5 * amp * patch
base = np.array([0.62, 0.62, 0.64])           # pale grey-beige: the real cameras see the plywood as (158,159,164) top / (125,124,119) wrist (measured)
img = np.clip(lum[..., None] * base, 0, 1)
out = Path(__file__).resolve().parent / "assets" / "table" / "wood.png"
Image.fromarray((img * 255).astype(np.uint8)).save(out); print("wrote", out)
