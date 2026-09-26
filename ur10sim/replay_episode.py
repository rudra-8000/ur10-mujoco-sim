#!/usr/bin/env python3
"""Replay a dataset episode in the MuJoCo scene (arm kinematic from the recorded joints, gripper cam driven from the recorded gripper value,
peg + hole physics on) and check that peg / hole placement is consistent with what the robot did.

  python ur10sim/replay_episode.py --dataset square_hole_v21_clean --episode 3 --hole square --peg-yaw 0
Options: --peg-x/--peg-y (start pose of the peg centre, default from the config's peg block), --peg-yaw (0 or 180), --stride, --shots N.
Outputs (sim/out/replay_<dataset>_ep<k>_yaw<y>.png): sim top | sim wrist | real top | real wrist at N evenly spaced frames, plus a text summary:
peg-to-TCP distance at grasp, max peg lift, peg axis/position relative to the hole at the end."""
import argparse, copy, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_view as SV
SV._setup_gl("egl")
import numpy as np
import pandas as pd
import mujoco
from PIL import Image
import build_scene as B
import pincopen as PC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="square_hole_v21_clean"); ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--hole", default=None, choices=["circle", "square"], help="default: from the dataset name")
    ap.add_argument("--peg-x", type=float, default=None); ap.add_argument("--peg-y", type=float, default=None)
    ap.add_argument("--peg-yaw", type=float, default=None); ap.add_argument("--stride", type=int, default=1); ap.add_argument("--no-attach", action="store_true", help="pure physics grasp (the sim grasp is not reliable yet); default: from the closing frame to the release frame the peg is welded to the gripper by the grasp assist once both pads touch it")
    ap.add_argument("--squeeze", type=float, default=0.08, help="extra gripper fraction commanded while holding (the real servo is current limited and squeezes past the measured plateau)")
    ap.add_argument("--video", default=None, help="write an mp4 (sim top | wrist | overview over real top | wrist)")
    ap.add_argument("--vstride", type=int, default=2, help="video frame every N dataset frames (30 fps data -> 15 fps video for 2)")
    ap.add_argument("--shots", type=int, default=8)
    a = ap.parse_args()
    cfg = B.load_config(); W, H = cfg["render"]["width"], cfg["render"]["height"]
    hole = a.hole or ("square" if "square" in a.dataset else "circle")
    for k in ("circle", "square"):
        cfg["holes"][k]["enabled"] = (k == hole)
    cfg["peg"]["mode"] = "table"
    for k, v in (("x", a.peg_x), ("y", a.peg_y), ("yaw_deg", a.peg_yaw)):
        if v is not None: cfg["peg"][k] = v
    df = pd.read_parquet(B.DATASETS / a.dataset / "data" / f"chunk-{a.episode // 1000:03d}" / f"episode_{a.episode:06d}.parquet")
    S = np.stack(df["observation.state"]); N = len(S)
    s0 = int(np.argmax(S[:, 6] < 0.05))                 # episodes start with the gripper closed on nothing; begin at the first open frame
    cfg["robot"]["source"] = "dataset"; cfg["robot"]["dataset"] = dict(name=a.dataset, episode=a.episode, frame=s0)
    sc = B.Scene(cfg); m, d = sc.model, sc.data
    peg_b, gid = sc.bid("peg"), sc.bid("pincopen_base")
    hb = sc.bid("hole_" + hole); tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    lo, hi = sc.ctrl_range
    print(f"peg start {d.xpos[peg_b].round(3)} (yaw {cfg['peg']['yaw_deg']}), hole {hole} at {d.xpos[hb].round(3)}, {N} frames, replay from frame {s0}")
    sub = 33 // a.stride if a.stride > 1 else 33
    g_all = S[:, 6]; i0_ = s0 + int(np.argmax(g_all[s0:] > 0.3)); plate_ = np.median(g_all[i0_:][g_all[i0_:] > 0.3])
    iC_ = i0_ + int(np.argmax(g_all[i0_:] >= 0.95 * plate_)); iN_ = iC_ + int(np.argmax(g_all[iC_:] < 0.85 * plate_))
    rel = None; cmd = float(d.ctrl[0])
    shots = sorted(np.linspace(s0, N - 1, a.shots).astype(int).tolist())
    vid = None
    if a.video:
        import subprocess
        PW, PH = 420, 315
        def real_stack(cam):
            p = B.DATASETS / a.dataset / "videos" / f"chunk-{a.episode // 1000:03d}" / f"observation.images.{cam}" / f"episode_{a.episode:06d}.mp4"
            raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-vf", f"select=not(mod(n\\,{a.vstride})),scale={PW}:{PH}", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                 capture_output=True).stdout
            return np.frombuffer(raw, np.uint8).reshape(-1, PH, PW, 3)
        rt, rw = real_stack("cam_high"), real_stack("cam_right_wrist")
        vdir = Path(a.video).with_suffix(""); vdir.mkdir(parents=True, exist_ok=True)
        from PIL import Image as _I
        rs = lambda x: np.asarray(_I.fromarray(x).resize((PW, PH)))
        vid = 0
    log = []; tiles = []; q_prev = S[s0, :6]; t0 = time.time()
    for k in range(s0, N, a.stride):
        tgt = float(np.clip(PC.cam_angle_for_frac(S[k, 6] + (a.squeeze if S[k, 6] > 0.3 else 0.0)), lo, hi))
        for j in range(sub):
            qa = q_prev + (S[k, :6] - q_prev) * (j + 1) / sub                      # kinematic arm: set the pose AND the joint velocity so that contacts (friction) see the motion
            d.qvel[sc.arm_dof] = (qa - d.qpos[sc.arm_adr]) / m.opt.timestep; d.qpos[sc.arm_adr] = qa
            cmd += float(np.clip(tgt - cmd, -6e-3, 6e-3)); d.ctrl[0] = cmd     # servo speed limit 6 rad/s (fast steps flip the linkage branch)
            if not a.no_attach:
                sc.grasp_assist(bool(iC_ - 10 <= k < iN_))     # weld the peg once both pads touch it (see Scene.grasp_assist)
            mujoco.mj_step(m, d)
        q_prev = S[k, :6]
        d.qpos[sc.arm_adr] = q_prev; mujoco.mj_forward(m, d)
        Rp = d.xmat[peg_b].reshape(3, 3)
        log.append((k, S[k, 6], *d.xpos[peg_b], *Rp[:, 2], *d.site_xpos[tcp]))     # peg axis = body z (square end -z, round end +z)
        if vid is not None and (k % a.vstride) == 0 and k // a.vstride < len(rt):
            im = sc.render(W, H); i_ = k // a.vstride
            top = np.concatenate([SV._label(rs(im["top"]), f"SIM top  f{k}  grip {S[k, 6]:.2f}"), SV._label(rs(im["wrist"]), "SIM wrist"), SV._label(rs(im["overview"]), "SIM overview")], 1)
            bot = np.concatenate([SV._label(rt[i_], "REAL top"), SV._label(rw[min(i_, len(rw) - 1)], "REAL wrist"), np.zeros_like(rt[i_])], 1)
            _I.fromarray(np.concatenate([top, bot], 0)).save(vdir / f"{vid:05d}.png"); vid += 1
        if shots and k >= shots[0]:
            shots.pop(0)
            im = sc.render(W, H); c2 = copy.deepcopy(cfg); c2["robot"]["dataset"]["frame"] = int(k)
            real = B.real_frames(c2, W, H)
            tiles.append(np.concatenate([SV._label(im["top"], f"sim top f{k}"), SV._label(im["wrist"], f"sim wrist g={S[k, 6]:.2f}"),
                                         SV._label(real[0], "real top"), SV._label(real[1], "real wrist")], axis=1))
    if vid:
        import subprocess as _sp, shutil
        _sp.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(30 // a.vstride), "-i", str(vdir / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.video], check=True)
        shutil.rmtree(vdir); print("wrote video", a.video, f"({vid} frames)")
    print(f"replayed in {time.time() - t0:.0f}s")
    L = np.array(log); g = L[:, 1]; P = L[:, 2:5]; TC = L[:, 8:11]
    i0 = int(np.argmax(g > 0.3)); plate = np.median(g[i0:][g[i0:] > 0.3]); iC = i0 + int(np.argmax(g[i0:] >= 0.95 * plate)); iN = iC + int(np.argmax(g[iC:] < 0.85 * plate))   # release = gripper opening below 85 % of the grasp plateau
    print(f"first close frame {L[i0, 0]:.0f}: peg centre - TCP = {(P[i0] - TC[i0]).round(3)} (norm {np.linalg.norm(P[i0] - TC[i0]) * 1e3:.0f} mm)")
    print(f"peg z: start {P[0, 2]:.3f}, carried max {P[i0:iN, 2].max():.3f} (lift {(P[i0:iN, 2].max() - P[0, 2]) * 1e3:.0f} mm); peg-TCP distance during carry (median) {np.median(np.linalg.norm(P[i0:iN] - TC[i0:iN], axis=1)) * 1e3:.0f} mm, max {np.linalg.norm(P[i0:iN] - TC[i0:iN], axis=1).max() * 1e3:.0f} mm")
    hp = d.xpos[hb]; j = max(iN - 1, 0); ax = L[j, 5:8]; ends = {"square end": P[j] - 0.1035 * ax, "round end": P[j] + 0.097 * ax}
    low = min(ends, key=lambda e: ends[e][2])
    print(f"at release (frame {L[j, 0]:.0f}): peg centre {P[j].round(3)}, axis {ax.round(2)} ({np.degrees(np.arccos(abs(ax[2]))):.0f} deg from vertical), lower end = {low} at {ends[low].round(3)}")
    print(f"   hole block centre {hp.round(3)}: lower-end xy offset {((ends[low] - hp)[:2] * 1e3).round(0)} mm, TCP xy offset {((TC[j] - hp)[:2] * 1e3).round(0)} mm, lower end height above the table {(ends[low][2] - cfg['table']['z']) * 1e3:.0f} mm")
    out = B.REPO / cfg["render"]["out_dir"] / f"replay_{a.dataset}_ep{a.episode}_yaw{int(cfg['peg']['yaw_deg'])}.png"
    Image.fromarray(np.concatenate(tiles, axis=0)).save(out); print("wrote", out)


main()
