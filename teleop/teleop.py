#!/usr/bin/env python3
"""Standalone UR10 teleoperation (GELLO leader arm or Meta Quest 3) with a live
camera view. Drives the real robot, or (with --sim) the MuJoCo sim in
../ur10sim instead -- same GELLO/Quest code either way.

  python teleop.py --method gello --sim               # MuJoCo sim, no real robot at all
  python teleop.py --method gello                      # dry-run: display only
  python teleop.py --method gello --live               # drives the real robot
  python teleop.py --method quest --live
  python teleop.py --method quest --sim --sim-case single_square

Before driving the real robot (or the sim), GELLO teleop asks whether you
want to (re)calibrate the leader arm -- see "Calibration" in GUIDE.md. Quest
has no saved calibration file; its equivalent is the in-headset left-grip
anchor action, done live each time you start tracking (see GUIDE.md). Either
way this is about the *leader device*, real hardware even in --sim mode --
only the robot side is simulated.

Camera view: an X11 window (works over `ssh -X`; needs $DISPLAY) and, for
Quest, a panel inside the headset. --no-video disables the window.
Keys: SPACE / q / e / Esc = stop (robot holds), Enter or s = skip the GELLO
trigger gate. Run on the robot PC (your lerobot env active) for --live; see
GUIDE.md. --sim needs no special environment beyond this repo's own deps.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import teleop_core as T  # noqa: E402
from robot import client_ur10 as _c  # noqa: E402


def _fake_obs():
    return {
        **{f"joint_{i}": float(_c.HOME_RAD[i]) for i in range(6)},
        "gripper": 0.0,
        "cam_high": np.zeros((480, 640, 3), dtype=np.uint8),
        "cam_right_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
    }


def _maybe_calibrate_gello(args) -> bool:
    """Ask whether to (re)calibrate before connecting. Returns True if the
    caller should force a fresh calibration once connected.

    - No calibration file yet: nothing to ask -- lerobot will walk you
      through it (move to home pose, press ENTER) the first time it connects.
    - A calibration file exists: ask. This is the prompt the user of this
      repo actually wants -- normally lerobot only re-asks when a mismatch
      is detected, which you can't see coming.
    """
    if args.no_calibrate_prompt:
        return args.force_calibrate
    if args.force_calibrate:
        print("--force-calibrate: will recalibrate after connecting.")
        return True
    try:
        path = T.gello_calibration_path(args.teleop_id)
    except Exception as e:  # noqa: BLE001 -- e.g. lerobot not importable here; let _connect_gello give the real error
        print(f"(could not check for an existing calibration file: {type(e).__name__}: {e})")
        return False
    if not path.is_file():
        print(f"No GELLO calibration found for id '{args.teleop_id}' -- "
              f"you'll be asked to move it to the home pose and press ENTER.")
        return False
    ans = input(f"Existing GELLO calibration found ({path}).\n"
                f"Recalibrate now? [y/N]: ").strip().lower()
    return ans in ("y", "yes")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="UR10 teleop: GELLO or Quest, real robot or MuJoCo sim, with camera view")
    p.add_argument("--method", required=True, choices=["gello", "quest"])
    p.add_argument("--live", action="store_true", help="drive the real robot (default: dry-run, display only)")
    p.add_argument("--sim", action="store_true", help="drive the MuJoCo sim (../ur10sim) instead of the real robot")
    p.add_argument("--sim-case", default="single_circle",
                    choices=["single_circle", "single_square", "multi_circle", "multi_square"],
                    help="--sim only: which ur10sim.rl_env scene to load")
    p.add_argument("--sim-yaw", type=float, default=0.0, help="--sim only: peg start yaw in degrees (0 or 180)")
    p.add_argument("--ur-ip", default="192.168.100.3")
    p.add_argument("--gripper-port", default=None, help="PincOpen/Dynamixel gripper USB port (default: see robot/client_ur10.py)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--tcp-offset-mm", type=float, default=145.0)
    p.add_argument("--force-calibrate", action="store_true", help="recalibrate GELLO without asking")
    p.add_argument("--no-calibrate-prompt", action="store_true",
                   help="never ask about calibration (use --force-calibrate to also force a recalibration)")
    T.add_teleop_args(p)
    args = p.parse_args()
    if args.sim and args.live:
        p.error("--sim and --live are mutually exclusive (sim never touches the real robot)")
    args.dry_run = not args.live and not args.sim

    force_recalibrate = False
    if args.method == "gello":
        force_recalibrate = _maybe_calibrate_gello(args)

    if args.sim:
        from sim_backend import SimRobotIO
        import sim_backend as _sim_c
        print(f"starting MuJoCo sim ({args.sim_case}, peg yaw {args.sim_yaw})...")
        io = SimRobotIO(case=args.sim_case, yaw=args.sim_yaw)
        robot = io
        h = types.SimpleNamespace(c=_sim_c, gripper_open=0.0, fake_obs=_fake_obs)
    else:
        h = types.SimpleNamespace(c=_c, gripper_open=0.0, fake_obs=_fake_obs)
        if args.live:
            input("LIVE: the robot WILL move. Workspace clear, e-stop in reach? Press ENTER to continue (Ctrl-C cancels)… ")
        robot = T._connect_robot(args, h)
        io = T.RobotIO(robot, h)

    gello = quest = None
    if args.method == "gello":
        gello, err = T._connect_gello(args)
        if gello is None:
            print(f"GELLO unavailable: {err}")
            return 1
        if force_recalibrate:
            print("Recalibrating GELLO now...")
            # connect() (via _connect_gello) already started the async background
            # read thread (use_async=True by default), which continuously reads
            # the Dynamixel bus -- calibrate()'s disable_torque() WRITE races with
            # that thread's READs on the same serial port ("Port is in use!") if
            # it's still running. Stop it first, recalibrate + reconfigure exactly
            # like a normal connect() would, then restart it the same way connect()
            # does.
            was_async = bool(getattr(gello.config, "use_async", False) and gello.thread is not None)
            if was_async:
                gello._stop_read_thread()
            gello.calibration = None       # bypass the "keep existing?" prompt; go straight to capture
            gello.calibrate()
            gello.configure()
            if was_async:
                raw_action = gello.bus.sync_read("Present_Position", normalize=False)
                gello.latest_action = gello._process_action(raw_action)
                gello._start_read_thread()
    else:
        from quest_teleop import start_server
        quest, err = start_server(args)
        if quest is None:
            print(f"Quest server unavailable: {err}")
            return 1
        print(f"\nOpen this on the Quest browser (accept the self-signed cert): {quest.url}\n")
    feed = T.VideoFeed(not args.no_video, args.video_width, args.video_fps)
    if quest is not None:
        feed.sinks.append(quest.push_cam)
    if not feed.alive and not feed.sinks:
        logging.warning("video window: %s", feed.status)
    pump = T.CamPump(io, feed)
    pump.caption = f"TELEOP ({args.method}{' sim' if args.sim else ''})"
    pump.start()

    if args.method == "gello":
        w = T.Teleop(args, io, gello)
    else:
        from quest_teleop import QuestTeleop
        w = QuestTeleop(args, h, io, quest)

    fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    old = None
    if fd is not None:
        import termios
        import tty
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    print("SPACE/q/e/Esc = stop     Enter/s = skip GELLO trigger gate\n")
    try:
        while not w.done.is_set():
            if fd is not None:
                import select
                if select.select([fd], [], [], 0)[0]:
                    k = sys.stdin.read(1)
                    if k in (" ", "q", "e", "\x1b"):
                        w.abort.set()
                    elif k in ("\n", "\r", "s") and hasattr(w, "start_now"):
                        w.start_now.set()
            S = w.S
            print(f"\r[{S['phase']:<15}] {S['hz']:5.1f} Hz  gripper {S['gripper']:.2f}  "
                  f"joints {[round(v) for v in S['joints']] if not np.isnan(S['joints'][0]) else '--'}   ", end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        w.abort.set()
    finally:
        print()
        w.abort.set()
        w.done.wait(10)
        if fd is not None and old is not None:
            import termios
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        pump.stop()
        feed.stop()
        if quest is not None:
            quest.stop()
        if gello is not None:
            try:
                gello.disconnect()
            except Exception:  # noqa: BLE001
                pass
        if robot is not None and robot.is_connected:
            robot.disconnect()
    if w.S["error"]:
        print("ERROR:", w.S["error"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
