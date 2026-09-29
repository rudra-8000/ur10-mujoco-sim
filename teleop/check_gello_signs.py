#!/usr/bin/env python3
"""One-shot per-joint sign check for a GELLO leader arm, against the sim or
the real robot -- isolates each joint in turn instead of relying on a
subjective "something looked reversed" during normal teleop, which is too
ambiguous to act on safely (flipping the wrong sign, or the right sign the
wrong way, just trades one confusing symptom for another).

  python check_gello_signs.py --sim
  python check_gello_signs.py --live --ur-ip 192.168.100.3

For each of the 6 joints: move ONLY that joint by hand (either direction,
any amount), watch the video window, then answer one yes/no. At the end,
prints the exact `joint_signs` line to edit for any joint that comes back
"no" -- in

  lerobot_ur10/src/lerobot/teleoperators/lerobot_teleoperator_gello/config_gello.py

Does not calibrate or move the robot to a specific pose first -- run this
right after a normal teleop session where you already noticed something odd,
GELLO already roughly aligned. Ctrl-C at any prompt aborts cleanly.
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

JOINT_LABELS = ["joint_0 (base)", "joint_1 (shoulder)", "joint_2 (elbow)",
                "joint_3 (wrist1)", "joint_4 (wrist2)", "joint_5 (wrist3)"]


def _fake_obs():
    return {
        **{f"joint_{i}": float(_c.HOME_RAD[i]) for i in range(6)},
        "gripper": 0.0,
        "cam_high": np.zeros((480, 640, 3), dtype=np.uint8),
        "cam_right_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING)  # keep this script's own output the focus
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", action="store_true", help="check against the MuJoCo sim (mutually exclusive with --live)")
    p.add_argument("--live", action="store_true", help="check against the real robot")
    p.add_argument("--sim-case", default="single_circle",
                    choices=["single_circle", "single_square", "multi_circle", "multi_square"])
    p.add_argument("--ur-ip", default="192.168.100.3")
    p.add_argument("--gripper-port", default=None)
    p.add_argument("--fps", type=int, default=30)
    T.add_teleop_args(p)
    args = p.parse_args()
    if args.sim == args.live:
        p.error("pass exactly one of --sim or --live")
    args.dry_run = False

    if args.sim:
        import sim_backend as _sim_c
        print(f"starting MuJoCo sim ({args.sim_case})...")
        io = _sim_c.SimRobotIO(case=args.sim_case)
        h = types.SimpleNamespace(c=_sim_c, gripper_open=0.0, fake_obs=_fake_obs)
    else:
        h = types.SimpleNamespace(c=_c, gripper_open=0.0, fake_obs=_fake_obs)
        input("LIVE: the robot WILL move. Workspace clear, e-stop in reach? Press ENTER to continue (Ctrl-C cancels)… ")
        robot = T._connect_robot(args, h)
        io = T.RobotIO(robot, h)

    gello, err = T._connect_gello(args)
    if gello is None:
        print(f"GELLO unavailable: {err}")
        return 1

    feed = T.VideoFeed(not args.no_video, args.video_width, args.video_fps)
    pump = T.CamPump(io, feed)
    pump.caption = "GELLO SIGN CHECK"
    pump.start()
    if not feed.alive and not feed.sinks:
        print(f"(no video window: {feed.status} -- you'll need another way to watch the arm)")

    w = T.Teleop(args, io, gello)
    print("\nAligning to your GELLO's current pose, then waiting for the trigger squeeze"
          " (same as normal teleop) before the check starts...\n")
    while w.S["phase"] not in ("FOLLOWING", "DONE") and not w.done.is_set():
        time.sleep(0.1)
    if w.done.is_set():
        print(f"teleop ended before following started: {w.S.get('error') or 'aborted'}")
        return 1

    flips = []
    try:
        print("Following now. For each joint: move ONLY that one (either direction, any amount),")
        print("watch the video window, then answer whether it moved the way you'd expect.\n")
        for i, label in enumerate(JOINT_LABELS):
            input(f"[{i+1}/6] Move ONLY {label} now. Press ENTER once you've watched it move.")
            ans = input(f"    Did {label} in the video rotate the SAME direction as your hand? [Y/n]: ").strip().lower()
            if ans in ("n", "no"):
                flips.append(i)
                print(f"    -> flagged: {label} looks reversed")
            else:
                print("    -> ok")
    except KeyboardInterrupt:
        print("\naborted, no changes suggested")
    finally:
        w.abort.set()
        w.done.wait(10)
        pump.stop()
        feed.stop()
        try:
            gello.disconnect()
        except Exception:  # noqa: BLE001
            pass
        if not args.sim and io.robot is not None and io.robot.is_connected:
            io.robot.disconnect()
        if args.sim:
            io.disconnect()

    print("\n" + "=" * 60)
    if not flips:
        print("All 6 joints checked out -- no sign changes needed.")
        return 0
    print(f"Flip the sign for: {', '.join(JOINT_LABELS[i] for i in flips)}")
    print("Edit lerobot_ur10/src/lerobot/teleoperators/lerobot_teleoperator_gello/config_gello.py,")
    print("the `joint_signs` list (index = joint number, currently [1, 1, -1, 1, 1, 1]):")
    print("  for each flagged index, negate that entry (1 -> -1, or -1 -> 1).")
    print(f"  e.g. flagged indices {flips} -> that many entries change sign, others unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
