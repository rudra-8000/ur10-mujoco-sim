#!/usr/bin/env python3
"""UR10 connection helpers used by teleop.py: build the lerobot robot object,
move it home, and (lazily) hand back the IK/FK module used for Quest's
Cartesian hand tracking.

Trimmed from openpi's examples/ur10/client_ur10.py -- this repo only needs
the connect/home/IK pieces, not that script's policy-rollout/video-recording
code. Requires your lerobot UR10 fork to be installed and importable (see
GUIDE.md) -- this file has no fallback for "fork not installed" because
teleop cannot function without it.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ur10sim"))  # sibling package: kinematics is shared with the sim, not duplicated

logger = logging.getLogger(__name__)

HOME_DEG = (0.0, -90.0, 90.0, -90.0, -90.0, 90.0)
HOME_RAD = tuple(np.deg2rad(d) for d in HOME_DEG)

# Camera serials -- match your hardware (see GUIDE.md: find yours with
# `rs-enumerate-devices` from librealsense).
CAM_SERIALS = {
    "cam_high":        "204322061013",   # D415, top-down
    "cam_right_wrist": "923322071837",   # D435i, wrist
}


def _build_robot(args: argparse.Namespace):
    """Construct a UR10 robot object with cameras attached, via your lerobot
    UR10 fork. Adjust the import path / gripper port below to your setup."""
    from lerobot.cameras.configs import ColorMode
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.cameras import make_cameras_from_configs
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.lerobot_robot_ur10 import UR10Config  # adjust to your package layout

    cam_cfgs = {
        name: RealSenseCameraConfig(
            serial_number_or_name=serial,
            fps=getattr(args, "fps", 30),
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        )
        for name, serial in CAM_SERIALS.items()
    }
    gripper_port = getattr(args, "gripper_port",
                            "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO51RF-if00-port0")
    robot_cfg = UR10Config(ip=args.ur_ip, gripper_port=gripper_port)
    robot = make_robot_from_config(robot_cfg)
    robot.cameras = make_cameras_from_configs(cam_cfgs)
    return robot


def smooth_move_home(robot, home_rad=HOME_RAD, steps: int = 100, duration: float = 5.0) -> None:
    """Blocking moveJ to the home pose. Stops any lingering servoJ session first
    (servoJ and moveJ can't be interleaved safely on the same RTDE control session)."""
    try:
        robot.rtde_ctrl.servoStop()
        time.sleep(0.2)
    except Exception:  # noqa: BLE001
        pass
    logger.info("Moving to home: %s", [f"{np.rad2deg(v):.1f}°" for v in home_rad])
    robot.rtde_ctrl.moveJ(list(home_rad), speed=0.3, acceleration=0.3)
    logger.info("Home reached.")


def _get_ur10_ik():
    """Lazy import: only needed for Quest's Cartesian hand-tracking IK. Uses
    the sim's own ur10_kinematics module (../../ur10sim/) rather than a
    duplicate copy -- both real-robot and sim teleop share the same FK/IK."""
    import ur10_kinematics
    return ur10_kinematics
