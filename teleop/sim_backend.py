#!/usr/bin/env python3
"""Drive the MuJoCo sim (../ur10sim) with GELLO/Quest teleop instead of the
real robot -- the exact same teleop_core.Teleop / quest_teleop.QuestTeleop
loops, unchanged, just a different backend underneath.

Use this to practice or test teleop (GELLO calibration, Quest controls, IK
reachability, etc.) with zero real robot hardware -- only the input device
itself (a GELLO leader arm, or a Quest headset) needs to be real; there's no
UR10, no cameras, and no network connection to a robot at all.

  python teleop.py --method gello --sim
  python teleop.py --method quest --sim
"""
from __future__ import annotations

import concurrent.futures
import sys
import time
from pathlib import Path

import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1] / "ur10sim"
sys.path.insert(0, str(SIM_ROOT))

# NOTE: mujoco is imported lazily, inside the pool worker below, AFTER
# rl_env.setup_gl() sets MUJOCO_GL=egl -- mujoco picks its rendering backend
# at import time, so importing it before setup_gl runs would leave it stuck
# on the GLFW default, which fails with no $DISPLAY.
import rl_env as RL  # noqa: E402
import pincopen as PC  # noqa: E402

HOME_RAD = tuple(np.deg2rad(d) for d in (0.0, -90.0, 90.0, -90.0, -90.0, 90.0))
SUB = 33  # physics substeps per control step -- matches sim_rollout.py / replay_episode.py


def _get_ur10_ik():
    """Lazy import, same interface as robot/client_ur10.py's version: only
    needed for Quest's Cartesian hand-tracking IK."""
    import ur10_kinematics
    return ur10_kinematics


def smooth_move_home(io: "SimRobotIO", steps: int = 60) -> None:
    """Ease the sim arm to HOME_RAD. quest_teleop.py calls
    `self.h.c.smooth_move_home(io.robot)` -- `io.robot` in --sim mode IS the
    SimRobotIO instance (see its `.robot = self`), so this signature matches."""
    io._ease_to(np.array(HOME_RAD, float), steps)


class SimRobotIO:
    """Same duck-typed interface teleop_core.Teleop / quest_teleop.QuestTeleop
    already use against the real teleop_core.RobotIO: `.obs()`, `.send(action)`,
    `.hold()`, and a `.robot` attribute they check with `is None` / `getattr`.
    `.robot = self` (never None) so those checks behave as "connected", and
    `getattr(io.robot, "servoj_t", default)`-style calls just fall through to
    their default since a Scene has no such attribute.

    All actual MuJoCo/EGL calls run on ONE dedicated worker thread (via a
    single-worker ThreadPoolExecutor), never on the caller's thread. This
    matters because obs()/send() are called from at least two different
    Python threads in normal use (the idle CamPump preview AND the GELLO/Quest
    worker), and a mujoco.Renderer's EGL context is thread-affine -- handing
    it between threads without an explicit release fails with EGL_BAD_ACCESS.
    Funneling every call through one persistent thread sidesteps that
    entirely instead of trying to hand the context around.
    """

    def __init__(self, case: str = "single_circle", yaw: float = 0.0, width: int = 320, height: int = 240):
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim-gl")
        self.robot = self
        self.is_connected = True
        self._pool.submit(self._init_sim, case, yaw, width, height).result()

    def _init_sim(self, case: str, yaw: float, width: int, height: int) -> None:
        RL.setup_gl()
        import mujoco
        self.mujoco = mujoco
        self.env = RL.UR10PegEnv(width=width, height=height)
        self.env.reset(RL.make_spec(case, yaw))
        self.sc = self.env.sc
        self.W, self.H = width, height
        self._grip_cmd = float(self.sc.data.ctrl[0])
        self._held = False

    # -- the interface teleop_core.Teleop / quest_teleop.QuestTeleop actually use
    def obs(self) -> dict:
        return self._pool.submit(self._obs_impl).result()

    def send(self, action: dict) -> None:
        self._pool.submit(self._send_impl, action).result()

    def hold(self, last_gripper: float = 0.0) -> None:
        pass  # the kinematic arm holds its last commanded pose on its own; nothing to re-send

    def disconnect(self) -> None:
        self._pool.submit(self.env.close).result()
        self._pool.shutdown(wait=True)

    # -- internals (all run ONLY on the pool's single worker thread)
    def _obs_impl(self) -> dict:
        d = self.sc.data
        q = d.qpos[self.sc.arm_adr]
        cam = float(d.qpos[self.sc.cam_adr])
        frac = float(np.clip(RL.UR10PegEnv.frac_of_cam(cam), 0.0, 1.0))
        imgs = self.sc.render(self.W, self.H, cams=("top", "wrist"))
        return {f"joint_{i}": float(q[i]) for i in range(6)} | {
            "gripper": frac, "cam_high": imgs["top"], "cam_right_wrist": imgs["wrist"]}

    def _send_impl(self, action: dict) -> None:
        target = np.array([float(action[f"joint_{i}"]) for i in range(6)], float)
        grip_target = float(np.clip(action.get("gripper", 0.0), 0.0, 1.0))
        self._step_to(target, grip_target)

    def _step_to(self, q_target: np.ndarray, grip_target: float) -> None:
        sc, m, d = self.sc, self.sc.model, self.sc.data
        q0 = d.qpos[sc.arm_adr].copy()
        lo, hi = sc.ctrl_range
        want = grip_target > (0.3 if not self._held else 0.25)
        tgt_cam = float(np.clip(PC.cam_angle_for_frac(grip_target), lo, hi))
        for j in range(SUB):
            qa = q0 + (q_target - q0) * (j + 1) / SUB
            d.qvel[sc.arm_dof] = (qa - d.qpos[sc.arm_adr]) / m.opt.timestep
            d.qpos[sc.arm_adr] = qa
            self._grip_cmd += float(np.clip(tgt_cam - self._grip_cmd, -6e-3, 6e-3))
            d.ctrl[0] = self._grip_cmd
            self._held = sc.grasp_assist(want)
            self.mujoco.mj_step(m, d)

    def _ease_to(self, q_target: np.ndarray, steps: int) -> None:
        """Called via smooth_move_home(io, ...) -- io is this SimRobotIO, so
        run the ramp through the pool too rather than touching mujoco directly."""
        self._pool.submit(self._ease_to_impl, q_target, steps).result()

    def _ease_to_impl(self, q_target: np.ndarray, steps: int) -> None:
        q0 = self.sc.data.qpos[self.sc.arm_adr].copy()
        for k in range(1, steps + 1):
            self._step_to(q0 + (q_target - q0) * k / steps, 0.0)
            time.sleep(0.01)
