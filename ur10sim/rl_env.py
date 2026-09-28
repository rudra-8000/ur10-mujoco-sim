#!/usr/bin/env python3
"""Single-environment RL task for the UR10 peg-in-hole scene (no RLinf dependency; rlinf/envs/sim/ur10mj wraps many of these in subprocesses).

One episode = one scene: {single circle hole | single square hole | both holes (normal / swapped)} x peg yaw {0, 180}, prompt = 'place the peg into the
circular|square hole' (the prompts of the combined multihole dataset).  Control at 30 Hz: the policy gives absolute joint targets (rad) + a gripper
value (0 open .. ~0.83 squeezing); the kinematic arm tracks them with a speed limit and a table floor, the gripper is the real MuJoCo actuator
(commanded value + squeeze, slew-limited) and the peg is welded by Scene.grasp_assist when both pads touch it.
Reward (sparse with a little shaping, each item once per episode): +0.2 peg lifted >= 8 cm while held, +0.2 correct end down within 4 cm (xy) of the
correct hole below 12 cm, +1.0 success (correct end >= 12 mm inside the correct hole, within 10 mm of its axis, vertical within 25 deg, for 5 steps; held or released); wrong hole insertion ends the episode (0).
"""
import copy
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROMPTS = {"circle": "place the peg into the circular hole", "square": "place the peg into the square hole"}
CASES = ("single_circle", "single_square", "multi_circle", "multi_square")      # (layout, target hole)
HOLE_XY = {"single_circle": (0.007, -0.872), "single_square": (0.010, -0.870)}
MULTI_XY = {"circle": (0.07, -0.872), "square": (-0.07, -0.872)}                # normal layout; 'flip' swaps the x signs


def setup_gl(device_id=None):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    vend = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if vend.exists():
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(vend))
    if device_id is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
    sys.unraisablehook = lambda *a, **k: None


def make_spec(case: str, yaw: float, flip: bool = False, jitter=(0.0, 0.0, 0.0), rng=None) -> dict:
    """scene description; jitter = (hole xy, peg xy, peg yaw deg) uniform half-ranges (m, m, deg)"""
    rng = rng or np.random.default_rng()
    layout, target = case.split("_")
    u = lambda w: float(rng.uniform(-w, w)) if w else 0.0
    holes = {}
    if layout == "single":
        x, y = HOLE_XY[case]
        holes[target] = (x + u(jitter[0]), y + u(jitter[0]))
    else:
        for h, (x, y) in MULTI_XY.items():
            holes[h] = ((-x if flip else x) + u(jitter[0]), y + u(jitter[0]))
    return dict(case=case, target=target, holes=holes, peg_xy=(0.01 + u(jitter[1]), -0.662 + u(jitter[1])), peg_yaw=float(yaw) + u(jitter[2]), prompt=PROMPTS[target], flip=flip)


class UR10PegEnv:
    def __init__(self, config_path=None, width=640, height=480, max_steps=700, max_joint_speed=1.5, squeeze=0.05, floor_z=0.0, settle_steps=400):
        import scene_view as SV
        import build_scene as B
        import pincopen as PC
        import mujoco
        self.mujoco, self.B, self.PC = mujoco, B, PC
        self.base_cfg = B.load_config(config_path)
        self.W, self.H, self.max_steps = width, height, max_steps
        self.max_joint_speed, self.squeeze, self.floor_z, self.settle_steps = max_joint_speed, squeeze, floor_z, settle_steps
        self.sc = None
        self.spec = None

    # ------------------------------------------------------------------ helpers
    def _cfg_for(self, spec):
        cfg = copy.deepcopy(self.base_cfg)
        for h in ("circle", "square"):
            cfg["holes"][h]["enabled"] = h in spec["holes"]
            if h in spec["holes"]:
                cfg["holes"][h].update(x=spec["holes"][h][0], y=spec["holes"][h][1], yaw_deg=0.0)
        cfg["peg"].update(enabled=True, mode="table", x=spec["peg_xy"][0], y=spec["peg_xy"][1], yaw_deg=spec["peg_yaw"])
        cfg["robot"].update(source="joints", joints_deg=[0, -90, 90, -90, -90, 90], gripper_frac=0.0)
        cfg["render"]["settle_steps"] = self.settle_steps
        return cfg

    def close(self):
        if self.sc is not None:
            self.sc.close()
            self.sc = None

    @staticmethod
    def frac_of_cam(cam):
        return float((np.radians(54.3) - cam) / np.radians(145.0))

    def _tcp_z(self, q):
        s = self._scratch
        s.qpos[:] = self.sc.data.qpos
        s.qpos[self.sc.arm_adr] = q
        self.mujoco.mj_kinematics(self.sc.model, s)
        return float(s.site_xpos[self._tcp][2])

    # ------------------------------------------------------------------ API
    def reset(self, spec: dict):
        B, mj = self.B, self.mujoco
        cfg = self._cfg_for(spec)
        if self.sc is None or self.sc.key != B.scene_key(cfg):
            self.close()
            self.sc = B.Scene(cfg)
        else:
            self.sc.cfg = cfg
            self.sc.apply_state()
        sc = self.sc
        m, d = sc.model, sc.data
        self.spec = spec
        self._scratch = mj.MjData(m)
        self._tcp = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "tcp")
        self._peg = sc.bid("peg")
        self._hole = {h: sc.bid("hole_" + h) for h in spec["holes"]}
        self.q = d.qpos[sc.arm_adr].copy()
        self.cmd_cam = float(d.ctrl[0])
        self.held = False
        self.t = 0
        self.stage = dict(lift=False, near=False, success=False, wrong=False)
        self.finished = False
        self._last = None
        self.succ_steps = 0
        self.floor_hits = 0
        self.max_lift = 0.0
        return self.observation()

    def observation(self):
        sc, d = self.sc, self.sc.data
        from PIL import Image
        im = sc.render(self.W, self.H, cams=("top", "wrist"))
        small = lambda x: np.asarray(Image.fromarray(x).resize((320, 240), Image.BOX))
        g = float(np.clip(self.frac_of_cam(float(d.qpos[sc.cam_adr])), 0.0, 1.0))
        return dict(top=small(im["top"]), wrist=small(im["wrist"]), state=np.array([*self.q, g], np.float32), prompt=self.spec["prompt"])

    def step(self, action):
        """action: (7,) absolute joint targets [rad] + gripper value.  returns obs, reward, terminated, truncated, info"""
        if self.finished:                                                       # episode over: keep returning the final observation, no reward
            obs, info, te, tr = self._last
            return obs, 0.0, te, tr, info
        sc, mj, d, m, PC = self.sc, self.mujoco, self.sc.data, self.sc.model, self.PC
        target, grip = np.asarray(action[:6], float), float(action[6])
        want = grip > (0.25 if self.held else 0.3)
        lo, hi = sc.ctrl_range
        tgt_cam = float(np.clip(PC.cam_angle_for_frac(grip + (self.squeeze if grip > 0.3 else 0.0)), lo, hi))
        dq = self.max_joint_speed / 30.0
        q_new = self.q + np.clip(target - self.q, -dq, dq)
        if self._tcp_z(q_new) < self.floor_z:                                   # the table stops the robot
            a, b = 0.0, 1.0
            for _ in range(12):
                mid = 0.5 * (a + b)
                a, b = (mid, b) if self._tcp_z(self.q + (q_new - self.q) * mid) >= self.floor_z else (a, mid)
            q_new = self.q + (q_new - self.q) * a
            self.floor_hits += 1
        for j in range(33):
            qa = self.q + (q_new - self.q) * (j + 1) / 33
            d.qvel[sc.arm_dof] = (qa - d.qpos[sc.arm_adr]) / m.opt.timestep
            d.qpos[sc.arm_adr] = qa
            self.cmd_cam += float(np.clip(tgt_cam - self.cmd_cam, -6e-3, 6e-3))
            d.ctrl[0] = self.cmd_cam
            self.held = sc.grasp_assist(want)
            mj.mj_step(m, d)
        self.q = q_new
        self.t += 1
        if not (np.isfinite(d.qpos).all() and np.abs(d.xpos[self._peg]).max() < 3.0):        # unstable contact: end the episode, reward 0
            self.finished = True
            self._last = (self.observation_safe(), dict(success=False, lifted=self.stage["lift"], near=self.stage["near"], wrong_hole=False, sim_error=True, held=False, floor_hits=self.floor_hits, t=self.t, tip_dxy=9.0, tip_height=9.0, peg_axis_z=0.0), True, False)
            return self._last[0], 0.0, True, False, dict(success=False, lifted=False, near=False, wrong_hole=False, sim_error=True, held=False, floor_hits=self.floor_hits, t=self.t, tip_dxy=9.0, tip_height=9.0, peg_axis_z=0.0)
        reward, terminated, info = self._score()
        truncated = self.t >= self.max_steps
        info.update(held=self.held, floor_hits=self.floor_hits, t=self.t, sim_error=False)
        obs = self.observation()
        if terminated or truncated:
            self.finished = True
            self._last = (obs, dict(info), terminated, truncated)
        return obs, reward, terminated, truncated, info

    def observation_safe(self):
        z = np.zeros((240, 320, 3), np.uint8)
        return dict(top=z, wrist=z, state=np.zeros(7, np.float32), prompt=self.spec["prompt"])

    def _score(self):
        d, sp = self.sc.data, self.spec
        target, other = sp["target"], ("square" if sp["target"] == "circle" else "circle")
        Rp = d.xmat[self._peg].reshape(3, 3)
        ax, p = Rp[:, 2], d.xpos[self._peg]                                     # +z = round end, -z = square end
        need_round = target == "circle"
        tip = p + (0.097 if need_round else -0.1035) * ax                       # end that has to go into the hole
        tip_down = (ax[2] < -0.9) if need_round else (ax[2] > 0.9)
        r = 0.0
        tz0 = self.B.load_config  # noqa: F841 (keep import used)
        floor_z = float(self.base_cfg["table"]["z"])
        lift = float(p[2] - 0.023)
        self.max_lift = max(self.max_lift, lift)
        if self.held and lift > 0.08 and not self.stage["lift"]:
            self.stage["lift"] = True; r += 0.2
        hp = d.xpos[self._hole[target]]
        dxy = float(np.linalg.norm((tip - hp)[:2]))
        if tip_down and dxy < 0.04 and (tip[2] - floor_z) < 0.12 and not self.stage["near"]:
            self.stage["near"] = True; r += 0.2
        inside = tip_down and dxy < 0.01 and (tip[2] - floor_z) < 0.03 and abs(ax[2]) > 0.9      # tip >= 12 mm below the block top, near the axis
        self.succ_steps = self.succ_steps + 1 if inside else 0
        terminated = False
        if other in self._hole:                                                # wrong hole: the OTHER end of the peg inside the other hole
            ho = d.xpos[self._hole[other]]
            wrong_end = p + (0.097 if not need_round else -0.1035) * ax
            if float(np.linalg.norm((wrong_end - ho)[:2])) < 0.008 and (wrong_end[2] - floor_z) < 0.04 and not self.held:
                self.stage["wrong"] = True; terminated = True
        if self.succ_steps >= 5 and not self.stage["success"]:
            self.stage["success"] = True; r += 1.0; terminated = True
        info = dict(success=self.stage["success"], lifted=self.stage["lift"], near=self.stage["near"], wrong_hole=self.stage["wrong"], tip_dxy=dxy,
                    tip_height=float(tip[2] - floor_z), peg_axis_z=float(ax[2]))
        return r, terminated, info


if __name__ == "__main__":                                                     # smoke test: random spec, zero action
    import time
    setup_gl()
    env = UR10PegEnv()
    for case in CASES:
        obs = env.reset(make_spec(case, 0.0))
        t0 = time.time()
        for _ in range(10):
            obs, r, te, tr, info = env.step(np.array([*obs["state"][:6], 0.0]))
        print(case, obs["prompt"], "state", obs["state"].round(2), "img", obs["top"].shape, "10 steps %.2fs" % (time.time() - t0), info)
