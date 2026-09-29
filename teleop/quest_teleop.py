#!/usr/bin/env python3
"""Meta Quest 3 teleoperation backend for the UR10.

Port of https://github.com/rudra-8000/UR10-Meta-Quest-3-Teleop (teleop_server_3.py)
into this repo's control stack.  Differences from that standalone server:

  * The robot is driven through the SAME lerobot robot object (servoJ joint
    targets) as GELLO teleop -- no second RTDE connection.  Cartesian hand
    targets are converted with robot.ur10_kinematics.inverse_kinematics.
  * numpy only (no scipy, no aiohttp): HTTPS + WSS is served with `websockets`.
  * Software safety: per-cycle joint step limit, workspace clamp, stale-data hold.
  * Clutch semantics: every time tracking (re)starts, the hand->robot mapping is
    re-anchored, so pausing and moving your hand never makes the robot jump.
  * Camera frames are streamed to the headset (/cam_ws) and shown as a panel in
    the WebXR scene (?cam=x,y,z in the page URL moves it).
  * Right trigger controls the gripper (toggle, or analog with --quest-gripper analog).

Controls (same as the original repo, except where noted):
    Left  GRIP  (press)  IDLE -> CALIBRATED (captures reference);  again -> IDLE
    Right GRIP  (hold)   track the right controller (release = pause / clutch)
    Right TRIGGER        toggle gripper (open/closed)   [analog with --quest-gripper analog]
    Left  TRIGGER        move robot to home (blocking, smooth) and reset to IDLE

Run standalone via teleop.py --method quest. See GUIDE.md for headset setup.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import socket
import ssl
import subprocess
import threading
import time
from http import HTTPStatus
from pathlib import Path

import numpy as np

QUEST_DIR = Path(__file__).resolve().parent / "quest"

# WebXR is Y-up right-handed; the UR base is Z-up (same matrix as the original repo)
COORD_R = np.array([[-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0]])


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))   # UDP connect: sends nothing, just picks the route
            return s.getsockname()[0]
        except Exception:  # noqa: BLE001
            return "127.0.0.1"


def ensure_cert(ip: str) -> tuple[Path, Path]:
    d = QUEST_DIR / "certs"
    d.mkdir(parents=True, exist_ok=True)
    cert, key = d / f"cert_{ip}.pem", d / f"key_{ip}.pem"
    if not (cert.exists() and key.exists()):
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", str(key), "-out", str(cert),
             "-days", "365", "-nodes", "-subj", f"/CN={ip}",
             "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"],
            check=True, capture_output=True)
    return cert, key


# ───────────────────────────── math (numpy only) ─────────────────────────────
def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def q2rp(pos) -> np.ndarray:
    return COORD_R @ np.asarray(pos, float)


def q2rr(quat) -> np.ndarray:
    return COORD_R @ quat_to_mat(quat) @ COORD_R.T


def wrap_to(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """q + 2*pi*k so every joint is within pi of ref (keeps servoJ continuous)."""
    return ref + (q - ref + np.pi) % (2 * np.pi) - np.pi


# ───────────────────────────── server ─────────────────────────────
class QuestServer(threading.Thread):
    """HTTPS page + WSS: /ws (controllers in), /robot_ws (robot state out, 30 Hz),
    /cam_ws (JPEG camera frames out)."""

    def __init__(self, port: int = 8443, host: str = "0.0.0.0", ip: str | None = None):
        super().__init__(daemon=True, name="quest-server")
        self.port, self.host = port, host
        self.ip = ip or lan_ip()
        self.url = f"https://{self.ip}:{port}/"
        self.error = ""
        self.ready = threading.Event()
        self._lock = threading.Lock()
        self._ctrl = {"left": None, "right": None, "ts": 0.0}
        self._viz = {"joints": [0.0] * 6, "tcp": [0.0] * 6, "mode": "IDLE"}
        self._cam = (0, b"")
        self.n_ctrl = 0          # connected /ws clients
        self.n_cam = 0
        self._stop_ev = None
        self._loop = None

    # -- api used by the worker
    def snapshot(self):
        with self._lock:
            return dict(self._ctrl)

    def headset_fresh(self) -> bool:
        with self._lock:
            return (time.monotonic() - self._ctrl["ts"]) < 0.5 and self._ctrl["right"] is not None

    def set_state(self, joints, tcp, mode):
        with self._lock:
            self._viz = {"joints": [float(v) for v in joints], "tcp": [float(v) for v in tcp], "mode": mode}

    def push_cam(self, jpeg: bytes):
        with self._lock:
            self._cam = (self._cam[0] + 1, jpeg)

    def stop(self):
        if self._loop is not None and self._stop_ev is not None:
            self._loop.call_soon_threadsafe(self._stop_ev.set)

    # -- server internals
    def run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            logging.getLogger("quest").error("quest server stopped: %s", self.error)
            self.ready.set()

    async def _main(self):
        from websockets.asyncio.server import serve
        self._loop = asyncio.get_running_loop()
        self._stop_ev = asyncio.Event()
        cert, key = ensure_cert(self.ip)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        html = (QUEST_DIR / "index.html").read_text()

        def process_request(connection, request):
            path = request.path.split("?")[0]
            if path in ("/", "/index.html"):
                resp = connection.respond(HTTPStatus.OK, html)
                resp.headers["Content-Type"] = "text/html; charset=utf-8"
                return resp
            if path in ("/ws", "/robot_ws", "/cam_ws"):
                return None
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")

        async with serve(self._handler, self.host, self.port, ssl=ctx, process_request=process_request,
                         max_size=2 ** 22, ping_interval=10):
            logging.getLogger("quest").info("quest server up: %s", self.url)
            self.ready.set()
            await self._stop_ev.wait()

    async def _handler(self, ws):
        from websockets.exceptions import ConnectionClosed
        path = ws.request.path.split("?")[0]
        log = logging.getLogger("quest")
        if path == "/ws":
            self.n_ctrl += 1
            log.info("headset connected (%s)", ws.remote_address[0])
            try:
                async for msg in ws:
                    try:
                        d = json.loads(msg)
                        with self._lock:
                            self._ctrl = {"left": d.get("left"), "right": d.get("right"), "ts": time.monotonic()}
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                self.n_ctrl -= 1
                log.info("headset disconnected")
        elif path == "/robot_ws":
            try:
                while True:
                    with self._lock:
                        payload = json.dumps(self._viz)
                    await ws.send(payload)
                    await asyncio.sleep(1 / 30)
            except ConnectionClosed:
                pass
        elif path == "/cam_ws":
            self.n_cam += 1
            seq = 0
            try:
                while True:
                    with self._lock:
                        s, jpg = self._cam
                    if s != seq and jpg:
                        seq = s
                        await ws.send(jpg)
                    await asyncio.sleep(0.03)
            except ConnectionClosed:
                pass
            finally:
                self.n_cam -= 1


# ───────────────────────────── worker ─────────────────────────────
class QuestTeleop:
    """Quest hand -> robot, same worker interface as teleop_core.Teleop (S / abort / done)."""

    kind = "quest"

    def __init__(self, args, h, io, server: QuestServer):
        self.args, self.h, self.io, self.server = args, h, io, server
        self.abort = threading.Event()
        self.done = threading.Event()
        self.S = dict(phase="WAIT HEADSET", step=0, infer_ms=0.0, lat=collections.deque(maxlen=200),
                      joints=[float("nan")] * 6, gripper=0.0, t0=time.time(),
                      error="", aborted=False, ik_skips=0, clamped=0, hz=0.0, hint="")
        self.thread = threading.Thread(target=self._main, daemon=True)
        self.thread.start()

    def _read_q(self, fallback):
        if self.io.robot is None:
            return fallback
        o = self.io.obs()
        return np.array([float(o[f"joint_{i}"]) for i in range(6)])

    def _main(self):
        log = logging.getLogger("teleop")
        a, io, srv = self.args, self.io, self.server
        ik = self.h.c._get_ur10_ik()
        ik.TCP_OFFSET_MM = a.tcp_offset_mm
        ik.APPLY_TCP_OFFSET = a.tcp_offset_mm != 0
        tcp_mm = a.tcp_offset_mm
        C_MIN = np.array([-0.9, -0.9, a.quest_zmin])
        C_MAX = np.array([0.9, 0.9, 1.10])
        dt = 1.0 / max(1.0, a.teleop_hz)
        try:
            q_last = self._read_q(np.array(self.h.c.HOME_RAD, float))
            o = io.obs()
            grip_cmd = float(o.get("gripper", 0.0)) if io.robot is not None else 0.0
            grip_target = round(grip_cmd)          # 0 open / 1 closed, for toggle mode
            mode, calib = "IDLE", {}
            lg_prev = lt_prev = rg_prev = rt_prev = 0.0
            t_prev = time.perf_counter()
            last_ik_warn = 0.0
            self.S["t0"] = time.time()
            log.info("quest teleop: open %s on the Quest browser. Left grip = calibrate, right grip = track.", srv.url)
            while not self.abort.is_set():
                t0 = time.perf_counter()
                snap = srv.snapshot()
                left, right = snap["left"], snap["right"]
                fresh = (time.monotonic() - snap["ts"]) < 0.5 and left is not None and right is not None
                if not fresh:
                    if mode == "TRACKING":
                        mode = "CALIBRATED"
                        log.warning("quest data stale -> paused")
                    self.S["phase"] = "WAIT HEADSET"
                    self.S["hint"] = f"open {srv.url} in the Quest browser and enter AR/VR"
                else:
                    lg, lt = float(left.get("grip", 0)), float(left.get("trigger", 0))
                    rg, rt = float(right.get("grip", 0)), float(right.get("trigger", 0))
                    lg_edge, lt_edge = lg > 0.5 >= lg_prev, lt > 0.5 >= lt_prev
                    rt_edge = rt > 0.5 >= rt_prev
                    rg_on, rg_off = rg > 0.5, (rg <= 0.5 < rg_prev)
                    if lt_edge:                      # home
                        mode, calib = "IDLE", {}
                        log.info("quest: homing…")
                        self.S["phase"] = "HOMING"
                        try:
                            if io.robot is not None:
                                self.h.c.smooth_move_home(io.robot)
                                time.sleep(0.5)
                            q_last = self._read_q(np.array(self.h.c.HOME_RAD, float))
                        except Exception as e:  # noqa: BLE001
                            log.error("home failed: %s", e)
                    elif rt_edge and a.quest_gripper == "toggle":
                        grip_target = 0.0 if grip_target >= 0.5 else 1.0
                        log.info("quest: gripper -> %s", "CLOSED" if grip_target else "open")
                    elif lg_edge:
                        if mode == "IDLE":
                            mode = "CALIBRATED"
                            log.info("quest: calibrated (hold right grip to track)")
                        else:
                            mode, calib = "IDLE", {}
                            log.info("quest: -> IDLE (reset)")
                    elif rg_off and mode == "TRACKING":
                        mode = "CALIBRATED"
                        log.info("quest: tracking paused")
                    elif rg_on and mode == "CALIBRATED":
                        # (re)anchor the mapping at the CURRENT hand + robot pose (clutch)
                        q_now = self._read_q(q_last)
                        pos, rv = ik.forward_kinematics(q_now)
                        calib = dict(r_pos=q2rp(right["pos"]), r_rot=q2rr(right["quat"]),
                                     home_pos=np.array(pos), home_rot=ik._rotvec_to_rotmat(np.array(rv)))
                        mode = "TRACKING"
                        log.info("quest: tracking started")
                    if a.quest_gripper == "analog":
                        grip_target = float(np.clip(rt, 0.0, 1.0))
                    lg_prev, lt_prev, rg_prev, rt_prev = lg, lt, rg, rt
                    self.S["phase"] = {"IDLE": "QUEST IDLE", "CALIBRATED": "QUEST READY",
                                       "TRACKING": "QUEST TRACKING"}[mode]
                    self.S["hint"] = {"IDLE": "left grip = calibrate   |   left trigger = home",
                                      "CALIBRATED": "hold RIGHT GRIP to track   |   left grip = reset",
                                      "TRACKING": "tracking — release right grip to pause   |   right trigger = gripper"}[mode]

                # ---- command
                q_cmd = q_last.copy()
                if mode == "TRACKING" and fresh:
                    dp = (q2rp(right["pos"]) - calib["r_pos"]) * a.quest_pos_scale
                    dr = q2rr(right["quat"]) @ calib["r_rot"].T
                    tpos = np.clip(calib["home_pos"] + dp, C_MIN, C_MAX)
                    trot = ik._rotmat_to_rotvec(dr @ calib["home_rot"])
                    try:
                        q_t, _ = ik.inverse_kinematics(tpos, trot, q_ref=q_last, tcp_offset_mm=tcp_mm)
                        q_t = wrap_to(np.asarray(q_t, float), q_last)
                        d = q_t - q_last
                        lim = np.clip(d, -a.teleop_max_step_rad, a.teleop_max_step_rad)
                        if np.any(lim != d):
                            self.S["clamped"] += 1
                        q_cmd = q_last + lim
                    except ValueError as e:
                        self.S["ik_skips"] += 1
                        if time.time() - last_ik_warn > 2:
                            log.warning("IK failed (target unreachable), holding: %s", str(e)[:70])
                            last_ik_warn = time.time()
                grip_cmd += float(np.clip(grip_target - grip_cmd, -0.08, 0.08))
                io.send({f"joint_{i}": float(q_cmd[i]) for i in range(6)} | {"gripper": float(grip_cmd)})
                q_last = q_cmd
                pos, rv = ik.forward_kinematics(q_last)
                srv.set_state(q_last, list(pos) + list(rv), mode)
                self.S["joints"] = [float(np.rad2deg(v)) for v in q_last]
                self.S["gripper"] = grip_cmd
                self.S["step"] += 1
                now = time.perf_counter()
                self.S["lat"].append((now - t_prev) * 1000)
                self.S["hz"] = 1.0 / max(1e-6, now - t_prev)
                self.S["infer_ms"] = (now - t_prev) * 1000
                t_prev = now
                rest = dt - (time.perf_counter() - t0)
                if rest > 0:
                    time.sleep(rest)
            self.S["aborted"] = True
            log.info("quest teleop stopped by presenter")
        except BaseException as e:  # noqa: BLE001
            self.S["error"] = f"{type(e).__name__}: {e}"
            log.error("quest teleop error: %s", self.S["error"])
        finally:
            io.hold()
            self.S["phase"] = "DONE"
            self.S["t_end"] = time.time()
            self.done.set()


def start_server(args):
    """Returns (server|None, reason)."""
    if getattr(args, "no_quest", False):
        return None, "disabled (--no-quest)"
    if not (QUEST_DIR / "index.html").exists():
        return None, f"missing {QUEST_DIR}/index.html"
    try:
        srv = QuestServer(port=args.quest_port, ip=getattr(args, "quest_ip", None) or None)
        srv.start()
        srv.ready.wait(10)
        if srv.error:
            return None, srv.error[:70]
        return srv, ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:60]}"
