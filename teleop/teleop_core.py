#!/usr/bin/env python3
"""Shared teleop building blocks: robot I/O, the live camera window, and the
GELLO follow-loop worker.

Extracted from the openpi ur10_agent demo TUI (`demo_tui.py`) and trimmed
down to exactly what standalone teleop needs -- no policy-rollout / curses /
menu code. If you're comparing against that file: `RobotIO`, `VideoFeed`,
`CamPump`, `Teleop`, `_connect_robot`, `_connect_gello` and `add_teleop_args`
are unchanged in behavior, just lifted out on their own.
"""

from __future__ import annotations

import collections
import logging
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np


# ───────────────────────────── robot I/O ─────────────────────────────
class RobotIO:
    """Single choke point for robot access (one lock shared by teleop and the
    idle camera pump). robot=None (dry-run) returns fake observations."""

    def __init__(self, robot, h):
        self.robot, self.h = robot, h
        self.lock = threading.Lock()

    def obs(self):
        if self.robot is None:
            return self.h.fake_obs()
        with self.lock:
            return self.robot.get_observation()

    def send(self, action):
        if self.robot is not None:
            with self.lock:
                self.robot.send_action(action)

    def hold(self, last_gripper=0.0):
        if self.robot is None:
            return
        try:
            o = self.obs()
            a = {f"joint_{i}": float(o[f"joint_{i}"]) for i in range(6)}
            a["gripper"] = float(o.get("gripper", last_gripper))
            for _ in range(5):
                self.send(a)
                time.sleep(0.05)
        except Exception as e:  # noqa: BLE001
            logging.getLogger("teleop").warning("hold failed: %s", e)


# ───────────────────────────── camera window ─────────────────────────────
def _feed_main() -> None:
    """Child process: read length-prefixed JPEG frames on stdin, show them in
    a window (cv2 if it has a GUI backend, else tkinter) -- works over ssh -X."""
    os.environ.setdefault("QT_X11_NO_MITSHM", "1")
    import cv2

    def frames():
        inp = sys.stdin.buffer
        while True:
            hdr = inp.read(4)
            if len(hdr) < 4:
                return
            n = struct.unpack("<I", hdr)[0]
            data = b""
            while len(data) < n:
                chunk = inp.read(n - len(data))
                if not chunk:
                    return
                data += chunk
            yield cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    win = "UR10 cameras"
    try:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        for img in frames():
            cv2.imshow(win, img)
            cv2.waitKey(1)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return
        return
    except cv2.error as e:
        print(f"cv2 GUI unavailable ({str(e).splitlines()[0][:80]}); using tkinter", file=sys.stderr, flush=True)
    import tkinter as tk

    from PIL import Image, ImageTk
    root = tk.Tk()
    root.title(win)
    lbl = tk.Label(root)
    lbl.pack()
    closed = []
    root.protocol("WM_DELETE_WINDOW", lambda: closed.append(1))
    for img in frames():
        if closed:
            return
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        lbl.configure(image=photo)
        lbl.image = photo
        root.update()


class VideoFeed:
    """Streams a side-by-side camera view to a child window process."""

    def __init__(self, enabled, width, fps):
        self.width, self.min_dt = width, 1.0 / max(1.0, fps)
        self.proc = None
        self.sinks = []          # callables(jpeg_bytes): e.g. QuestServer.push_cam (headset panel)
        self.status = "off"
        self._t = 0.0
        self._lock = threading.Lock()
        self.errlog = Path(__file__).resolve().parent / "logs" / "teleop_video_feed.err"
        self.errlog.parent.mkdir(parents=True, exist_ok=True)
        if enabled:
            self.start()

    def start(self):
        if not os.environ.get("DISPLAY"):
            self.status = "no DISPLAY (ssh -X / -Y to the robot PC)"
            return
        try:
            err = open(self.errlog, "w")
            self.proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--feed"],
                                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
            self.status = "on"
        except Exception as e:  # noqa: BLE001
            self.proc, self.status = None, f"failed: {e}"

    @property
    def alive(self):
        if self.proc is not None and self.proc.poll() is not None:
            self.status = "closed" if self.proc.returncode == 0 else "died — see logs/teleop_video_feed.err"
            self.proc = None
        return self.proc is not None

    def stop(self):
        p, self.proc = self.proc, None
        if p is not None:
            try:
                p.stdin.close()
                p.wait(timeout=2)
            except Exception:  # noqa: BLE001
                p.kill()
        self.status = "off"

    def push(self, raw, caption):
        now = time.time()
        if now - self._t < self.min_dt or not (self.alive or self.sinks):
            return
        self._t = now
        import cv2
        try:
            w = self.width
            hh = w * 3 // 4
            tiles = [cv2.resize(np.asarray(raw[k], np.uint8), (w, hh), interpolation=cv2.INTER_AREA)
                     for k in ("cam_high", "cam_right_wrist")]
            img = cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR)
            band = np.zeros((26, img.shape[1], 3), np.uint8)
            cv2.putText(band, caption[: img.shape[1] // 8], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, "top", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(img, "wrist", (w + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            ok, jpg = cv2.imencode(".jpg", np.vstack([band, img]), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                jb = jpg.tobytes()
                for sink in self.sinks:
                    sink(jb)
                if self.alive:
                    with self._lock:
                        self.proc.stdin.write(struct.pack("<I", len(jb)) + jb)
                        self.proc.stdin.flush()
        except Exception:  # noqa: BLE001
            self.alive  # noqa: B018  (refresh status; child probably closed)


class CamPump(threading.Thread):
    """Idle camera preview (~10 Hz): reads obs and feeds the window/headset."""

    def __init__(self, io, feed):
        super().__init__(daemon=True)
        self.io, self.feed = io, feed
        self.active = True
        self.caption = "preview"
        self._stop_ev = threading.Event()

    def run(self):
        while not self._stop_ev.is_set():
            if self.active and (self.feed.alive or self.feed.sinks):
                try:
                    self.feed.push(self.io.obs(), self.caption + ("  [DRY-RUN: fake frames]" if self.io.robot is None else ""))
                except Exception:  # noqa: BLE001
                    time.sleep(0.5)
            time.sleep(0.1)

    def stop(self):
        self._stop_ev.set()


# ───────────────────────────── GELLO follow loop ─────────────────────────────
class Teleop:
    """GELLO leader arm -> robot follower. First ramps the robot to the leader's
    current pose (never a jump), then waits for the trigger to be squeezed,
    then follows at args.teleop_hz with a per-cycle joint step limit as a
    software safety. dry-run (no robot): display only."""

    kind = "teleop"

    def __init__(self, args, io, gello):
        self.args, self.io, self.gello = args, io, gello
        self.abort = threading.Event()
        self.start_now = threading.Event()   # skip the trigger gate
        self.done = threading.Event()
        self.S = dict(phase="ALIGNING", step=0, infer_ms=0.0, lat=collections.deque(maxlen=200),
                      joints=[float("nan")] * 6, gripper=0.0, t0=time.time(),
                      error="", aborted=False, clamped=0, hz=0.0)
        self.thread = threading.Thread(target=self._main, daemon=True)
        self.thread.start()

    def _leader(self):
        g = self.gello.get_action()
        return np.array([float(g[f"joint_{i}"]) for i in range(6)] + [float(np.clip(g["gripper"], 0.0, 1.0))])

    def _main(self):
        log = logging.getLogger("teleop")
        a, io = self.args, self.io
        try:
            lead = self._leader()
            last = lead.copy()
            if io.robot is not None:
                o = io.obs()
                cur = np.array([float(o[f"joint_{i}"]) for i in range(6)] + [float(o.get("gripper", 0.0))])
                gap = float(np.abs(cur[:6] - lead[:6]).max())
                log.info("teleop: leader/robot max joint gap %.1f°", np.rad2deg(gap))
                if gap > a.teleop_max_align_rad:
                    raise RuntimeError(
                        f"leader and robot differ by {np.rad2deg(gap):.0f}° on some joint "
                        f"(limit {np.rad2deg(a.teleop_max_align_rad):.0f}°) — put the GELLO near the robot's pose "
                        f"(or check its calibration), or raise --teleop-max-align-rad")
                period = max(0.008, float(getattr(io.robot, "servoj_t", 0.008)))
                n = max(20, int(max(gap, 0.0) / 0.005))
                for v in np.linspace(cur, lead, n):
                    if self.abort.is_set():
                        raise KeyboardInterrupt
                    io.send({f"joint_{i}": float(v[i]) for i in range(6)} | {"gripper": float(v[6])})
                    time.sleep(period)
                last = lead.copy()
            dt = 1.0 / max(1.0, a.teleop_hz)
            grip0 = last[6]
            self.S["phase"] = "WAIT TRIGGER"
            log.info("teleop: aligned. Squeeze the GELLO trigger fully (>= %.2f) to start following.", a.teleop_trigger)
            while not self.abort.is_set():
                lead = self._leader()
                self.S["joints"] = [float(np.rad2deg(v)) for v in lead[:6]]
                self.S["gripper"] = float(lead[6])
                if lead[6] >= a.teleop_trigger or self.start_now.is_set():
                    break
                io.send({f"joint_{i}": float(last[i]) for i in range(6)} | {"gripper": float(grip0)})
                time.sleep(dt)
            if self.abort.is_set():
                raise KeyboardInterrupt
            self.S["phase"] = "FOLLOWING"
            self.S["t0"] = time.time()
            log.info("teleop: trigger pressed -> following GELLO at %.0f Hz (step limit %.3f rad/cycle); "
                     "gripper follows the trigger from the first cycle (starts closed)", a.teleop_hz, a.teleop_max_step_rad)
            t_prev = time.perf_counter()
            while not self.abort.is_set():
                t0 = time.perf_counter()
                lead = self._leader()
                delta = lead[:6] - last[:6]
                lim = np.clip(delta, -a.teleop_max_step_rad, a.teleop_max_step_rad)
                if np.any(lim != delta):
                    self.S["clamped"] += 1
                cmd = np.concatenate([last[:6] + lim, [lead[6]]])
                io.send({f"joint_{i}": float(cmd[i]) for i in range(6)} | {"gripper": float(cmd[6])})
                last = cmd
                self.S["joints"] = [float(np.rad2deg(v)) for v in lead[:6]]
                self.S["gripper"] = float(lead[6])
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
            log.info("teleop stopped by presenter")
        except KeyboardInterrupt:
            self.S["aborted"] = True
        except BaseException as e:  # noqa: BLE001
            self.S["error"] = f"{type(e).__name__}: {e}"
            log.error("teleop error: %s", self.S["error"])
        finally:
            io.hold()
            self.S["phase"] = "DONE"
            self.S["t_end"] = time.time()
            self.done.set()


# ───────────────────────────── connection helpers ─────────────────────────────
def add_teleop_args(p) -> None:
    """Teleop / video / quest CLI args shared by teleop.py."""
    p.add_argument("--teleop-port", default="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO528D-if00-port0",
                   help="GELLO leader USB port")
    p.add_argument("--teleop-id", default="gello_teleop", help="GELLO calibration id (selects the calibration file)")
    p.add_argument("--teleop-hz", type=float, default=50.0, help="teleop loop rate (GELLO and Quest)")
    p.add_argument("--teleop-max-step-rad", type=float, default=0.03,
                   help="teleop: max joint change per cycle (software speed limit)")
    p.add_argument("--teleop-max-align-rad", type=float, default=1.5,
                   help="GELLO teleop: refuse to start if leader/robot differ by more than this on any joint")
    p.add_argument("--teleop-trigger", type=float, default=0.72,
                   help="GELLO teleop: gripper trigger value (0..1) that counts as squeezed to start following")
    p.add_argument("--no-quest", action="store_true", help="do not start the Quest teleop server")
    p.add_argument("--quest-port", type=int, default=8443, help="Quest teleop HTTPS/WSS port")
    p.add_argument("--quest-ip", default=None, help="IP the Quest should use (default: auto-detected LAN IP)")
    p.add_argument("--quest-pos-scale", type=float, default=0.8, help="Quest hand->robot motion scale")
    p.add_argument("--quest-zmin", type=float, default=0.05, help="Quest teleop: minimum TCP z (m, base frame)")
    p.add_argument("--quest-gripper", choices=["toggle", "analog"], default="toggle",
                   help="Quest right trigger: toggle open/closed, or analog 0..1")
    p.add_argument("--no-video", action="store_true", help="no camera window")
    p.add_argument("--video-width", type=int, default=400, help="per-camera width in the video window")
    p.add_argument("--video-fps", type=float, default=10.0, help="video window / headset panel refresh rate")


def _connect_robot(args, h):
    if args.dry_run:
        return None
    print("connecting to robot…", flush=True)
    robot = h.c._build_robot(args)
    robot.connect()
    try:
        robot.rtde_ctrl.stopScript()
        time.sleep(0.5)
        robot.rtde_ctrl.reuploadScript()
        time.sleep(0.5)
    except Exception:  # noqa: BLE001
        pass
    return robot


def _connect_gello(args):
    """Returns (gello|None, reason). Connecting may prompt on stdin for
    calibration if no calibration file exists yet for --teleop-id -- see
    teleop.py's own explicit calibration prompt for the normal (file exists)
    case."""
    try:
        from lerobot.teleoperators import make_teleoperator_from_config
        from lerobot.teleoperators.lerobot_teleoperator_gello import GelloConfig
    except Exception as e:  # noqa: BLE001
        return None, f"lerobot GELLO not importable here ({type(e).__name__}) — run on the robot PC with the lerobot env active"
    if not os.path.exists(args.teleop_port):
        return None, "leader USB port not found (--teleop-port)"
    try:
        print(f"connecting GELLO on {args.teleop_port}…", flush=True)
        t = make_teleoperator_from_config(GelloConfig(port=args.teleop_port, id=args.teleop_id))
        t.connect()
        return t, ""
    except Exception as e:  # noqa: BLE001
        return None, f"connect failed: {type(e).__name__}: {str(e)[:60]}"


def gello_calibration_path(teleop_id: str) -> Path:
    """Where lerobot stores this GELLO's calibration file (Teleoperator base
    class convention: <HF_LEROBOT_CALIBRATION>/teleoperators/gello/<id>.json)."""
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
    return HF_LEROBOT_CALIBRATION / "teleoperators" / "gello" / f"{teleop_id}.json"


if __name__ == "__main__":
    if "--feed" in sys.argv:
        _feed_main()
