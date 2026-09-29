# teleop

Teleoperation module for `ur10-mujoco-sim`: drive the UR10 (real, or this
repo's MuJoCo sim) with a **GELLO** leader arm (a 1:1 joint-space
leader/follower rig) or a **Meta Quest 3** headset (Cartesian hand tracking
over WebXR), with a live camera view either way.

**Read [`GUIDE.md`](GUIDE.md) first** — it covers GELLO hardware assembly and
calibration, Meta Quest 3 setup and controls, driving the sim vs. the real
robot, networking, and safety. This README is just the quick-start.

## What's here

| File | What it does |
|---|---|
| `teleop.py` | The launcher — `--method gello` or `--method quest`, `--live` to actually drive the robot |
| `teleop_ctl.sh` | One-line wrapper around `teleop.py` |
| `teleop_core.py` | Robot I/O, the GELLO follow-loop, the camera window |
| `quest_teleop.py` | The Meta Quest HTTPS/WebXR server + hand-tracking worker |
| `quest/` | The WebXR page served to the headset (`index.html`) + auto-generated TLS certs |
| `robot/client_ur10.py` | Connects to the UR10 (via your lerobot fork), moves it home |
| `robot/ur10_kinematics.py` | Forward/inverse kinematics used by Quest's Cartesian tracking |

## Requirements

- **To drive the sim** (`--sim`): nothing beyond this repo's own
  `../requirements.txt` (mujoco etc.) — no real robot, no network. Needs
  mujoco with a working offscreen EGL/GL backend (see GUIDE.md if rendering
  fails).
- **To drive the real robot** (`--live`): a UR10 (CB3 or e-Series) reachable
  over the network, running with a lerobot-compatible robot driver (RTDE
  control) — see `GUIDE.md`.
- For GELLO (either mode — it's the input device, not the thing being
  driven): the GELLO leader hardware (Dynamixel servos) and this project's
  `lerobot_ur10` fork installed (`pip install -e .` from
  https://github.com/rudra-8000/lerobot_ur10 — **not** plain `pip install
  lerobot`; the GELLO teleoperator and UR10 driver classes live in this
  fork, not on PyPI). See GUIDE.md.
- For Quest (either mode): a Meta Quest 3 (or similar WebXR headset) on the
  same LAN as whatever machine is running `teleop.py`.
- `pip install -r requirements.txt`

## Quick start

```bash
# Try it risk-free first: GELLO or Quest driving the MuJoCo sim, no real robot
python teleop.py --method gello --sim
python teleop.py --method quest --sim

# GELLO, dry-run against the real robot (no robot moves, just prints/shows the leader arm)
python teleop.py --method gello

# GELLO, live on the real robot
python teleop.py --method gello --live

# Quest 3, live — prints a URL, open it on the headset's browser
python teleop.py --method quest --live
```

or via the wrapper:
```bash
bash teleop_ctl.sh gello --live
bash teleop_ctl.sh quest --live
```

Both open a camera window (top + wrist side by side) over X11 (`ssh -X`) and,
for Quest, a panel inside the headset. `SPACE` / `q` / `e` / `Esc` stops and
holds the robot in place; `Enter` / `s` skips the GELLO trigger gate.

## Calibration

Before GELLO teleop connects, `teleop.py` checks for an existing calibration
file and **asks whether you want to recalibrate** rather than silently
reusing it or silently skipping the question. See "Calibration" in
`GUIDE.md` for what calibration actually captures and why Quest doesn't need
the same kind of prompt.

## How `--sim` works

`sim_backend.SimRobotIO` implements the exact same `.obs()` / `.send()` /
`.hold()` interface as the real robot's `teleop_core.RobotIO`, backed by
`../ur10sim`'s `rl_env.UR10PegEnv` instead of an RTDE connection — so
`teleop_core.Teleop` and `quest_teleop.QuestTeleop` run completely unchanged
either way. All MuJoCo/EGL calls are funneled through one dedicated worker
thread (a mujoco offscreen renderer's GL context is thread-affine, and
`obs()` is called from more than one Python thread in normal use — see the
docstring in `sim_backend.py`).
