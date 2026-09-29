# UR10 Teleop Guide — GELLO & Meta Quest 3

This covers setting up either teleop method end to end: hardware, first
connection, calibration, networking, controls, and troubleshooting. Read
`README.md` first for the file layout and quick-start commands.

**Safety, every time, both methods:** the robot moves under motor power when
you run `--live`. Clear the workspace, know where the e-stop is, and know the
abort key (`SPACE`/`q`/`e`/`Esc` — it stops and holds in place, doesn't
power off) before your first live session with any new hardware or after any
change to calibration/config. `--sim` has none of these risks -- it only
ever moves the simulated arm in `../ur10sim`.

---

## 0. Driving the sim instead of the real robot (`--sim`)

Before touching the real robot with a new GELLO calibration or a first Quest
session, try it against the MuJoCo sim:
```bash
python teleop.py --method gello --sim
python teleop.py --method quest --sim --sim-case single_square
```
This uses `sim_backend.SimRobotIO`, which implements the same `.obs()` /
`.send()` / `.hold()` interface `teleop_core.Teleop` and
`quest_teleop.QuestTeleop` already use for the real robot, backed by
`../ur10sim`'s `rl_env.UR10PegEnv` instead of an RTDE connection — so the
GELLO/Quest control code itself is completely unchanged; only what's on the
other end of `.send()` differs. `--sim-case` picks which scene
(`single_circle` / `single_square` / `multi_circle` / `multi_square`,
default `single_circle`); `--sim-yaw` sets the peg's start yaw in degrees.

The GELLO leader arm or Quest headset is still **real hardware** either
way — only the robot side is simulated. GELLO's calibration prompt (section
1.4) behaves identically in `--sim`.

**If scene construction or rendering fails** (`body mass is too small,
cannot compute center of mass`, or an EGL/GL error on the first render):
you likely have an older `mujoco` package than this was built against, or
one without a working offscreen EGL backend. Check `python -c "import
mujoco; print(mujoco.__version__)"` — if you have more than one Python
environment on this machine, make sure you're running `teleop.py` with the
same one you use for the rest of `ur10-mujoco-sim`, not a stray older
`mujoco` install elsewhere on `$PATH`.

---

## 1. GELLO leader arm

GELLO is a 3D-printed, Dynamixel-servo leader arm built to be a joint-space
mirror of the follower robot — you physically move the leader, the follower
matches it 1:1. Hardware design and BOM: <https://wuphilipp.github.io/gello_site/>.
This repo only covers the *software* side (reading it as a lerobot
teleoperator); build/assemble the arm itself from that site.

### 1.0 Software install (do this first, on any machine, before --sim or --live)

Reading GELLO is real hardware I/O over USB/Dynamixel — this is needed no
matter what's on the *other* end (the real robot or the sim), so it's not
something `--sim` lets you skip.

**`pip install lerobot` (plain, from PyPI) is not enough.** The GELLO
teleoperator class (`lerobot.teleoperators.lerobot_teleoperator_gello`) and,
for `--live`, the UR10 robot driver (`lerobot.robots.lerobot_robot_ur10`)
are this project's own fork, not part of vanilla lerobot:

```bash
git clone https://github.com/rudra-8000/lerobot_ur10.git
pip install -e ./lerobot_ur10
```
Do this inside the same Python environment you run `teleop.py` from. Skip it
entirely only if you're doing `--method quest` — the Quest headset talks
WebSocket/JSON directly and never touches lerobot at all.

### 1.1 Wiring

- 6 arm joints + 1 gripper = 7 Dynamixel servos (XL330-M288 for the joints,
  XL330-M077 for the gripper in this project's build), daisy-chained on one
  TTL bus.
- A U2D2 (or FTDI) USB-to-TTL adapter connects the chain to your PC.
- Each servo needs a unique Dynamixel ID (1–6 for the arm joints, 7 for the
  gripper) — see 1.2 if you're setting up fresh hardware or replaced a servo.

### 1.2 First-time motor ID setup (skip if your GELLO already has IDs set)

Dynamixel servos ship with a default ID (usually 1) and need to be connected
**one at a time** to get unique IDs. If you're using `lerobot`'s GELLO
teleoperator class directly, its `setup_motors()` method walks you through
this interactively (connect the controller board to one motor at a time,
press Enter, it sets and verifies the ID) — check your lerobot version's
GELLO teleoperator for the exact entry point, since this method isn't always
exposed on the CLI.

### 1.3 Finding the USB port

```bash
ls -la /dev/serial/by-id/
```
Look for something like `usb-FTDI_USB__-__Serial_Converter_FTAO528D-if00-port0`
— use the **by-id** path, not `/dev/ttyUSB0` (that number can change across
reboots or if you plug in another USB-serial device). Pass it with
`--teleop-port`.

**Port permissions:** if connecting fails with a permission error, add your
user to the `dialout` group (`sudo usermod -aG dialout $USER`, then log out
and back in) rather than running as root — a fresh udev rule isn't usually
needed for a standard FTDI/U2D2 adapter, but if `dialout` alone doesn't fix
it, check `dmesg` for what device node actually got created.

### 1.4 Calibration

Calibration captures **where "home pose" is on your specific GELLO's
servos** (raw encoder counts don't mean anything until you've told it what
pose they correspond to) and the gripper's open/closed encoder range. It's
saved once per `--teleop-id` (default `gello_teleop`) to:
```
~/.cache/huggingface/lerobot/calibration/teleoperators/gello/<teleop-id>.json
```

**What `teleop.py` does before connecting:**
- No calibration file for this id yet → nothing to ask; you'll be walked
  through it right after connecting (move the arm to the home pose shown on
  screen, press Enter).
- A calibration file already exists → **you're asked**:
  ```
  Existing GELLO calibration found (<path>).
  Recalibrate now? [y/N]:
  ```
  Answer `N` (or just Enter) to reuse it — this is what you want on every
  normal day. Answer `y` if the arm was reassembled, a servo was swapped, or
  the joints feel offset from what the robot is actually doing.

Flags for scripting (e.g. an unattended launch): `--force-calibrate` always
recalibrates without asking; `--no-calibrate-prompt` never asks (combine
with `--force-calibrate` to force silently, or without it to always reuse
silently).

**How to tell if you need to recalibrate without guessing:** on connect (or
after a `--live` alignment step), `teleop.py` logs the max joint gap between
the leader and the follower robot's current pose. A consistently large gap
across sessions (not just "the leader happened to be in a different pose
when you plugged it in") points at stale calibration, not user error.

### 1.5 If a joint seems reversed

`joint_signs` in `config_gello.py` maps GELLO's raw encoder direction to the
follower's joint convention, per joint. It's tuned for the reference GELLO
build; a physically different assembly (motor mounted the other way on some
joint, etc.) can need a different sign on exactly that joint. Don't guess --
`check_gello_signs.py` isolates one joint at a time and asks a plain
yes/no, then tells you exactly which sign(s) to flip:
```bash
python check_gello_signs.py --sim     # or --live
```

### 1.6 Running

```bash
python teleop.py --method gello              # dry-run
python teleop.py --method gello --live       # real robot
```
Sequence: robot eases smoothly to the leader's current pose (refuses if any
joint differs by more than `--teleop-max-align-rad`, default ~86°, so it
never lurches to catch up) → waits for you to squeeze the gripper trigger
past `--teleop-trigger` (default 0.72) → starts following at `--teleop-hz`
(default 50 Hz), with each cycle's joint motion capped at
`--teleop-max-step-rad` (default 0.03 rad) as a software speed limit
independent of whatever the leader does. The gripper follows the trigger
from the first following cycle, so teleop starts **gripper closed** (trigger
fully pressed) and opens as you release it.

---

## 2. Meta Quest 3

This drives the robot in **Cartesian** space (your hand's position/rotation
in the real world maps to the gripper's position/rotation), not joint space
— fundamentally different from GELLO, which is why the setup and controls
look different.

### 2.1 What you need

- A Meta Quest 3 (or compatible WebXR headset) with the Quest Browser.
- **No APK / SideQuest / developer mode needed** — this is a plain WebXR web
  page, not a native app. You just open a URL in the headset's browser and
  grant it VR permission when prompted.
- The headset on the **same LAN** as the machine running `teleop.py` (the
  "robot PC"), and **internet access** on top of that — the page loads
  three.js from `cdn.jsdelivr.net`, so a fully air-gapped LAN won't work
  unless you vendor that script locally (see 2.5).

### 2.2 Starting the server and connecting

```bash
python teleop.py --method quest --live
```
This prints something like:
```
Open this on the Quest browser (accept the self-signed cert): https://192.168.1.42:8443/
```
On the headset: open that URL in the Quest Browser, accept the "connection
is not private" warning (it's a self-signed cert generated locally into
`quest/certs/` — expected, not a real security issue on your own LAN), then
tap **Enter VR** on the page and grant the permission prompt.

If the auto-detected IP is wrong for your network (multi-homed robot PC,
VPN interface, etc.), override it: `--quest-ip <the-robot-PCs-actual-LAN-IP>`.
Find candidates with `hostname -I` on the robot PC and pick the one the
headset can actually reach (often not the first one listed).

### 2.3 Controls

| Action | Effect |
|---|---|
| Left **grip** (press) | IDLE → CALIBRATED (captures a reference pose). Press again → back to IDLE. |
| Right **grip** (hold) | Track the right controller. Release = pause ("clutch"). |
| Right **trigger** | Toggle gripper open/closed (or analog 0–1 with `--quest-gripper analog`) |
| Left **trigger** | Move the robot to its home pose (blocking, smooth), reset to IDLE |

**The clutch matters:** every time you re-press right grip to resume
tracking, the hand→robot mapping is re-anchored at your *current* hand pose
and the robot's *current* pose — so releasing the grip to reposition your
arm, then squeezing again, never makes the robot jump. This is the direct
equivalent of picking up a mouse and putting it back down.

### 2.4 "Calibration" here is different from GELLO's

Quest has **no calibration file** — there's nothing persistent to ask about
before starting, which is why `teleop.py` doesn't prompt for Quest the way
it does for GELLO. The left-grip press *is* the calibration: it's a live,
repeatable action you do at the start of (or any time during) a session, not
a one-time hardware setup step. Do it once after opening the page and again
any time the mapping feels off (e.g. after taking the headset off and back
on with your arm in a different spot).

### 2.5 Safety / tuning knobs

- `--quest-pos-scale` (default 0.8): hand-motion → robot-motion scale.
  **Start much lower (e.g. 0.3) the first time** you try this on real
  hardware — you can always increase it once you trust the mapping.
- `--quest-zmin` (default 0.05 m): workspace floor clamp in the robot's base
  frame — the commanded target's Z is never allowed below this, regardless
  of where your hand goes.
- `--teleop-max-step-rad` (default 0.03): same per-cycle joint speed limit
  as GELLO, applied to the IK solution.
- Stale headset data (dropped WebSocket, headset put down) automatically
  pauses tracking rather than freezing the last command in place forever.
- No internet on your LAN: download three.js yourself and edit the
  `<script src=...>` line in `quest/index.html` to point at a local copy
  instead of the CDN.

---

## 3. Networking (both methods)

- GELLO only needs a USB connection to the robot PC — no network involved
  for the leader itself.
- Quest needs the robot PC reachable on the LAN at whatever port
  `--quest-port` uses (default 8443). If nothing loads on the headset, check
  a firewall isn't blocking that port on the robot PC first.
- Both methods drive the robot through the **same** RTDE connection as
  everything else in this project (policy rollouts, recording) — never run
  teleop and a policy client against the robot at the same time; only one
  RTDE control session can own the arm.

---

## 4. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `GELLO unavailable: leader USB port not found` | Wrong `--teleop-port`, or the U2D2/FTDI adapter isn't plugged in / powered. Check `ls /dev/serial/by-id/`. |
| `lerobot GELLO not importable here` / `ModuleNotFoundError: No module named 'lerobot'` | The `lerobot_ur10` fork isn't installed in this Python environment (see 1.0) — plain `pip install lerobot` does not provide it. This applies to `--sim` too: GELLO input reading needs it regardless of what's on the robot side. |
| Leader/robot pose mismatch refusal on connect | Either genuinely misaligned (raise `--teleop-max-align-rad` only if you understand why the gap is real) or a stale calibration — see 1.4. |
| `Quest server unavailable: missing quest/index.html` | You're running `teleop.py` from somewhere other than this repo's root, or the `quest/` folder didn't come along — check your working directory. |
| Quest page loads but nothing tracks | Confirm you tapped **Enter VR** (the page does nothing outside an active WebXR session) and that left/right grip presses show up in the on-page status log. |
| `IK failed (target unreachable), holding` (repeated) | Your hand moved somewhere outside the arm's reachable workspace for its current joint configuration — bring it back toward center; this is a safety skip, not a crash. |
| Video window never appears | `--no-video` wasn't passed but `$DISPLAY` isn't set — teleop still works, you just don't get the camera window. Use `ssh -X`/`-Y` to the robot PC if you want it remotely. |
| `--sim`: `body mass is too small, cannot compute center of mass` | Wrist-mount/D435i visual meshes in `../ur10sim/build_scene.py` are declared zero-mass; some `mujoco` versions refuse to compile a body whose every geom is exactly mass 0. Already patched here (`mass="1e-6"`, physically negligible) — if you regenerate `build_scene.py` from a newer copy of the sim and this comes back, re-apply the same one-line change. |
| `--sim`: `EGLError ... EGL_BAD_ACCESS` | A mujoco offscreen renderer's GL context is thread-affine; this happens if something outside `sim_backend.py` calls `SimRobotIO`'s internals directly instead of going through `.obs()`/`.send()` (which funnel everything through one dedicated worker thread on purpose — see `sim_backend.py`'s docstring). |
