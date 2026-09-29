#!/usr/bin/env python3
"""Assemble the MuJoCo UR10 workspace scene from scene_config.yaml: UR10 arm (built from the DH numbers in
ur10_kinematics.py, simple primitive visuals), the PincOpen gripper (try3, assets/pincopen) mounted on the
flange, table, peg + hole blocks (assets/peg_hole), a top camera, a wrist camera on the gripper and a debug overview camera.

Library used by scene_view.py.  CPU physics; rendering needs an OpenGL backend (see scene_view.py).
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE                                  # outputs (cfg render.out_dir) are relative to this folder
sys.path.insert(0, str(HERE))
DEFAULT_CONFIG = HERE / "scene_config.yaml"
PINC = HERE / "assets" / "pincopen"
PEGH = HERE / "assets" / "peg_hole"
UR10 = HERE / "assets" / "ur10"
WRISTCAM = HERE / "assets" / "wrist_camera"
D435I = HERE / "assets" / "d435i"
import os
DATASETS = Path(os.environ.get("UR10SIM_DATASETS", HERE.parent / "datasets"))   # LeRobot v2.1 dataset folders (only needed for robot.source: dataset, replays, real-frame comparison)


# ───────────────────────────── config ─────────────────────────────
def load_config(path=None) -> dict:
    path = Path(path or DEFAULT_CONFIG)
    cfg = yaml.safe_load(path.read_text())
    cfg["_path"] = str(path)
    return cfg


def scene_key(cfg: dict) -> str:
    """everything that requires rebuilding the model / re-running physics (cameras and render size do not, except size)"""
    c = {k: v for k, v in cfg.items() if k not in ("cameras", "_path")}
    if mount_cfg(cfg["cameras"]["wrist"]).get("enabled"):      # the wrist camera is part of the model then (mount angle, D435i pose)
        c["wrist_mount"] = {k: v for k, v in cfg["cameras"]["wrist"].items() if k != "tilt_deg"}
    c["render"] = {k: v for k, v in cfg["render"].items() if k in ("settle_steps",)}
    return json.dumps(c, sort_keys=True)



_APPEAR = {}


def apply_appearance(img: np.ndarray, path) -> np.ndarray:
    """make a sim render look like the real camera: Gaussian blur (sigma px), per-channel colour map (piecewise-linear LUT fitted by quantile
    matching) and sensor noise.  Spec json written by calibrate_appearance.py."""
    p = Path(path) if Path(path).is_absolute() else HERE / path
    if p not in _APPEAR:
        _APPEAR[p] = json.loads(p.read_text())
    a = _APPEAR[p]
    x = img.astype(np.float32)
    if a.get("gainmap"):                                                 # smooth per-pixel colour gain (lighting / vignetting of the real camera)
        gp = Path(a["gainmap"]) if Path(a["gainmap"]).is_absolute() else p.parent / a["gainmap"]
        if gp not in _APPEAR:
            _APPEAR[gp] = np.load(gp).astype(np.float32)
        gm = _APPEAR[gp]
        from scipy.ndimage import zoom
        x = x * zoom(gm, (x.shape[0] / gm.shape[0], x.shape[1] / gm.shape[1], 1), order=1)
    if a.get("blur_sigma", 0) > 0:
        from scipy.ndimage import gaussian_filter
        x = gaussian_filter(x, sigma=(a["blur_sigma"], a["blur_sigma"], 0))
    xs = np.asarray(a["lut_x"], np.float32)
    out = np.empty_like(x)
    for c in range(3):
        out[..., c] = np.interp(x[..., c], xs, np.asarray(a["lut_y"][c], np.float32))
    if a.get("noise_std", 0) > 0:
        out += np.random.default_rng().normal(0, a["noise_std"], out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)

# ───────────────────────────── math ─────────────────────────────
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mat2quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def rpy_to_R(rpy_deg):
    r, p, y = np.radians(rpy_deg)
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def wrist_body_yaw(cam: dict) -> dict:
    """apply cam['body_yaw_deg']: rotate the WHOLE camera body (position, view direction, up) about the gripper tool axis (x of G) --
    with the gripper pointing down this is a yaw of the camera mount, not just of its view"""
    yaw = cam.get("body_yaw_deg", 0.0)
    if not yaw:
        return cam
    Ry = rot_x(np.radians(yaw))
    c = dict(cam)
    for k in ("pos", "look_at", "up_hint"):
        if c.get(k) is not None:
            c[k] = list(Ry @ np.array(c[k], float))
    if c.get("rpy_deg") is not None:
        c["_R_pre"] = Ry
    return c


def camera_R(cam: dict, *, wrist: bool) -> np.ndarray:
    """camera frame (columns x=right, y=up, z=backward; MuJoCo cameras look along -z) from the config block"""
    if wrist:
        cam = wrist_body_yaw(cam)
    if cam.get("rpy_deg") is not None:
        R = rpy_to_R(cam["rpy_deg"])
        if "_R_pre" in cam:
            R = cam["_R_pre"] @ R
    else:
        pos, tgt, up = np.array(cam["pos"], float), np.array(cam["look_at"], float), np.array(cam["up_hint"], float)
        z = pos - tgt
        z /= np.linalg.norm(z)
        x = np.cross(up, z)
        if np.linalg.norm(x) < 1e-6:
            raise ValueError("camera up_hint is parallel to the viewing direction -- pick another up_hint")
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=1)
    if not wrist and cam.get("yaw_deg"):
        R = rot_z(np.radians(cam["yaw_deg"])) @ R      # extra rotation of the whole orientation about the world +z axis (anticlockwise seen from above)
    if wrist and cam.get("tilt_deg"):
        R = R @ rot_x(-np.radians(cam["tilt_deg"]))
    if cam.get("roll_deg"):
        R = R @ rot_z(np.radians(cam["roll_deg"]))
    return R


def fovy_deg(cam: dict, width: int, height: int) -> float:
    fh = np.radians(cam.get("fov_h_deg", 69.0))
    return float(np.degrees(2 * np.arctan(np.tan(fh / 2) * height / width)))


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)



# ───────────────────────────── wrist camera mount + D435i ─────────────────────────────
MATS = {"IR_Lens": (0.036, 0.036, 0.036, 1), "IR_Emitter_Lens": (0.287, 0.665, 0.328, 1), "IR_Rim": (0.799, 0.807, 0.799, 1),
        "Cameras_Gray": (0.296, 0.296, 0.296, 1), "Black_Acrylic": (0.070, 0.070, 0.070, 1), "RGB_Pupil": (0.087, 0.003, 0.009, 1),
        "Metal_Casing": (0.78, 0.78, 0.80, 1)}
D435_DEPTH = 0.025


def mount_cfg(w: dict) -> dict:
    return w.get("mount") or {}


def wrist_mount_frames(w: dict) -> dict:
    """geometry of the CAD camera mount + D435i in the pivot frame (pivot point of the mount, axes of G), before mount.angle_deg.
    D435i model frame (Menagerie): z out of the lenses (= mount plate outward normal), x long axis (RGB lens at +x), y = up of the
    device as seen from the front (RGB on the viewer's right).  The image of a lens: forward +z, up +y, therefore image-right = -x."""
    mc = mount_cfg(w)
    meta = json.loads((WRISTCAM / "wrist_camera_meta.json").read_text())
    dm = json.loads((D435I / "d435i_meta.json").read_text())
    piv, mid, n = (np.array(meta[k], float) for k in ("pivot_G", "hole_mid_G", "normal_G"))
    sgn = 1.0 if float(mc.get("rgb_toward_z", 1)) >= 0 else -1.0
    zc = n
    xc = np.array([0, 0, sgn])                       # long axis; the RGB lens (+x of the model) points to sgn * z_g
    yc = np.cross(zc, xc)
    Rc = np.stack([xc, yc, zc], axis=1)              # columns = model axes in G
    lens = mc.get("lens", "rgb")
    if lens not in dm["lenses"]:
        raise ValueError(f"wrist.mount.lens must be one of {list(dm['lenses'])}")
    origin = mid + D435_DEPTH * n - piv              # model origin (centre of the front face) relative to the pivot
    return dict(meta=meta, dmeta=dm, pivot=piv, Rc=Rc, origin=origin, lens_local=np.array(dm["lenses"][lens], float), lens=lens)


def add_wrist_mount(gbody: ET.Element, asset: ET.Element, w: dict, W: int, H: int) -> None:
    """CAD camera mount (rotates about its pivot by mount.angle_deg) carrying the D435i; the sim camera 'wrist' sits at the chosen lens"""
    mc = mount_cfg(w)
    F = wrist_mount_frames(w)
    ver = mc.get("version", "V6_REINFORCED")
    ET.SubElement(asset, "mesh", name="camera_mount", file=str(WRISTCAM / "meshes" / f"camera_mount_{ver}.stl"), scale="0.001 0.001 0.001" if False else "1 1 1")
    for p in F["dmeta"]["parts"]:
        ET.SubElement(asset, "mesh", name=p["mesh"], file=str(D435I / "meshes" / f"{p['mesh']}.stl"))
    yaw = ET.SubElement(gbody, "body", name="wrist_yaw", quat=fmt(mat2quat(rot_x(np.radians(w.get("body_yaw_deg", 0.0))))))
    mount = ET.SubElement(yaw, "body", name="wrist_mount", pos=fmt(F["pivot"]), quat=fmt(mat2quat(rot_z(np.radians(mc.get("angle_deg", 0.0))))))
    vis = dict(type="mesh", contype="0", conaffinity="0", group="1", mass="1e-6")  # ~0 kg but nonzero: some MuJoCo versions (2.3.7) refuse to compile a body whose every geom is exactly mass=0 ("body mass is too small, cannot compute center of mass") -- purely visual/noncolliding meshes either way
    ET.SubElement(mount, "geom", mesh="camera_mount", rgba="0.93 0.55 0.12 1", **vis)
    cam = ET.SubElement(mount, "body", name="d435i", pos=fmt(F["origin"]), quat=fmt(mat2quat(F["Rc"])))
    for p in F["dmeta"]["parts"]:
        ET.SubElement(cam, "geom", mesh=p["mesh"], rgba=fmt(MATS[p["material"]]), **vis)
    Rmj = np.diag([-1.0, 1.0, -1.0])                 # MuJoCo camera axes (right, up, back) in the D435i model frame: right = -x, up = +y, looks along +z
    ET.SubElement(cam, "camera", name="wrist", pos=fmt(F["lens_local"]), quat=fmt(mat2quat(Rmj)), fovy=f"{fovy_deg(w, W, H):.4f}")


# ───────────────────────────── MJCF assembly ─────────────────────────────
def _abs_mesh(elem: ET.Element, folder: Path):
    if elem.get("file"):
        elem.set("file", str((folder / "meshes" / elem.get("file")).resolve()))


def build_xml(cfg: dict) -> str:
    import pincopen as PC
    root = ET.Element("mujoco", model="ur10_scene")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.001", integrator="implicitfast", impratio="10", noslip_iterations="3", gravity="0 0 -9.81")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1280", offheight="960")
    ET.SubElement(vis, "quality", shadowsize="4096")
    ET.SubElement(vis, "headlight", ambient="0.4 0.4 0.4", diffuse="0.2 0.2 0.2", specular="0 0 0")
    ET.SubElement(root, "statistic", extent="2.0", center="-0.4 -0.3 0.3")
    default = ET.SubElement(root, "default")
    asset = ET.SubElement(root, "asset")
    wb = ET.SubElement(root, "worldbody")

    # ---- gripper (pincopen.xml): defaults, meshes, body, equality, actuator
    g = ET.parse(PINC / "pincopen.xml").getroot()
    for ch in list(g.find("default")):
        default.append(ch)
    for m in g.find("asset"):
        _abs_mesh(m, PINC)
        asset.append(m)
    gbody = g.find("worldbody").find("body")            # pincopen_base

    # ---- peg / holes (peg_hole_test.xml + assets)
    pa = ET.parse(PEGH / "peg_hole_assets.xml").getroot()
    for m in pa.find("asset"):
        _abs_mesh(m, PEGH)
        asset.append(m)
    pt = ET.parse(PEGH / "peg_hole_test.xml").getroot().find("worldbody")
    pbodies = {b.get("name"): b for b in pt.findall("body")}

    tz = cfg["table"]["z"]
    tc, ts = cfg["table"]["center_xy"], cfg["table"]["size_xy"]
    # lighting: the real lab has several diffuse sources, so shadows are faint.  Strong ambient + several shadow-less fill lights + one very weak
    # shadow caster (a shadow only removes that light's share of the light, the ambient/fill stay).
    for i_, (lx, ly) in enumerate(((-1.2, 0.4), (1.2, 0.4), (-1.2, -1.6), (1.2, -1.6))):
        ET.SubElement(wb, "light", name=f"fill{i_}", pos=f"{lx} {ly} 2.2", dir=f"{-lx * 0.4} {-(ly + 0.6) * 0.4} -1", diffuse="0.18 0.18 0.18", specular="0 0 0", castshadow="false")
    ET.SubElement(wb, "light", name="key", pos="0.0 -0.6 2.6", dir="0 0 -1", diffuse="0.07 0.07 0.07", specular="0 0 0", castshadow="false")
    if cfg.get("floor", {}).get("enabled", True):                  # lab floor far below the board (the real cameras see it beside/behind the board)
        ET.SubElement(wb, "geom", name="floor", type="plane", pos=f"0 0 {cfg.get('floor', {}).get('z', -0.75)}", size="6 6 0.1", contype="0", conaffinity="0",
                      rgba=fmt(cfg.get("floor", {}).get("rgba", [0.33, 0.35, 0.36, 1.0])))
    tex = cfg["table"].get("texture")
    mat = {}
    if tex:                                                       # 2-D texture on the board (procedural plywood: make_wood_texture.py)
        tp = Path(tex) if Path(tex).is_absolute() else HERE / tex
        ET.SubElement(asset, "texture", name="table_tex", type="2d", file=str(tp))
        ET.SubElement(asset, "material", name="table_mat", texture="table_tex", texrepeat="1 1", texuniform="false", specular="0", shininess="0")
        mat = {"material": "table_mat"}
    ET.SubElement(wb, "geom", name="table", type="box", pos=fmt([tc[0], tc[1], tz - 0.02]), size=fmt([ts[0] / 2, ts[1] / 2, 0.02]),
                  rgba=fmt(cfg["table"]["rgba"]), friction="0.8 0.005 0.0005", solref="0.004 1", condim="4", **mat)
    # UR10 (CB3) from the official Universal Robots description (assets/ur10, built by build_ur10.py)
    u = ET.parse(UR10 / "ur10.xml").getroot()
    for ch in list(u.find("default")):
        default.append(ch)
    for m in u.find("asset"):
        m.set("file", str((UR10 / "meshes" / m.get("file")).resolve()))
        asset.append(m)
    urb = u.find("worldbody").find("body")
    wb.append(urb)
    flange = next(b for b in urb.iter("body") if b.get("name") == "flange_dh")
    # mount the gripper: G frame expressed in the flange frame
    mt = PC._META["mount"]
    Rm = np.array(mt["R_flange_gripper"])
    gbody.set("pos", fmt(mt["p_flange_gripper_m"]))
    gbody.set("quat", fmt(mat2quat(Rm)))
    ET.SubElement(gbody, "site", name="tcp", pos=fmt([mt["tcp_x_g_mm"] * 1e-3, -1.1e-3, 0]), size="0.004", rgba="1 0 0 1")
    w = cfg["cameras"]["wrist"]
    W, H = cfg["render"]["width"], cfg["render"]["height"]
    if mount_cfg(w).get("enabled"):
        add_wrist_mount(gbody, asset, w, W, H)
    else:
        Rw = camera_R(w, wrist=True)
        ET.SubElement(gbody, "camera", name="wrist", pos=fmt(wrist_body_yaw(w)["pos"]), quat=fmt(mat2quat(Rw)), fovy=f"{fovy_deg(w, W, H):.4f}")
    flange.append(gbody)
    for name in ("top", "overview"):
        c = cfg["cameras"][name]
        Rc = camera_R(c if name == "top" else {**c, "up_hint": c.get("up_hint", [0, 0, 1])}, wrist=False)
        ET.SubElement(wb, "camera", name=name, pos=fmt(c["pos"]), quat=fmt(mat2quat(Rc)), fovy=f"{fovy_deg(c, W, H):.4f}")

    # holes
    for key, bname in (("circle", "hole_circle"), ("square", "hole_square")):
        hc = cfg["holes"][key]
        if not hc["enabled"]:
            continue
        b = copy.deepcopy(pbodies[bname])
        b.set("pos", fmt([hc["x"], hc["y"], tz + 0.044]))
        b.set("quat", fmt(mat2quat(rot_z(np.radians(hc["yaw_deg"])))))
        if not cfg["holes"]["fixed"]:
            b.insert(0, ET.Element("freejoint", name=f"{bname}_free"))
        wb.append(b)
    wb.append(copy.deepcopy(pbodies["peg"]))
    eq = g.find("equality")
    ET.SubElement(eq, "weld", name="grasp_assist", body1="pincopen_base", body2="peg", active="false", solref="0.004 1", solimp="0.9 0.95 0.001")   # switched on by Scene.grasp_assist
    root.append(eq)
    root.append(g.find("actuator"))
    return ET.tostring(root, encoding="unicode")


# ───────────────────────────── scene object ─────────────────────────────
class Scene:
    def __init__(self, cfg: dict):
        import mujoco
        self.mj = mujoco
        self.cfg = cfg
        self.key = scene_key(cfg)
        self.model = mujoco.MjModel.from_xml_string(build_xml(cfg))
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self._size = None
        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.arm_adr = [m.jnt_qposadr[jid(f"joint_{i}")] for i in range(6)]
        self.arm_dof = [m.jnt_dofadr[jid(f"joint_{i}")] for i in range(6)]
        self.cam_adr = m.jnt_qposadr[jid("cam_joint")]
        self.peg_adr = m.jnt_qposadr[jid("peg_free")]
        self.peg_dof = m.jnt_dofadr[jid("peg_free")]
        self.bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.cid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, n)
        self.ctrl_range = m.actuator_ctrlrange[0].copy()
        self.apply_state()

    # -- robot state from config
    def robot_state(self):
        r = self.cfg["robot"]
        if r["source"] == "dataset":
            import pandas as pd
            d = r["dataset"]
            ep = int(d["episode"])
            f = DATASETS / d["name"] / "data" / f"chunk-{ep // 1000:03d}" / f"episode_{ep:06d}.parquet"
            df = pd.read_parquet(f)
            s = np.asarray(df["observation.state"].iloc[int(d["frame"])], float)
            return s[:6], float(s[6])
        return np.radians(r["joints_deg"]), float(r["gripper_frac"])

    def apply_state(self):
        import pincopen as PC
        mj, m, d, cfg = self.mj, self.model, self.data, self.cfg
        q, frac = self.robot_state()
        mj.mj_resetData(m, d)
        d.qpos[self.arm_adr] = q
        target = float(np.clip(PC.cam_angle_for_frac(frac), self.ctrl_range[0], self.ctrl_range[1]))
        tz = cfg["table"]["z"]
        peg = cfg["peg"]
        if peg["mode"] == "table":
            R = rot_z(np.radians(peg["yaw_deg"])) @ rot_y(np.pi / 2)
            d.qpos[self.peg_adr:self.peg_adr + 7] = [peg["x"], peg["y"], tz + 0.06, *mat2quat(R)]
        n = int(cfg["render"]["settle_steps"])
        ramp = max(1, int(0.65 * n))
        gid = self.bid("pincopen_base")
        for k in range(n):
            d.ctrl[0] = target * min(1.0, k / ramp)
            d.qpos[self.arm_adr] = q
            d.qvel[self.arm_dof] = 0
            if peg["mode"] == "gripper":
                Rg = d.xmat[gid].reshape(3, 3)
                off = peg["gripper_offset"]
                Ro = rpy_to_R(off["rpy_deg"])
                pos = d.xpos[gid] + Rg @ np.array(off["xyz"], float)
                d.qpos[self.peg_adr:self.peg_adr + 7] = [*pos, *mat2quat(Rg @ Ro)]
                d.qvel[self.peg_dof:self.peg_dof + 6] = 0
            mj.mj_step(m, d)
        d.qpos[self.arm_adr] = q
        mj.mj_forward(m, d)
        self.q, self.frac = q, frac

    # -- grasp assist: the pad physics does not hold the lying peg (sim-to-real gap), so once both pads touch the peg while the gripper is
    #    commanded closed, the peg is squared up between the pads (axis along the hinge axis y_g, centred in z_g) and welded to the gripper;
    #    it is released when the gripper is commanded open.
    def grasp_assist(self, want_grip: bool) -> bool:
        mj, m, d = self.mj, self.model, self.data
        if not hasattr(self, "_eq"):
            self._eq = mj.mj_name2id(m, mj.mjtObj.mjOBJ_EQUALITY, "grasp_assist")
            self._pads = {mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, n) for n in ("pad_L", "pad_R")}
            self._pegb = self.bid("peg"); self._gb = self.bid("pincopen_base")
        held = bool(d.eq_active[self._eq])
        if held and not want_grip:
            d.eq_active[self._eq] = 0
            return False
        if held:
            # (1) once the peg is off the table, slide it along the tool axis until its axis passes through the TCP (the dataset TCP is where the
            #     peg tip ends up over the hole; with the peg held at the pad centre the tip landed ~15 mm off).
            pcfg = self.cfg.get("peg", {})
            if getattr(self, "_x_target", None) is not None and d.xpos[self._pegb][2] > self.cfg["table"]["z"] + 0.055:
                m.eq_solref[self._eq, 0] = float(pcfg.get("assist_solref", 0.004))
                cur = m.eq_data[self._eq, 3]
                dx = float(np.clip(self._x_target - cur, -1e-4, 1e-4))
                m.eq_data[self._eq, 3] = cur + dx
                self._pivot[0] += dx
            # (2) a two-pad pinch does not fix rotation about the jaw axis z_g: the peg hangs from the pads under gravity.  Rotate it about z_g
            #     (through the grasp point) so that its centre hangs straight below the grasp point, at most 2 rad/s.
            if pcfg.get("hang_under_gravity", True):
                Rg = d.xmat[self._gb].reshape(3, 3)
                dn = Rg.T @ np.array([0.0, 0.0, -1.0])
                un = np.linalg.norm(dn[:2])
                pr = m.eq_data[self._eq, 3:6].copy()
                v = pr[:2] - self._pivot
                nv = np.linalg.norm(v)
                if un > 0.2 and nv > 5e-3:
                    u, c = dn[:2] / un, v / nv
                    ang = float(np.clip(np.arctan2(c[0] * u[1] - c[1] * u[0], c @ u), -2e-3, 2e-3))
                    cs, sn = np.cos(ang), np.sin(ang)
                    m.eq_data[self._eq, 3:5] = self._pivot + np.array([cs * v[0] - sn * v[1], sn * v[0] + cs * v[1]])
                    Rr = np.zeros(9); mj.mju_quat2Mat(Rr, m.eq_data[self._eq, 6:10].copy()); Rr = Rr.reshape(3, 3)
                    Rz = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1.0]])
                    m.eq_data[self._eq, 6:10] = mat2quat(Rz @ Rr)
        if not held and want_grip:
            touching = set()
            for c in d.contact[:d.ncon]:
                for a_, b_ in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if a_ in self._pads and m.geom_bodyid[b_] == self._pegb:
                        touching.add(a_)
            if len(touching) == 2:
                Rg, pg = d.xmat[self._gb].reshape(3, 3), d.xpos[self._gb]
                Rr, pr = Rg.T @ d.xmat[self._pegb].reshape(3, 3), Rg.T @ (d.xpos[self._pegb] - pg)
                ax = Rr[:, 2]; tg = np.array([0, np.sign(ax[1]) or 1.0, 0.0]); v = np.cross(ax, tg); c = float(ax @ tg)
                K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                Rr = (np.eye(3) + K + K @ K / (1 + c)) @ Rr; pr = pr.copy(); pr[2] = 0.0
                d.qpos[self.peg_adr:self.peg_adr + 7] = [*(pg + Rg @ pr), *mat2quat(Rg @ Rr)]; d.qvel[self.peg_dof:self.peg_dof + 6] = 0
                m.eq_data[self._eq, :] = 0
                m.eq_data[self._eq, 3:6] = pr; m.eq_data[self._eq, 6:10] = mat2quat(Rr); m.eq_data[self._eq, 10] = 1.0
                import pincopen as PC
                self._x_target = float(PC._META["mount"]["tcp_x_g_mm"]) * 1e-3 if self.cfg.get("peg", {}).get("axis_through_tcp", True) else None
                self._pivot = np.array([pr[0], 0.0])
                d.eq_active[self._eq] = 1
                return True
        return held

    # -- cameras (cheap update, no physics)
    def update_cameras(self, cfg: dict):
        self.cfg["cameras"] = cfg["cameras"]
        W, H = cfg["render"]["width"], cfg["render"]["height"]
        m = self.model
        for name in ("top", "wrist", "overview"):
            c = cfg["cameras"][name]
            i = self.cid(name)
            if name == "overview":
                c = {**c, "up_hint": c.get("up_hint", [0, 0, 1])}
            m.cam_fovy[i] = fovy_deg(c, W, H)
            if name == "wrist" and mount_cfg(c).get("enabled"):
                continue                                  # pose comes from the mount body (scene_key rebuilds on changes)
            R = camera_R(c, wrist=(name == "wrist"))
            m.cam_pos[i] = wrist_body_yaw(c)["pos"] if name == "wrist" else c["pos"]
            m.cam_quat[i] = mat2quat(R)
        self.mj.mj_forward(m, self.data)

    def render(self, width: int, height: int, cams=("top", "wrist", "overview")) -> dict:
        if self.renderer is None or self._size != (width, height):
            self._close_renderer()
            self.renderer = self.mj.Renderer(self.model, height, width)
            self._size = (width, height)
        out = {}
        for name in cams:
            self.renderer.update_scene(self.data, camera=name)
            img = self.renderer.render().copy()
            cc = self.cfg["cameras"].get(name, {})
            gain = cc.get("color_gain")                                       # per-camera white balance / exposure (real cameras differ from each other)
            if gain is not None:
                img = np.clip(img.astype(np.float32) * np.asarray(gain, np.float32), 0, 255).astype(np.uint8)
            if cc.get("appearance"):                                          # fitted blur + colour map + noise (calibrate_appearance.py)
                img = apply_appearance(img, cc["appearance"])
            out[name] = img
        return out

    def _close_renderer(self):
        if self.renderer is not None:
            if hasattr(self.renderer, "close"):        # MuJoCo >= 3; 2.3.x frees on garbage collection
                self.renderer.close()
            self.renderer = None

    def close(self):
        self._close_renderer()


# ───────────────────────────── real frames for comparison ─────────────────────────────
def real_frames(cfg: dict, width: int, height: int):
    """(top, wrist) RGB uint8 arrays of the configured dataset frame, or None"""
    r = cfg["robot"]
    if r["source"] != "dataset":
        return None
    from PIL import Image
    d = r["dataset"]
    ep, fr = int(d["episode"]), int(d["frame"])
    out_dir = REPO / cfg["render"]["out_dir"] / "real"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    for cam in ("cam_high", "cam_right_wrist"):
        png = out_dir / f"{d['name']}_ep{ep}_f{fr}_{cam}.png"
        if not png.exists():
            vid = DATASETS / d["name"] / "videos" / f"chunk-{ep // 1000:03d}" / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(vid), "-vf", f"select=eq(n\\,{fr})", "-vframes", "1", str(png)],
                           check=True)
        res.append(np.asarray(Image.open(png).convert("RGB").resize((width, height))))
    return res[0], res[1]
