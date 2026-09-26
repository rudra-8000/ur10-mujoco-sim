#!/usr/bin/env python3
"""Build the MuJoCo UR10 (CB3) arm from the OFFICIAL Universal Robots ROS2 description
(https://github.com/UniversalRobots/Universal_Robots_ROS2_Description @ 89bbe795, BSD-3-Clause, see assets/ur10/LICENSE_*):
  - kinematics: config/ur10/default_kinematics.yaml   (joint origins + rpy, exactly as in ur_macro.xacro)
  - meshes:     meshes/ur10/visual/*.dae              (converted here: one STL per material, colours kept) + visual_parameters.yaml offsets
  - masses/COGs: physical_parameters.yaml
The chain hangs from `base_link_inertia` (= the UR controller base frame the datasets use).  A fixed body `flange_dh` is added whose frame is
the DH frame 6 that ur10_kinematics.py uses (constant transform from wrist_3_link, solved numerically here), so the gripper mount
computed from that FK stays valid.  Output: assets/ur10/ur10.xml (+ ur10_meta.json).  verify_arm.py checks the FK.
"""
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
A = HERE / "assets" / "ur10"
SRC = A / "source"
sys.path.insert(0, str(HERE))
import ur10_kinematics as K  # noqa: E402

LINKS = ["base", "shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"]
JOINT_LINKS = LINKS[1:]                     # link i+1 carries joint i


class _L(yaml.SafeLoader):
    pass


_L.add_constructor("!degrees", lambda l, n: math.radians(float(l.construct_scalar(n))))


def rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def T_from(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_R(*rpy)
    T[:3, 3] = xyz
    return T


def mat2quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def fmt(v):
    return " ".join(f"{x:.8g}" for x in v)


# ---------------- COLLADA (.dae) -> triangles grouped by material ----------------
def _strip(tag):
    return tag.split("}")[-1]


def parse_dae(path):
    root = ET.parse(path).getroot()
    ns = {"c": root.tag.split("}")[0].strip("{")}
    find = lambda e, p: e.findall(p, ns)
    effects = {}
    for e in find(root, ".//c:library_effects/c:effect"):
        col = e.find(".//c:diffuse/c:color", ns)
        effects[e.get("id")] = [float(x) for x in col.text.split()] if col is not None else [0.7, 0.7, 0.7, 1.0]
    mats = {}
    for m in find(root, ".//c:library_materials/c:material"):
        eff = m.find("c:instance_effect", ns).get("url").lstrip("#")
        mats[m.get("id")] = effects.get(eff, [0.7, 0.7, 0.7, 1.0])
    # symbol -> material id, via the scene bindings
    sym = {}
    for im in find(root, ".//c:instance_material"):
        sym[im.get("symbol")] = im.get("target").lstrip("#")
    out = []
    for g in find(root, ".//c:library_geometries/c:geometry"):
        mesh = g.find("c:mesh", ns)
        srcs = {s.get("id"): np.array([float(x) for x in s.find("c:float_array", ns).text.split()]).reshape(-1, 3)
                for s in find(mesh, "c:source") if s.find("c:float_array", ns) is not None}
        verts = {v.get("id"): v.find("c:input[@semantic='POSITION']", ns).get("source").lstrip("#") for v in find(mesh, "c:vertices")}
        for tri in find(mesh, "c:triangles"):
            inputs = find(tri, "c:input")
            stride = max(int(i.get("offset")) for i in inputs) + 1
            vin = [i for i in inputs if i.get("semantic") == "VERTEX"][0]
            pos = srcs[verts[vin.get("source").lstrip("#")]]
            idx = np.array([int(x) for x in tri.find("c:p", ns).text.split()]).reshape(-1, stride)[:, int(vin.get("offset"))]
            tris = pos[idx].reshape(-1, 3, 3)
            rgba = mats.get(sym.get(tri.get("material"), ""), [0.7, 0.7, 0.7, 1.0])
            out.append((tris, rgba))
    return out


def write_stl(path, tris):
    import struct
    tris = np.asarray(tris, np.float32)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-20
    with open(path, "wb") as f:
        f.write(b"UR10 visual from Universal_Robots_ROS2_Description dae".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for t, nn in zip(tris, n):
            f.write(struct.pack("<12fH", *nn, *t.reshape(-1), 0))


# ---------------- build ----------------
def main():
    kin = yaml.safe_load((SRC / "default_kinematics.yaml").read_text())["kinematics"]
    vis = yaml.load((SRC / "visual_parameters.yaml").read_text(), Loader=_L)["mesh_files"]
    phys = yaml.safe_load((SRC / "physical_parameters.yaml").read_text())["inertia_parameters"]
    mass = {"base": phys["base_mass"], **{l: phys[f"{l}_mass"] for l in JOINT_LINKS}}
    cog = {l: phys["center_of_mass"].get(f"{l}_cog") for l in JOINT_LINKS}

    # meshes
    mesh_geoms = {}
    for l in LINKS:
        groups = parse_dae(SRC / f"{l.replace('_', '')}.dae")
        geoms = []
        for i, (tris, rgba) in enumerate(groups):
            name = f"{l}_{i}"
            write_stl(A / "meshes" / f"{name}.stl", tris)
            geoms.append((name, rgba))
        mesh_geoms[l] = geoms

    root = ET.Element("mujoco", model="ur10")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    default = ET.SubElement(root, "default")
    ET.SubElement(ET.SubElement(default, "default", {"class": "ur_vis"}), "geom", type="mesh", contype="0", conaffinity="0", group="1")
    asset = ET.SubElement(root, "asset")
    for l in LINKS:
        for name, _ in mesh_geoms[l]:
            ET.SubElement(asset, "mesh", name=f"ur10_{name}", file=f"{name}.stl")
    wb = ET.SubElement(root, "worldbody")

    def add_visuals(body, l):
        o = vis[l]["mesh_offset"]
        T = T_from([o["x"], o["y"], o["z"]], [o["roll"], o["pitch"], o["yaw"]])
        for name, rgba in mesh_geoms[l]:
            ET.SubElement(body, "geom", {"class": "ur_vis"}, mesh=f"ur10_{name}", pos=fmt(T[:3, 3]), quat=fmt(mat2quat(T[:3, :3])), rgba=fmt(rgba))

    def add_inertial(body, l):
        m = mass[l]
        c = cog.get(l) or {"x": 0.0, "y": 0.0, "z": 0.0}
        d = 0.002 * m
        ET.SubElement(body, "inertial", pos=fmt([c["x"], c["y"], c["z"]]), mass=f"{m}", diaginertia=fmt([d, d, d]))

    base = ET.SubElement(wb, "body", name="ur_base")
    add_inertial(base, "base"); add_visuals(base, "base")
    cur = base
    T_w3 = np.eye(4)
    for i, l in enumerate(JOINT_LINKS):
        k = kin[l]
        xyz, rpy = [k["x"], k["y"], k["z"]], [k["roll"], k["pitch"], k["yaw"]]
        T = T_from(xyz, rpy)
        b = ET.SubElement(cur, "body", name=f"ur_{l}", pos=fmt(T[:3, 3]), quat=fmt(mat2quat(T[:3, :3])))
        ET.SubElement(b, "joint", name=f"joint_{i}", type="hinge", axis="0 0 1", damping="2", armature="0.01")
        add_inertial(b, l); add_visuals(b, l)
        cur = b
    # flange_dh: constant transform wrist_3_link -> DH frame 6, solved at q = 0 from the two FKs
    T_urdf = np.eye(4)
    for l in JOINT_LINKS:
        k = kin[l]
        T_urdf = T_urdf @ T_from([k["x"], k["y"], k["z"]], [k["roll"], k["pitch"], k["yaw"]])
    T_dh = np.eye(4)
    for i in range(6):
        T_dh = T_dh @ K._dh_transform(K._A[i], K._D[i], K._ALPHA[i], 0.0)
    T_rel = np.linalg.inv(T_urdf) @ T_dh
    fl = ET.SubElement(cur, "body", name="flange_dh", pos=fmt(T_rel[:3, 3]), quat=fmt(mat2quat(T_rel[:3, :3])))
    ET.SubElement(fl, "site", name="flange", size="0.004", rgba="0 0 1 1")
    ET.ElementTree(root).write(A / "ur10.xml")
    (A / "ur10_meta.json").write_text(json.dumps(dict(
        source="UniversalRobots/Universal_Robots_ROS2_Description @ 89bbe795f38a7ab00fb66fe8831dfff79dc99edf (BSD-3-Clause)",
        T_wrist3_to_dh6=T_rel.tolist(), masses=mass), indent=1))
    print("wrote", A / "ur10.xml", "| wrist_3 -> DH frame 6 transform:\n", T_rel.round(6))


if __name__ == "__main__":
    sys.exit(main())
