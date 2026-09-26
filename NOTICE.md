# Third-party material

* **UR10 (CB3) description** — `ur10sim/assets/ur10/` (`source/` = Universal Robots ROS2 Description, commit 89bbe795; meshes and MJCF derived from it). BSD-3-Clause, Universal Robots A/S. Licence text: `ur10sim/assets/ur10/LICENSE_UniversalRobots_BSD3`.
* **Intel RealSense D435i visual model** — `ur10sim/assets/d435i/` (decimated copy of `realsense_d435i` from MuJoCo Menagerie, google-deepmind/mujoco_menagerie). Apache-2.0, licence text: `ur10sim/assets/d435i/LICENSE_Apache2_MuJoCo_Menagerie`.
* **Lab-designed parts** — PincOpen gripper (revision try3 linkage from the try2 assembly export), camera mount V2/V6, peg and hole blocks: STL exports in `cad/`, derived meshes in `ur10sim/assets/`. Licence: to be set by the repository owner.
* `ur10sim/ur10_kinematics.py` uses the publicly published nominal UR10 DH parameters.
