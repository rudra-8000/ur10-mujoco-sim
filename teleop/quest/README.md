# Quest 3 teleop page

`index.html` = `index_3.html` from https://github.com/rudra-8000/UR10-Meta-Quest-3-Teleop
plus a camera panel (`/cam_ws`). Served by `quest_teleop.py` (HTTPS+WSS on port
8443, self-signed cert auto-generated into `quest/certs/`, gitignored).
three.js is loaded from cdn.jsdelivr.net, so the Quest needs internet access
while using this page. See `quest_teleop.py`'s docstring and `../GUIDE.md`
(Meta Quest 3 section) for the full setup and controls.
