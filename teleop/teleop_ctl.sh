#!/usr/bin/env bash
# Standalone teleop launcher (run on the robot PC, your lerobot env active).
#   bash teleop_ctl.sh gello [--live] [teleop.py options]
#   bash teleop_ctl.sh quest [--live] [--quest-ip IP] ...
# Dry-run (display only) unless --live. Camera view opens as an X11 window
# (ssh -X) and, for quest, inside the headset. See teleop.py --help and GUIDE.md.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
method="${1:-}"
case "$method" in gello|quest) shift ;; *) sed -n 2,6p "$0"; exit 1 ;; esac
PY="$(command -v python3 || command -v python)"
UR10_IP="${UR10_IP:-192.168.100.3}"
exec "$PY" "$SCRIPT_DIR/teleop.py" --method "$method" --ur-ip "$UR10_IP" "$@"
