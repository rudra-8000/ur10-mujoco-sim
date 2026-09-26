#!/usr/bin/env python3
"""Render the MuJoCo UR10 workspace described by scene_config.yaml.

  python ur10sim/scene_view.py                 # render once -> ur10sim/out/{top,wrist,overview,compare}.png
  python ur10sim/scene_view.py --watch         # re-render every time the config file is saved
  python ur10sim/scene_view.py --watch --window   # ... and show a live window (tkinter; works over ssh -X)
  python ur10sim/scene_view.py --config my.yaml --no-compare

Outputs (out_dir in the config): top.png, wrist.png, overview.png (debug view of robot + table) and, when robot.source is 'dataset',
compare.png = [sim | real | 50% blend] for both cameras so the camera/hole/peg placement can be tuned against real frames.

Rendering uses EGL.  On this machine only the NVIDIA EGL vendor works (the render nodes for Mesa are not accessible); a tiny offscreen
context (~100-200 MB) is created on the GPU with the most free memory (set MUJOCO_EGL_DEVICE_ID to override).  Use --gl osmesa /
--gl glfw to try other backends.  Physics runs on the CPU.
"""
import argparse
import atexit
import os
import subprocess
import sys
import time
from pathlib import Path


def _setup_gl(backend: str) -> None:
    if backend == "egl":
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        vend = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
        if vend.exists():
            os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(vend))
        if "MUJOCO_EGL_DEVICE_ID" not in os.environ:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
                best = max((tuple(int(x) for x in l.split(",")) for l in out), key=lambda t: t[1])
                os.environ["MUJOCO_EGL_DEVICE_ID"] = str(best[0])
            except Exception:  # noqa: BLE001
                pass
    else:
        os.environ["MUJOCO_GL"] = backend
    # the EGL context destructor raises harmlessly at interpreter exit
    sys.unraisablehook = lambda *a, **k: None


def _label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    ImageDraw.Draw(im).rectangle([0, 0, 8 * len(text) + 8, 16], fill=(0, 0, 0))
    ImageDraw.Draw(im).text((4, 2), text, fill=(255, 255, 255))
    import numpy as np
    return np.asarray(im)


def render_once(scene, cfg, out_dir: Path, compare: bool):
    import numpy as np
    from PIL import Image
    import build_scene as B
    W, H = cfg["render"]["width"], cfg["render"]["height"]
    imgs = scene.render(W, H)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in imgs.items():
        Image.fromarray(v).save(out_dir / f"{k}.png")
    files = [out_dir / f"{k}.png" for k in imgs]
    composite = None
    real = B.real_frames(cfg, W, H) if compare and cfg["render"].get("compare", True) else None
    if real is not None:
        rows = []
        for name, sim, rl in (("top (cam_high)", imgs["top"], real[0]), ("wrist (cam_right_wrist)", imgs["wrist"], real[1])):
            blend = ((sim.astype(np.float32) + rl.astype(np.float32)) / 2).astype(np.uint8)
            rows.append(np.concatenate([_label(sim, f"SIM {name}"), _label(rl, "REAL"), _label(blend, "BLEND")], axis=1))
        composite = np.concatenate(rows, axis=0)
        Image.fromarray(composite).save(out_dir / "compare.png")
        files.append(out_dir / "compare.png")
    else:
        composite = np.concatenate([_label(imgs["top"], "SIM top"), _label(imgs["wrist"], "SIM wrist"), _label(imgs["overview"], "overview")], axis=1)
        Image.fromarray(composite).save(out_dir / "all.png")
        files.append(out_dir / "all.png")
    return files, composite


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="scene config yaml (default: ur10sim/scene_config.yaml)")
    ap.add_argument("--watch", action="store_true", help="re-render whenever the config file changes")
    ap.add_argument("--window", action="store_true", help="live tkinter window (needs $DISPLAY / ssh -X); implies --watch")
    ap.add_argument("--no-compare", action="store_true", help="skip the real-frame comparison")
    ap.add_argument("--gl", default="egl", choices=["egl", "osmesa", "glfw"])
    args = ap.parse_args()
    _setup_gl(args.gl)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_scene as B

    cfg_path = Path(args.config or B.DEFAULT_CONFIG)
    scene = None

    def load():
        try:
            return B.load_config(cfg_path)
        except Exception as e:  # noqa: BLE001
            print(f"[config error] {e}")
            return None

    def refresh(cfg):
        nonlocal scene
        t0 = time.time()
        if scene is None or scene.key != B.scene_key(cfg):
            if scene is not None:
                scene.close()
            scene = B.Scene(cfg)
            what = "rebuilt scene + physics"
        else:
            scene.update_cameras(cfg)
            what = "cameras only"
        scene.cfg = cfg
        files, comp = render_once(scene, cfg, B.REPO / cfg["render"]["out_dir"], not args.no_compare)
        print(f"[{time.strftime('%H:%M:%S')}] {what} in {time.time() - t0:.1f}s -> " + ", ".join(str(f.relative_to(B.REPO)) for f in files), flush=True)
        return comp

    cfg = load()
    if cfg is None:
        return 1
    atexit.register(lambda: scene and scene.close())
    comp = refresh(cfg)
    if not (args.watch or args.window):
        return 0

    mtime = cfg_path.stat().st_mtime
    if args.window:
        if not os.environ.get("DISPLAY"):
            print("--window needs $DISPLAY (ssh -X); falling back to --watch (PNG files only)")
            args.window = False
    if args.window:
        import tkinter as tk
        from PIL import Image, ImageTk
        root = tk.Tk()
        root.title(f"scene_view  ({cfg_path.name} -- save it to update)")
        lbl = tk.Label(root)
        lbl.pack()
        state = {"comp": comp, "mtime": mtime}

        def show():
            im = Image.fromarray(state["comp"])
            if im.width > 1500:
                im = im.resize((1500, int(im.height * 1500 / im.width)))
            ph = ImageTk.PhotoImage(im)
            lbl.configure(image=ph)
            lbl.image = ph

        def tick():
            mt = cfg_path.stat().st_mtime
            if mt != state["mtime"]:
                state["mtime"] = mt
                c = load()
                if c is not None:
                    try:
                        state["comp"] = refresh(c)
                        show()
                    except Exception as e:  # noqa: BLE001
                        print(f"[render error] {type(e).__name__}: {e}")
            root.after(400, tick)

        show()
        tick()
        root.mainloop()
        return 0
    print(f"watching {cfg_path} (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(0.4)
            mt = cfg_path.stat().st_mtime
            if mt != mtime:
                mtime = mt
                c = load()
                if c is not None:
                    try:
                        refresh(c)
                    except Exception as e:  # noqa: BLE001
                        print(f"[render error] {type(e).__name__}: {e}")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
