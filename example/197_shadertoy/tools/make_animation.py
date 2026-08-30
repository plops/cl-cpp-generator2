#!/usr/bin/env python3
"""Render a stretch of the earth-shader camera tour and assemble an animated GIF.

Each frame is produced by the headless EGL runner at an explicit --time offset
(the state pass is a pure function of iTime, so frames are independent).

usage: tools/make_animation.py [out.gif] [t0] [t1] [n_frames] [width] [height]
"""

import os
import subprocess
import sys
import tempfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNNER = os.path.join(ROOT, "headless_gpu_runner")
SHADERS = os.path.join(ROOT, "vulkan-shadertoy-x11/launcher/shaders/earth")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "earth_tour.gif")
    t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    t1 = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 48
    w = int(sys.argv[5]) if len(sys.argv) > 5 else 480
    h = int(sys.argv[6]) if len(sys.argv) > 6 else 300

    if w < 256 or h < 176:
        sys.exit("the state pass needs at least 256x176 texels for the data block")

    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            t = t0 + (t1 - t0) * i / n
            png = os.path.join(tmp, f"f{i:04d}.png")
            subprocess.run([RUNNER, "--res", str(w), str(h), "--frames", "2",
                            "--time", f"{t:.4f}", "--shader-dir", SHADERS,
                            "--screenshot", png],
                           check=True, stdout=subprocess.DEVNULL)
            frames.append(Image.open(png).convert("RGB").copy())
            print(f"\r  frame {i + 1}/{n} (t = {t:5.2f}s)", end="", flush=True)
    print()

    ms = int(round(1000.0 * (t1 - t0) / n))
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=192) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=ms, loop=0, optimize=True)
    print(f"wrote {out}  ({n} frames, {ms} ms/frame, {os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
