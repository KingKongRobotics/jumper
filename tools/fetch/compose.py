"""Lay the depth inset and the sensor overlay over a take.

    .venv/bin/python tools/fetch/compose.py out/fetch/seg4a --cam low

Reads <take>/<take>-<cam>.mp4 (or <take>/<take>.mp4 with no --cam), the depth
frames in <take>/pip and the overlay frames in <take>/hud, and writes
<take>/<take>-<cam>-hud.mp4. Frame rates are the take's (30 fps everywhere).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

MAP_FPS = 5.0   # auto.py writes the map's picture at this rate (MAP_DT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("take", type=Path)
    ap.add_argument("--cam", default="")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    name = a.take.name
    src = a.take / (f"{name}-{a.cam}.mp4" if a.cam else f"{name}.mp4")
    out = a.take / (f"{name}-{a.cam}-hud.mp4" if a.cam else f"{name}-hud.mp4")
    pip, hud = a.take / "pip", a.take / "hud"
    inputs = ["-i", str(src)]
    chain, last = [], "[0]"
    if pip.is_dir():
        inputs += ["-framerate", str(a.fps), "-i", str(pip / "pip_%05d.png")]
        chain.append("[1]scale=432:336:flags=neighbor,pad=iw+8:ih+8:4:4:white[p]")
        chain.append(f"{last}[p]overlay=W-w-40:40:shortest=1[v1]"); last = "[v1]"
    k = 1 + int(pip.is_dir())
    if hud.is_dir():
        inputs += ["-framerate", str(a.fps), "-i", str(hud / "hud_%05d.png")]
        chain.append(f"{last}[{k}]overlay=40:H-h-40:shortest=1[v2]"); last = "[v2]"
        k += 1
    if (a.take / "map").is_dir():
        # the occupancy map (mapper.py), written at MAP_FPS by auto.py, bottom right
        inputs += ["-framerate", str(MAP_FPS), "-i", str(a.take / "map" / "map_%05d.png")]
        chain.append(f"[{k}]scale=360:360:flags=neighbor,pad=iw+8:ih+8:4:4:white[m]")
        chain.append(f"{last}[m]overlay=40:40:shortest=1[v3]"); last = "[v3]"   # top left: bottom right hid the cup (Andrew)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(chain), "-map", last,
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    print(out)


if __name__ == "__main__":
    main()
