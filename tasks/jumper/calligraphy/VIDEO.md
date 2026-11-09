# Making the contest video

A brief for an agent (or a person) on the machine where `write.py` and `render.py`
ran: from one run's films and the repository's pictures to a video to post. Every
recipe below was run on this repository's own output with the ffmpeg its
environment already has (2026-10-10).

## The brief

- **The contest.** A video longer than 15 s on X or YouTube, tagged `#Jumper
  #CrabRobot #OpenSourceRobot` and mentioning `@KingKong Robotics`; judged 60 %
  by the audience and 40 % by experts, on appearance, playfulness and usefulness.
- **The story.** A crab robot is handed a brush. It writes its own name, 跳跳
  (*tiàotiào*, "Jumper"), then 你好 ("hello"); the camera pulls back to the text;
  the robot looks at it, puts the brush down and jumps -- 跳 is "to jump" -- then
  runs laps around it, sideways like a crab, while -- as 地书 (water calligraphy
  on stone) does -- the water dries and the words are gone; and it bows to the
  dry stone, its brush lying before it. (A run made with `write.py --no-jump`
  keeps the brush and has no jump; one made with `--finale dance` dances beside
  the text instead, and the water dries after it.)
- **What to deliver**, in `<run>/edit/`:

  | file | what |
  |---|---|
  | `contest_16x9.mp4` | 1920 x 1080, 30 fps, H.264, yuv420p, 45-60 s |
  | `contest_9x16.mp4` | the same cut, 1080 x 1920, at most 60 s (YouTube Shorts) |
  | `thumbnail.png` | 1280 x 720 |
  | `post.txt` | the post's text: a line in English, one in Chinese, the hashtags, the mention, the link |

## What there is

- **The run**, `<run>` -- the directory `write.py --out` wrote, for the 跳跳 你好
  film `logs/calligraphy/u8df3-u8df3_u4f60-u597d/<name>`: `film.mp4` (following the
  character being written, then the reveal and the drying), `top.mp4` (overhead,
  fixed), `low.mp4` (low, across the ink, fixed), `jump.mp4` (the brush put
  down and the jump, close, 50 fps), `result.png` (the text from above, 1920 x
  1080), `ink.svg`, `log.npz`.
- **The moments**, in seconds of all three films, which share one clock:

  ```bash
  python tasks/jumper/calligraphy/tools/timeline.py <run>                      # to read
  python tasks/jumper/calligraphy/tools/timeline.py <run> --json > <run>/edit/timeline.json
  ```

  `characters[i].start/end`, `seams[]`,
  `outro.walk/put/jump/apex/jump_end/jump_clip/run/run_end/bow/still`,
  `film.text_end/dry/dry_end/end`. Cut from these, never from guessed times.
- **The pictures** in `docs/media/`: `jumper-hero-en.png` and `jumper-hero-zh.png`
  (2172 x 724 banners, the robot on black and green), `jumper.png`. The other GIFs
  there are Jumper's other skills, not this film; leave them out.
- **ffmpeg**: the environment's own, `python -c "import imageio_ffmpeg;
  print(imageio_ffmpeg.get_ffmpeg_exe())"`. It has `xfade`, `overlay`, `fade`,
  `gblur`, `setpts`, `loudnorm` -- and **no `drawtext`**, so every caption is a
  transparent PNG drawn with Pillow and overlaid.
- **A font with Chinese**: `fc-list :lang=zh file`; Noto Sans CJK
  (`sudo apt install fonts-noto-cjk`) or WenQuanYi Zen Hei. A font without the
  glyphs draws boxes -- check a frame (see the end).
- **Music**: none in the repository. Use a track only if the user gives one; never
  download one. Without it, deliver the video silent and say that YouTube's
  editor can add one from its Audio Library.

## First: render the films at 1920 x 1080

`render.py` renders at 960 x 720 by default. For a video to post:

```bash
MUJOCO_GL=egl python tasks/jumper/calligraphy/tools/render.py <run> --size 1920x1080 --no-gif
```

It overwrites `film.mp4`, `top.mp4`, `low.mp4` and `jump.mp4` in the run's
directory. With `MUJOCO_GL=egl` and an NVIDIA GPU it takes minutes; on a CPU,
hours -- then render `--shots film jump` only (`jump.mp4` is 6 s: 5 minutes at
960 x 720 on a 4-core CPU).

`jump.mp4` is the brush put down and the jump, low and close, at 50 fps -- the
rate `write.py` logs at -- so it can be slowed to half speed: the robot is off
the ground for under half a second. Slowing a 30 fps film with `minterpolate`
instead smeared the robot in the air (tested 2026-10-11).

## The cut

About 56 s. Speeds are how much faster than real time; times come from the
timeline. On a 跳跳 你好 run with the jump (2026-10-11) these rows add up to 63.7 s
with the run at 1x, 56.5 s at 1.5x, cross-fades taken off: over 60 s, the 9:16
version is not a Short.

| # | shot | source, from -> to | speed | caption (English / 中文) |
|---|---|---|---|---|
| 1 | title | `jumper-hero-en.png` on the stone's colour, 2.5 s | -- | -- |
| 2 | hook | `low.mp4`, `writing.start - 0.5` -> `+ 4` | 1 | Can a crab robot write its own name? / 螃蟹机器人能写出自己的名字吗？ |
| 3 | 跳 | `film.mp4`, `characters[0]` | 14 | Its name: 跳跳 (tiàotiào) -- "Jumper" / 它的名字：跳跳 |
| 4 | 跳 | `top.mp4`, `characters[1]` | 16 | -- |
| 5 | 你 | `film.mp4`, `characters[2]` | 14 | ...and a greeting: 你好 -- hello / 再写一句：你好 |
| 6 | 好 | `film.mp4`, `characters[3]` | 14 | -- |
| 7 | it looks | `film.mp4`, `outro.walk` -> `outro.put - 0.5` | 1.5 | -- |
| 8 | the jump | `jump.mp4`, from 0.5 s to `outro.apex - 0.8 - outro.jump_clip` (the brush put down), then to `outro.apex + 0.8 - outro.jump_clip` | 1, then 0.5 | 跳 means "to jump". / 跳跳，跳！ |
| 9 | the run | `film.mp4`, `outro.run` -> `outro.run_end`; cut in 2 s of `low.mp4` where it runs past the camera | 1.5 | 地书: written with water on stone. It dries, and is gone. / 地书：以水为墨，干了便消失 |
| 10 | the bow | `film.mp4`, `outro.bow - 0.5` -> `outro.still + 1` | 1 | A bow to the dry stone. / 向石板鞠躬 |
| 11 | result | `result.png`, 2.5 s | -- | -- |
| 12 | end card | `jumper-hero-zh.png`, 3.5 s | -- | Open source, no training needed -- github.com/AgusBM/jumper (branch calligraphy) / 开源 · 无需训练 · #Jumper #CrabRobot #OpenSourceRobot |

- Cross-fade every cut by 0.5 s, the title and the end card by 0.8 s.
- **Never slow down within 1.5 s of a seam** (`seams[]`): the arm comes back down to
  finish the stroke there. In the 跳跳 你好 runs, finishing stroke 30 after its
  seam (247.3 s in the test run; the timeline gives this run's) the claw brushes
  the front of the trunk for half a second: at 14x a blink, in real time plain to
  see.
- Captions: white, a dark outline, at the bottom, 72 px at 1080 px high, two
  lines (English over Chinese), at least 2.5 s on screen, faded 0.4 s in and out.
- The 9:16 version is the same cut over a blurred, enlarged copy of itself, the
  captions drawn for 1080 x 1920 rather than scaled.

## Recipes

Tested on a 无 run (`render.py` output, 960 x 720, padded to 16:9): a 40 s 16:9
cut and its 9:16 version in 70 s on a 4-core CPU. Adapt `W, H` and the paths.

```python
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1920, 1080, 30
ENC = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
       "-r", str(FPS)]
# Anything not 16:9 (the banners, a 960 x 720 film) is padded with the stone's colour.
FIT = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
       f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0xd8d2c8,setsar=1")


def ff(*a):
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", *map(str, a)], check=True)


def seg(src, a, b, speed, out):
    """Seconds a to b of src, `speed` times faster, silent, W x H."""
    ff("-ss", a, "-to", b, "-i", src, "-an",
       "-vf", f"setpts=PTS/{speed},fps={FPS},{FIT}", *ENC, out)
    return out


def caption(text, out, font, size=72):
    """A transparent W x H PNG, the text centred near the bottom."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font, size)
    box = d.multiline_textbbox((0, 0), text, font=f, align="center", spacing=12)
    x, y = (W - (box[2] - box[0])) / 2, H - (box[3] - box[1]) - 80
    d.multiline_text((x, y), text, font=f, fill="white", align="center", spacing=12,
                     stroke_width=5, stroke_fill=(0, 0, 0, 200))
    im.save(out)
    return out


def overlay(clip, png, start, dur, out):
    """The caption over the clip from `start` for `dur` seconds, faded in and out."""
    fc = (f"[1:v]format=rgba,fade=t=in:st={start}:d=0.4:alpha=1,"
          f"fade=t=out:st={start + dur - 0.4}:d=0.4:alpha=1[c];"
          f"[0:v][c]overlay=0:0:shortest=1")
    ff("-i", clip, "-loop", "1", "-i", png, "-filter_complex", fc, *ENC, out)
    return out


def still(png, dur, out):
    ff("-loop", "1", "-t", dur, "-i", png, "-vf", f"{FIT},fps={FPS}", *ENC, out)
    return out


def length(path):
    err = subprocess.run([FF, "-hide_banner", "-i", path], capture_output=True,
                         text=True).stderr
    h, m, s = err.split("Duration: ")[1].split(",")[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def xfade(clips, out, d=0.5):
    """The clips one after another, each cross-faded into the next over d seconds."""
    args, fc, last, off = [], [], "0:v", 0.0
    for c in clips:
        args += ["-i", c]
    for i in range(1, len(clips)):
        off += length(clips[i - 1]) - d
        fc.append(f"[{last}][{i}:v]xfade=transition=fade:duration={d}:offset={off:.3f}[v{i}]")
        last = f"v{i}"
    ff(*args, "-filter_complex", ";".join(fc), "-map", f"[{last}]", *ENC, out)
    return out


def vertical(src, out):
    """9:16: the 16:9 picture across the middle, over a blurred, enlarged copy."""
    ff("-i", src, "-filter_complex",
       "[0:v]split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,"
       "crop=1080:1920,gblur=sigma=40[bg];[b]scale=1080:-2[fg];"
       "[bg][fg]overlay=0:(H-h)/2", *ENC, out)
    return out
```

- A speed that changes inside a shot (the bow at 1x within the dance at 1.5x) is two
  segments joined with a 0.2 s cross-fade.
- The jump at half speed: `seg(jump, a, b, 0.5, out)` -- `setpts=PTS/0.5` from the
  50 fps `jump.mp4`, no interpolation. Its times are its own: a moment `T` of the
  timeline is `T - outro.jump_clip` in it.
- For the 9:16 version, build the cut without captions, make it vertical, and
  overlay captions drawn at 1080 x 1920 (two lines, 64 px, 300 px from the bottom).
- Music, if given: `-i music.mp3 -af "afade=t=in:d=1,afade=t=out:st=<T-2>:d=2,loudnorm=I=-14"
  -shortest -c:a aac -b:a 192k`.

## The thumbnail and the post

- `thumbnail.png`: `result.png` scaled to 1280 x 720, with "A crab robot learned
  calligraphy" / "螃蟹机器人学会了书法" in the upper third (the same caption style,
  96 px).
- `post.txt`:

  ```text
  A crab robot writes its own name -- 跳跳, "Jumper" -- and says 你好 in water calligraphy (地书). Open source, nothing trained.
  螃蟹机器人跳跳用水在石板上写下自己的名字和“你好”，干了便消失。开源，无需训练。
  #Jumper #CrabRobot #OpenSourceRobot @KingKong Robotics
  https://github.com/AgusBM/jumper/tree/calligraphy
  ```

## Before handing it over

- Check both files with `ffmpeg -i`: duration (16:9 45-60 s, 9:16 at most 60 s),
  1920 x 1080 and 1080 x 1920, 30 fps.
- Take a frame inside every caption (`ffmpeg -ss <t> -i <file> -frames:v 1 f.png`)
  and look at it: every Chinese character drawn (no boxes), nothing cut at the
  edges, the caption clear of the brush in the hook.
- Watch the 9:16 version's first 3 s: they decide whether anyone keeps watching.
- Show the user those frames and the two durations; ask before posting anything.
