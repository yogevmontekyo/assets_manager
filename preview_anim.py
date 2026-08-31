#!/usr/bin/env python3
"""
Turn a pipeline output state's frames into something you can actually watch,
so centering drift and pixel-snap quality are obvious at a glance.

For every animation state it finds, it writes:

    <state>_anim.gif          final pixel-art frames, looping, NEAREST-scaled
    <state>_raw_anim.gif      the pre-downscale, post-centering frames
                              (isolates a CENTERING problem from a SNAP problem)
    <state>_onion.png         all frames mean-blended -- a well-centered anim
                              has a crisp head and only the limbs blur; a
                              mis-centered one blurs everywhere
    <state>_strip.png         filmstrip, one vertical guide line per frame

A magenta line marks the frame center in every view. If the character's head
/ torso slides off that line as the GIF plays, centering is off.

Usage:
    # every state under Output_Sprite_Sheet/
    python3 preview_anim.py

    # one state folder, slower, raw frames only
    python3 preview_anim.py Output_Sprite_Sheet/Player1 --kind raw --fps 6

    # a specific character
    python3 preview_anim.py Output_Sprite_Sheet/player1_front_sprite
"""
import argparse
import os
import re
import numpy as np
from PIL import Image, ImageDraw

BG_EXACT = (0, 255, 0)
_FRAME_RE = re.compile(r"^(?P<state>.+?)_(?P<idx>\d{2})(?P<suffix>|_raw)\.png$")

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(_PROJECT_DIR, "Output_Sprite_Sheet")


# --------------------------------------------------------------------------
def find_states(folder):
    """{state: {'final': [paths...], 'raw': [paths...]}} for one folder,
    frame lists sorted by index. Ignores *_preview.png and _*.png."""
    states = {}
    for fn in sorted(os.listdir(folder)):
        if fn.startswith("_") or fn.endswith("_preview.png"):
            continue
        m = _FRAME_RE.match(fn)
        if not m:
            continue
        key = "raw" if m["suffix"] == "_raw" else "final"
        st = states.setdefault(m["state"], {"final": [], "raw": []})
        st[key].append((int(m["idx"]), os.path.join(folder, fn)))
    for st in states.values():
        st["final"] = [p for _, p in sorted(st["final"])]
        st["raw"] = [p for _, p in sorted(st["raw"])]
    return states


def _auto_scale(w, h, target=360):
    return max(1, min(12, round(target / max(w, h))))


def _replace_bg(arr, mode, scale, bg=None):
    """arr is the already-NEAREST-upscaled RGB frame. Repaint background
    pixels -- given `bg` mask, else exact-green pixels."""
    if bg is None:
        bg = np.all(arr == BG_EXACT, axis=-1)
    if not bg.any() or mode == "green":
        return arr
    out = arr.copy()
    if mode == "gray":
        out[bg] = (74, 74, 74)
    elif mode == "magenta":
        out[bg] = (255, 0, 255)
    else:  # checker
        cell = 8 * scale
        yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
        light = (((yy // cell) + (xx // cell)) % 2).astype(bool)
        out[bg & light] = (96, 96, 96)
        out[bg & ~light] = (64, 64, 64)
    return out


def _load(paths, bg, scale):
    frames = []
    for p in paths:
        src = Image.open(p)
        big = (src.width * scale, src.height * scale)
        alpha = (np.array(src.getchannel("A").resize(big, Image.NEAREST)) == 0
                 if src.mode in ("RGBA", "LA", "PA") else None)
        rgb = np.array(src.convert("RGB").resize(big, Image.NEAREST))
        frames.append(_replace_bg(rgb, bg, scale, alpha))
    # pad to a common canvas (raw frames of a state already match; be safe)
    h = max(a.shape[0] for a in frames)
    w = max(a.shape[1] for a in frames)
    padded = []
    for a in frames:
        if a.shape[:2] != (h, w):
            q = np.zeros((h, w, 3), np.uint8)
            q[:] = a[0, 0]
            q[:a.shape[0], :a.shape[1]] = a
            a = q
        padded.append(a)
    return padded


def _guide(arr, idx=None):
    im = Image.fromarray(arr.copy())
    d = ImageDraw.Draw(im)
    cx = im.width // 2
    d.line([(cx, 0), (cx, im.height)], fill=(255, 0, 255), width=1)
    if idx is not None:
        d.text((2, 1), str(idx), fill=(255, 0, 255))
    return np.array(im)


def make_gif(frames, out_path, fps, guide):
    seq = [Image.fromarray(_guide(a, i) if guide else a)
           for i, a in enumerate(frames)]
    seq[0].save(out_path, save_all=True, append_images=seq[1:],
                duration=int(round(1000 / fps)), loop=0, disposal=2, optimize=False)


def make_onion(frames, out_path, guide):
    stack = np.stack(frames).astype(np.float32)
    blend = stack.mean(axis=0).astype(np.uint8)
    Image.fromarray(_guide(blend) if guide else blend).save(out_path)


def make_strip(frames, out_path, guide, gap=2):
    tiles = [_guide(a) if guide else a for a in frames]
    h = tiles[0].shape[0]
    sep = np.full((h, gap, 3), (20, 20, 20), np.uint8)
    row = []
    for i, t in enumerate(tiles):
        if i:
            row.append(sep)
        row.append(t)
    Image.fromarray(np.concatenate(row, axis=1)).save(out_path)


# --------------------------------------------------------------------------
def process_folder(folder, out_dir, kinds, fps, scale_arg, bg, guide,
                   onion, strip):
    states = find_states(folder)
    if not states:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for state, groups in sorted(states.items()):
        for kind in kinds:
            paths = groups["final" if kind == "final" else "raw"]
            if len(paths) < 1:
                continue
            probe = Image.open(paths[0])
            scale = (_auto_scale(*probe.size) if scale_arg == "auto"
                     else max(1, int(scale_arg)))
            frames = _load(paths, bg, scale)
            tag = "" if kind == "final" else "_raw"
            base = os.path.join(out_dir, f"{state}{tag}")
            if len(frames) > 1:
                make_gif(frames, base + "_anim.gif", fps, guide)
                print(f"  {state}{tag}: {len(frames)} frames @ {scale}x -> "
                      f"{os.path.basename(base)}_anim.gif")
            else:
                print(f"  {state}{tag}: 1 frame, no gif")
            if onion and len(frames) > 1:
                make_onion(frames, base + "_onion.png", guide)
            if strip:
                make_strip(frames, base + "_strip.png", guide)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=DEFAULT_ROOT,
                    help="A state folder with <state>_NN.png frames, or a "
                         "parent of such folders. Default: Output_Sprite_Sheet/")
    ap.add_argument("--kind", choices=["final", "raw", "both"], default="both",
                    help="Which frames to animate. raw = post-centering / "
                         "pre-downscale (isolates centering from snapping).")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--scale", default="auto",
                    help="Integer NEAREST upscale factor, or 'auto'.")
    ap.add_argument("--bg", choices=["checker", "gray", "green", "magenta"],
                    default="checker", help="How to paint the green backdrop.")
    ap.add_argument("--no-guide", action="store_true",
                    help="Omit the magenta center line + frame counter.")
    ap.add_argument("--no-onion", action="store_true",
                    help="Skip the mean-blend <state>_onion.png.")
    ap.add_argument("--strip", action="store_true",
                    help="Also write a <state>_strip.png filmstrip.")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write (default: alongside the frames).")
    args = ap.parse_args()

    kinds = ["final", "raw"] if args.kind == "both" else [args.kind]
    guide = not args.no_guide
    onion = not args.no_onion

    if not os.path.isdir(args.path):
        raise SystemExit(f"Not a folder: {args.path}")

    # a folder is either a state folder itself, or a parent of them
    targets = [args.path]
    if not find_states(args.path):
        targets = [os.path.join(args.path, d)
                   for d in sorted(os.listdir(args.path))
                   if os.path.isdir(os.path.join(args.path, d))]

    total = 0
    for folder in targets:
        out_dir = args.out_dir or folder
        if not os.path.isdir(folder):
            continue
        got = process_folder(folder, out_dir, kinds, args.fps, args.scale,
                             args.bg, guide, onion, args.strip)
        if got:
            print(f"{folder} -> {got} animation(s)")
        total += got
    if total == 0:
        raise SystemExit("No <state>_NN.png frame sets found. Point me at an "
                         "Output_Sprite_Sheet state folder.")
    print(f"\nDone. {total} animation(s).")


if __name__ == "__main__":
    main()
