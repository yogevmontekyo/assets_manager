#!/usr/bin/env python3
"""
Synthesize a sprite sheet with a KNOWN-FIXED body center, for regression-
testing centering and pixel snapping.

The character's torso / head / legs sit at exactly the same x in every
frame. Around that rigid core, the things that fool a naive centerer move a
lot frame to frame: a big cape that flares left and right, a bobbing hair
tuft, a swinging arm, and (row 2) a full vertical jump arc. So after the
pipeline runs, ANY horizontal wobble of the body is centering error --
there is no real drift to excuse it.

The torso is painted a unique flat blue that nothing else in the sprite
uses, so `--verify` can measure the body's position directly, immune to the
cape/hair.

Workflow:
    python3 make_test_sheet.py                       # writes the sheet
    python3 main.py Input_Generated_Character/_test_sheet.png \\
        --state-names walk jump
    python3 preview_anim.py Output_Sprite_Sheet/_test_sheet   # watch it
    python3 make_test_sheet.py --verify Output_Sprite_Sheet/_test_sheet

`--verify` prints, per state, the spread of the blue torso's centroid x
across the final frames (in output pixels) and PASS/FAIL against --tol.
"""
import argparse
import math
import os
import numpy as np
from PIL import Image, ImageDraw

BG = (0, 255, 0)
BODY = (40, 90, 210)      # unique flat blue -- the measurement target
SKIN = (240, 200, 160)
HAIR = (110, 70, 45)
CAPE = (150, 40, 40)
LEG = (70, 70, 90)

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(_PROJECT_DIR, "Input_Generated_Character", "_test_sheet.png")

CELL_W, CELL_H = 220, 380
GUTTER = 16
MARGIN = 20
_FEET_INSET = 30       # ground line = cell bottom - this
_JUMP_RISE = 60        # peak lift; keep < top slack so the head stays in-cell


def _rect(dr, x0, y0, x1, y1, fill):
    dr.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=fill)


def _draw_character(dr, cx, feet_y, t, jump=False):
    """cx = fixed body center x. feet_y = ground line for this frame. t in
    [0,1) is the animation phase."""
    s = math.sin(2 * math.pi * t)
    s2 = math.sin(4 * math.pi * t)

    # ---- cape: a big quad that flares side to side (the main distractor) ----
    flare = 46 * s
    cape_top = feet_y - 210
    dr.polygon([(cx - 30, cape_top), (cx + 30, cape_top),
                (cx + 34 + flare, feet_y - 70),
                (cx - 34 + flare, feet_y - 70)], fill=CAPE)

    # ---- legs: scissor for walk, tuck for jump ----
    if jump:
        _rect(dr, cx - 22, feet_y - 60, cx - 4, feet_y, LEG)
        _rect(dr, cx + 4, feet_y - 66, cx + 22, feet_y - 6, LEG)
    else:
        off = int(18 * s)
        _rect(dr, cx - 22 - off, feet_y - 74, cx - 4 - off, feet_y, LEG)
        _rect(dr, cx + 4 + off, feet_y - 74, cx + 22 + off, feet_y, LEG)

    # ---- torso: FIXED. this is what --verify measures ----
    _rect(dr, cx - 26, feet_y - 150, cx + 26, feet_y - 70, BODY)

    # ---- swinging arm (distractor) ----
    arm = int(30 * s2)
    _rect(dr, cx + 22, feet_y - 145, cx + 40 + arm, feet_y - 122, SKIN)
    _rect(dr, cx - 40, feet_y - 145, cx - 22, feet_y - 120, SKIN)

    # ---- head: FIXED x ----
    hy = feet_y - 150
    dr.ellipse([cx - 20, hy - 40, cx + 20, hy], fill=SKIN)

    # ---- hair tuft: bobs left/right (distractor) ----
    hair_dx = int(6 * s)
    dr.polygon([(cx - 20, hy - 30), (cx + 20, hy - 34),
                (cx + 8 + hair_dx, hy - 54), (cx - 12 + hair_dx, hy - 50)],
               fill=HAIR)


def build_sheet(n_frames=6, out_path=DEFAULT_OUT):
    states = [("walk", False), ("jump", True)]
    W = MARGIN * 2 + n_frames * CELL_W + (n_frames - 1) * GUTTER
    H = MARGIN * 2 + len(states) * CELL_H + (len(states) - 1) * GUTTER
    img = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(img)

    truth = {}
    for row, (name, jump) in enumerate(states):
        cell_top = MARGIN + row * (CELL_H + GUTTER)
        centers = []
        for k in range(n_frames):
            cell_left = MARGIN + k * (CELL_W + GUTTER)
            cx = cell_left + CELL_W // 2          # <-- constant across frames
            t = k / n_frames
            feet_y = cell_top + CELL_H - _FEET_INSET
            if jump:
                feet_y -= int(_JUMP_RISE * math.sin(math.pi * k / (n_frames - 1)))
            _draw_character(dr, cx, feet_y, t, jump=jump)
            centers.append(cx)
        truth[name] = centers

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    print(f"wrote {out_path}  ({W}x{H}, {n_frames} frames x {len(states)} states)")
    print("ground-truth body-center x per frame (source px):")
    for name, cs in truth.items():
        print(f"  {name}: {cs}  (constant within the state)")
    return out_path


# --------------------------------------------------------------------------
def _blue_centroid_x(arr):
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    m = (b > r + 40) & (b > g + 40)
    xs = np.nonzero(m)[1]
    return float(xs.mean()) if len(xs) else None


def verify(out_state_dir, tol=1.0):
    import re
    rx = re.compile(r"^(?P<st>.+?)_(?P<i>\d{2})\.png$")
    groups = {}
    for fn in sorted(os.listdir(out_state_dir)):
        if fn.startswith("_") or fn.endswith("_preview.png") or fn.endswith("_raw.png"):
            continue
        m = rx.match(fn)
        if m:
            groups.setdefault(m["st"], []).append(os.path.join(out_state_dir, fn))
    if not groups:
        raise SystemExit(f"no <state>_NN.png frames in {out_state_dir}")

    print(f"verify {out_state_dir}  (tol {tol:.2f}px, output-frame pixels)")
    all_ok = True
    for st, paths in sorted(groups.items()):
        cxs = []
        for p in sorted(paths):
            c = _blue_centroid_x(np.array(Image.open(p).convert("RGB")))
            if c is not None:
                cxs.append(c)
        if not cxs:
            print(f"  {st}: no blue torso found (n-colors too low?) -- SKIP")
            continue
        spread = max(cxs) - min(cxs)
        ok = spread <= tol
        all_ok &= ok
        print(f"  {st}: n={len(cxs)}  body-center spread={spread:.2f}px  "
              f"std={np.std(cxs):.2f}  [{'PASS' if ok else 'FAIL'}]")
        if not ok:
            print(f"       per-frame x: {[round(c, 1) for c in cxs]}")
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=6, help="frames per state")
    ap.add_argument("--out", default=DEFAULT_OUT, help="sheet path to write")
    ap.add_argument("--verify", metavar="STATE_DIR", default=None,
                    help="instead of writing: measure body-center drift in a "
                         "pipeline output folder (e.g. Output_Sprite_Sheet/_test_sheet)")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="max allowed body-center spread in output px (--verify)")
    args = ap.parse_args()

    if args.verify:
        ok = verify(args.verify, args.tol)
        raise SystemExit(0 if ok else 1)
    build_sheet(args.frames, args.out)
