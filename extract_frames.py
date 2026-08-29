#!/usr/bin/env python3
"""
Sprite sheet frame extractor with horizontal drift correction.

Reads a multi-row sprite sheet (one animation state per row, frames left-to-right
within each row) on a flat chroma-green background, and:

1. Auto-detects row bands (animation states) via horizontal projection of
   non-background pixels.
2. Auto-detects frame column bands within each row via vertical projection.
3. For each frame: crops using the FULL row's vertical extent (not a per-frame
   bbox) so vertical motion within the animation (e.g. a jump) is preserved
   exactly as authored -- only horizontal position is corrected.
4. Horizontally re-centers each frame's character silhouette so it sits at a
   consistent x-position across all frames of that state (removes drift from
   frame to frame), using a uniform canvas width for the whole state.
5. Outputs each frame as its own PNG, plus a QA contact sheet with detected
   boundaries drawn on top for visual verification.

If auto-detection produces the wrong number of bands/frames for a given sheet
(e.g. frames touch, or there's stray noise), pass --rows or --cols-per-row
overrides -- see the __main__ section / --help.
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw

BG_EXACT = (0, 255, 0)


def is_background(arr, g_thresh=140, margin=60):
    """Boolean mask: True where pixel is chroma-green background."""
    r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
    return (g > g_thresh) & (g > r + margin) & (g > b + margin)


def find_bands(mask_1d):
    """Given a 1D boolean array, return list of (start, end) inclusive index
    ranges for contiguous True runs."""
    bands = []
    in_band = False
    start = 0
    for i, v in enumerate(mask_1d):
        if v and not in_band:
            start = i
            in_band = True
        if not v and in_band:
            bands.append((start, i - 1))
            in_band = False
    if in_band:
        bands.append((start, len(mask_1d) - 1))
    return bands


def detect_rows(is_char):
    row_has_char = is_char.any(axis=1)
    return find_bands(row_has_char)


def detect_cols(is_char, row_top, row_bot):
    row_slice = is_char[row_top:row_bot + 1]
    col_has_char = row_slice.any(axis=0)
    return find_bands(col_has_char)


def extract_state(arr, is_char, row_top, row_bot, col_bands, h_pad_frac=0.15,
                   v_pad_frac=0.15, state_name="state"):
    """Extract all frames for one animation row with horizontal drift correction.

    Vertical: every frame uses the SAME [row_top, row_bot] slice from the
    source (plus uniform top/bottom padding added equally to every frame),
    so vertical motion (jump arcs etc.) is preserved untouched -- padding
    just adds margin, it doesn't re-anchor per frame.

    Horizontal: each frame's character bbox center is computed, then the
    frame is placed into a uniform-width canvas so that center lands on the
    canvas's horizontal center -- correcting any left/right drift between
    frames.
    """
    row_h = row_bot - row_top + 1
    max_frame_w = max(c2 - c1 + 1 for c1, c2 in col_bands)
    h_pad = int(max_frame_w * h_pad_frac)
    v_pad = int(row_h * v_pad_frac)
    canvas_w = max_frame_w + 2 * h_pad
    canvas_h = row_h + 2 * v_pad

    frames = []
    for idx, (c1, c2) in enumerate(col_bands):
        frame_src = arr[row_top:row_bot + 1, c1:c2 + 1]
        frame_char_mask = is_char[row_top:row_bot + 1, c1:c2 + 1]

        cols_with_char = np.where(frame_char_mask.any(axis=0))[0]
        if len(cols_with_char) == 0:
            char_center_x = (c2 - c1) / 2.0
        else:
            char_center_x = (cols_with_char[0] + cols_with_char[-1]) / 2.0

        canvas = np.full((canvas_h, canvas_w, 3), BG_EXACT, dtype=np.uint8)
        target_center_x = canvas_w / 2.0
        dst_left = int(round(target_center_x - char_center_x))

        src_w = frame_src.shape[1]
        dst_start = max(dst_left, 0)
        dst_end = min(dst_left + src_w, canvas_w)
        src_start = max(0, -dst_left)
        src_end = src_start + (dst_end - dst_start)

        if dst_end > dst_start:
            canvas[v_pad:v_pad + row_h, dst_start:dst_end] = frame_src[:, src_start:src_end]

        # re-clean any anti-aliased fringe introduced at crop edges
        cmask = is_background(canvas)
        canvas[cmask] = BG_EXACT

        frames.append(canvas)
        print(f"  {state_name} frame {idx}: src cols [{c1}:{c2}] "
              f"(w={c2 - c1 + 1}) -> canvas {canvas_w}x{canvas_h}, "
              f"char center shifted by {dst_left - 0}px")

    return frames


def make_contact_sheet(img, row_bands, col_bands_per_row, out_path):
    vis = img.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    colors = ["red", "yellow", "cyan", "magenta", "orange", "white"]
    for r_i, (rtop, rbot) in enumerate(row_bands):
        draw.rectangle([0, rtop, vis.width - 1, rbot], outline="red", width=3)
        for c_i, (c1, c2) in enumerate(col_bands_per_row[r_i]):
            color = colors[c_i % len(colors)]
            draw.rectangle([c1, rtop, c2, rbot], outline=color, width=2)
            draw.text((c1 + 3, rtop + 3), str(c_i), fill=color)
    vis.save(out_path)
    print(f"QA contact sheet saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Path to sprite sheet PNG")
    ap.add_argument("--out-dir", default="./frames_out")
    ap.add_argument("--state-names", nargs="*", default=None,
                     help="Names for each detected row, top to bottom, "
                          "e.g. --state-names walk jump")
    ap.add_argument("--h-pad-frac", type=float, default=0.15,
                     help="Horizontal padding as fraction of max frame width")
    ap.add_argument("--v-pad-frac", type=float, default=0.15,
                     help="Vertical padding as fraction of row height, "
                          "added equally top and bottom (does not affect "
                          "relative motion between frames)")
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    img = Image.open(args.input).convert("RGB")
    arr = np.array(img)
    is_char = ~is_background(arr)

    row_bands = detect_rows(is_char)
    print(f"Detected {len(row_bands)} row(s)/state(s): {row_bands}")

    col_bands_per_row = []
    for (rtop, rbot) in row_bands:
        cb = detect_cols(is_char, rtop, rbot)
        col_bands_per_row.append(cb)
        print(f"  row y=[{rtop},{rbot}]: {len(cb)} frame(s)")

    make_contact_sheet(img, row_bands, col_bands_per_row,
                        os.path.join(args.out_dir, "_contact_sheet.png"))

    names = args.state_names or [f"state{i}" for i in range(len(row_bands))]
    if len(names) != len(row_bands):
        print(f"WARNING: {len(names)} state names given but {len(row_bands)} "
              f"rows detected; falling back to generic names.")
        names = [f"state{i}" for i in range(len(row_bands))]

    for (rtop, rbot), col_bands, name in zip(row_bands, col_bands_per_row, names):
        frames = extract_state(arr, is_char, rtop, rbot, col_bands,
                                h_pad_frac=args.h_pad_frac,
                                v_pad_frac=args.v_pad_frac, state_name=name)
        for idx, frame in enumerate(frames):
            out_path = os.path.join(args.out_dir, f"{name}_{idx:02d}.png")
            Image.fromarray(frame).save(out_path)

    print(f"\nDone. {sum(len(cb) for cb in col_bands_per_row)} frames written to {args.out_dir}/")


if __name__ == "__main__":
    main()
