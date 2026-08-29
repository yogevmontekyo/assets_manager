#!/usr/bin/env python3
"""
Full sprite pipeline: sheet -> drift-corrected frames -> clean pixel art.

Combines extract_frames.py (row/frame detection, horizontal drift
correction, vertical-motion-preserving crop) and pixelate.py
(coverage-aware downscale with no background bleed + dither-free palette
quantization) into one run.

Works for both cases with no special-casing:
  - A single full-body character on a flat-chroma background -> detected
    as 1 row / 1 frame.
  - A multi-row animation sheet (one state per row, frames left to right)
    -> each row becomes an animation state, each column a frame.

All frames across the whole sheet share ONE locked color palette, so an
animation's frames don't flicker/drift in color from frame to frame.

Usage:
    python3 main.py input.png --out-dir out --target-size 64 --n-colors 12 \\
        --state-names walk jump

Outputs (in --out-dir):
    <state>_<NN>_raw.png        full-res, padded, drift-corrected crop
    <state>_<NN>.png            final clean pixel-art frame (target_size^2)
    <state>_<NN>_preview.png    8x nearest-neighbor upscale for inspection
    _contact_sheet.png          detected row/frame boundaries overlaid on source
    _palette_swatch.png         the shared locked palette
    _report.txt                 verification results for every frame
"""
import argparse
import os
import numpy as np
from PIL import Image

import extract_frames as ef
import pixelate as px


def make_palette_swatch(palette, out_path, swatch_size=48):
    n = len(palette)
    img = Image.new("RGB", (swatch_size * n, swatch_size), (255, 255, 255))
    arr = np.array(img)
    for i, color in enumerate(palette):
        arr[:, i * swatch_size:(i + 1) * swatch_size] = color
    Image.fromarray(arr).save(out_path)


def run_pipeline(input_path, out_dir, target_size=64, n_colors=12,
                  h_pad_frac=0.15, v_pad_frac=0.15, coverage_thresh=0.35,
                  state_names=None):
    os.makedirs(out_dir, exist_ok=True)
    report_lines = []

    # ---- Stage 1: detect rows (states) and columns (frames) ----
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img)
    is_char = ~ef.is_background(arr)

    row_bands = ef.detect_rows(is_char)
    col_bands_per_row = [ef.detect_cols(is_char, rtop, rbot) for rtop, rbot in row_bands]

    names = state_names or [f"state{i}" for i in range(len(row_bands))]
    if len(names) != len(row_bands):
        print(f"WARNING: {len(names)} state names given but {len(row_bands)} "
              f"rows detected; using generic names instead.")
        names = [f"state{i}" for i in range(len(row_bands))]

    ef.make_contact_sheet(img, row_bands, col_bands_per_row,
                           os.path.join(out_dir, "_contact_sheet.png"))
    report_lines.append(f"Detected {len(row_bands)} state(s): "
                         f"{list(zip(names, [len(c) for c in col_bands_per_row]))}")

    # ---- Stage 2: extract every frame at full resolution, drift-corrected ----
    all_frames = []  # list of (state_name, frame_idx, PIL.Image)
    for (rtop, rbot), col_bands, name in zip(row_bands, col_bands_per_row, names):
        raw_frames = ef.extract_state(arr, is_char, rtop, rbot, col_bands,
                                       h_pad_frac=h_pad_frac, v_pad_frac=v_pad_frac,
                                       state_name=name)
        for idx, frame_arr in enumerate(raw_frames):
            frame_img = Image.fromarray(frame_arr, "RGB")
            frame_img.save(os.path.join(out_dir, f"{name}_{idx:02d}_raw.png"))
            all_frames.append((name, idx, frame_img))

    # ---- Stage 3: coverage-aware downscale every frame (no quantize yet) ----
    # target_size is the output HEIGHT; width is derived per-state from that
    # state's actual padded-crop aspect ratio, so proportions are preserved
    # instead of being force-squashed into a square.
    downscaled = []  # (name, idx, downscaled_img)
    fg_pixel_pools = []
    state_target_dims = {}  # name -> (tw, th)
    for name, idx, frame_img in all_frames:
        if name not in state_target_dims:
            src_w, src_h = frame_img.size
            th = target_size
            tw = max(1, round(target_size * src_w / src_h))
            state_target_dims[name] = (tw, th)
            report_lines.append(f"  {name}: source aspect {src_w}x{src_h} "
                                 f"({src_w/src_h:.3f}) -> output {tw}x{th} "
                                 f"({tw/th:.3f})")
        down = px.coverage_aware_downscale(frame_img, state_target_dims[name], coverage_thresh)
        downscaled.append((name, idx, down))
        fg_pixel_pools.append(px.foreground_pixels_of_downscaled(down))

    # ---- Stage 4: build ONE shared palette across every frame in the sheet ----
    shared_palette = px.build_shared_palette(fg_pixel_pools, n_colors=n_colors)
    make_palette_swatch(shared_palette, os.path.join(out_dir, "_palette_swatch.png"))
    report_lines.append(f"Shared palette ({n_colors} colors) built from "
                         f"{len(downscaled)} frame(s): "
                         f"{[tuple(int(v) for v in c) for c in shared_palette]}")

    # ---- Stage 5: quantize every frame against the shared palette ----
    for name, idx, down in downscaled:
        final_img, _ = px.quantize_locked(down, fixed_palette_rgb=shared_palette)
        out_path = os.path.join(out_dir, f"{name}_{idx:02d}.png")
        final_img.save(out_path)
        fw, fh = final_img.size
        final_img.resize((fw * 8, fh * 8), Image.NEAREST).save(
            os.path.join(out_dir, f"{name}_{idx:02d}_preview.png"))

        check = px.verify_clean(final_img)
        status = "OK" if check["suspected_bleed_colors"] == 0 else "FAIL"
        report_lines.append(f"  {name}_{idx:02d}: colors={check['total_colors']} "
                             f"bleed={check['suspected_bleed_colors']} [{status}]")

    report_path = os.path.join(out_dir, "_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nDone. Outputs written to {out_dir}/")
    return report_lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Path to source sheet or single-character PNG")
    ap.add_argument("--out-dir", default="./pipeline_out")
    ap.add_argument("--target-size", type=int, default=64,
                     help="Output sprite grid HEIGHT in pixels; width is "
                          "derived per-state to preserve the source crop's "
                          "aspect ratio (not forced square)")
    ap.add_argument("--n-colors", type=int, default=12,
                     help="Size of the shared locked palette (excl. background)")
    ap.add_argument("--h-pad-frac", type=float, default=0.15)
    ap.add_argument("--v-pad-frac", type=float, default=0.15)
    ap.add_argument("--coverage-thresh", type=float, default=0.35,
                     help="Min fraction of a source block that must be "
                          "foreground for the output pixel to count as character")
    ap.add_argument("--state-names", nargs="*", default=None)
    args = ap.parse_args()

    run_pipeline(args.input, args.out_dir, args.target_size, args.n_colors,
                 args.h_pad_frac, args.v_pad_frac, args.coverage_thresh,
                 args.state_names)