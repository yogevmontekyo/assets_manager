#!/usr/bin/env python3
"""
Batch entrypoint for snap_background.py -- the background counterpart to
main.py.

Runs the pixel-grid snap + upscale on every image in the project's
background_image/ folder (or one file / folder you name) and writes the
results to background_image/snapped/.

snap_background.py does the actual work: detect the AI image's true native
pixel grid, reduce to it with centre-cell median sampling (no seam bleed),
optionally flatten to an N-colour palette, then NEAREST-upscale to a target
resolution. Multi-panel comparison sheets are split and each panel snapped
on its own. See that file's docstring for the method.

Usage:
    # snap every image in background_image/ to 1920x1080 (crop-to-fill)
    python3 main_background.py

    # a specific file, fit the whole panel and pad instead of cropping
    python3 main_background.py background_image/intro_background.jpeg --fit pad

    # just NEAREST-upscale the snapped native image 4x, nothing else
    python3 main_background.py --scale 4

Output file names (in --out-dir, one flat folder -- names are prefixed with
the source stem, and with _<i> for panel i of a multi-panel sheet):
    <stem>[_<i>]_native_<C>x<R>.png     the image at its true pixel grid
    <stem>[_<i>]_<W>x<H>.png            upscaled to the target / --scale
    <stem>[_<i>]_native_preview4x.png   4x NEAREST preview of the native
"""
import argparse
import os

import snap_background as sb

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN_DIR = os.path.join(_PROJECT_DIR, "background_image")
DEFAULT_OUT_DIR = os.path.join(DEFAULT_IN_DIR, "snapped")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def resolve_inputs(input_arg):
    """input_arg None -> every image directly in background_image/ (the
    snapped/ output subfolder is not descended into); a folder -> every
    image in it; a file -> just that file."""
    if input_arg and os.path.isfile(input_arg):
        return [input_arg]

    folder = input_arg or DEFAULT_IN_DIR
    if not os.path.isdir(folder):
        raise SystemExit(f"Input not found: {folder}")
    imgs = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMAGE_EXTS)
                  and os.path.isfile(os.path.join(folder, f)))
    if not imgs:
        raise SystemExit(f"No images ({', '.join(IMAGE_EXTS)}) found in {folder}")
    return imgs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default=None,
                    help="A background image, or a folder of them. Omit to "
                         "process every image in the project's "
                         "background_image/ folder.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="Output folder (default: background_image/snapped/)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--scale", type=int, default=None,
                    help="Just NEAREST-upscale the snapped native image by "
                         "this integer factor and stop -- no fit/crop/pad or "
                         "target resolution. Overrides --width/--height/--fit.")
    ap.add_argument("--fit", choices=["crop", "pad"], default="crop",
                    help="crop = scale-and-crop to fill; pad = fit whole panel, pad sides")
    ap.add_argument("--anchor", choices=["top", "center", "bottom"], default="center",
                    help="which vertical band to keep when --fit crop discards overflow")
    ap.add_argument("--native-cols", type=int, default=None,
                    help="Override detected native grid width")
    ap.add_argument("--native-rows", type=int, default=None,
                    help="Override detected native grid height")
    ap.add_argument("--sample-frac", type=float, default=0.5,
                    help="Fraction of each source cell (centred) to sample "
                         "for its colour; lower = crisper edges (0.35-0.6)")
    ap.add_argument("--colors", type=int, default=0,
                    help="Snap the native image to this many flat colours, no "
                         "dithering (e.g. 48); 0 = keep all colours")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    inputs = resolve_inputs(args.input)
    print(f"Processing {len(inputs)} image(s) -> {args.out_dir}/")
    for img_path in inputs:
        print(f"\n=== {os.path.basename(img_path)} ===")
        sb.process(img_path, args.out_dir, args.width, args.height, args.fit,
                   args.anchor, args.native_cols, args.native_rows,
                   args.sample_frac, args.colors, args.scale,
                   preview=not args.no_preview)
