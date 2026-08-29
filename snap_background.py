#!/usr/bin/env python3
"""
Snap an AI-generated "pixel-art" background to its true pixel grid, then
upscale it to a screen resolution with crisp, un-blurred blocks.

Why this is needed
------------------
Image models render "pixel art" at an arbitrary output resolution: every
art-pixel ends up spanning a non-integer number of image pixels (here
~2.93), the grid has a sub-pixel phase offset, and JPEG saved fuzzy,
noisy cell edges on top. A naive ``resize`` (or an integer ``// k``
downscale with the wrong phase) blends neighbouring art-pixels together
and the result looks soft, not pixelated.

Method (mirrors the coverage-aware tiling in pixelate.py, minus the
chroma-key step which does not apply to a full scene):

1. Trim the flat grey margin/gutter. If the file is a comparison sheet
   with several panels side by side, each panel is found and processed
   on its own.
2. Detect the native pixel count + sub-pixel phase: for each candidate
   grid, sample cell centres, blow the result back up NEAREST, and score
   the MSE against the source. The real grid round-trips almost losslessly
   -> a sharp minimum; a grid one step too fine splits real pixels across
   two cells and the error jumps (that split is what turns a 1px outline
   into a soft 2-3px band).
3. Reduce to that grid taking, per cell, the MEDIAN of only its central
   ~50% -- the anti-aliased/JPEG seam between neighbouring art-pixels is
   never averaged in.
4. Optionally snap to a flat N-colour palette (no dither) so edge ramps
   collapse to one solid colour.
5. NEAREST upscale by an integer factor large enough to cover the target,
   then crop-to-fill (or fit + pad) to the exact target size.

Usage
-----
    python3 snap_background.py background_image/intro_background.jpeg \
        --out-dir background_image/snapped --width 1920 --height 1080

    # fit the whole panel and pad the sides instead of cropping
    python3 snap_background.py in.png --fit pad

    # force the native grid instead of detecting it
    python3 snap_background.py in.png --native-cols 210 --native-rows 222
"""
import argparse
import os
import numpy as np
from PIL import Image

GREY_MARGIN = (75, 75, 75)


# --------------------------------------------------------------------------
# 1. panel / content detection
# --------------------------------------------------------------------------
def _runs(mask):
    """(start, end) inclusive ranges of contiguous True in a 1-D bool array."""
    out, ins, start = [], False, 0
    for i, v in enumerate(mask):
        if v and not ins:
            start, ins = i, True
        elif not v and ins:
            out.append((start, i - 1))
            ins = False
    if ins:
        out.append((start, len(mask) - 1))
    return out


def find_panels(arr, margin_rgb=GREY_MARGIN, tol=30, min_frac=0.08):
    """Split a sheet into panels by locating flat-grey gutters.

    Returns a list of (x0, x1, y0, y1) inclusive boxes, left-to-right.
    A plain single-image input yields one box (the grey border trimmed).
    """
    dist = np.abs(arr.astype(int) - np.array(margin_rgb)).sum(axis=2)
    content = dist > tol

    col_runs = [r for r in _runs(content.any(axis=0))
                if (r[1] - r[0] + 1) >= min_frac * arr.shape[1]]
    if not col_runs:
        col_runs = [(0, arr.shape[1] - 1)]

    boxes = []
    for x0, x1 in col_runs:
        rows_here = content[:, x0:x1 + 1].any(axis=1)
        row_runs = [r for r in _runs(rows_here)
                    if (r[1] - r[0] + 1) >= min_frac * arr.shape[0]]
        y0 = min(r[0] for r in row_runs) if row_runs else 0
        y1 = max(r[1] for r in row_runs) if row_runs else arr.shape[0] - 1
        boxes.append((x0, x1, y0, y1))
    return boxes


# --------------------------------------------------------------------------
# 2. native grid detection
# --------------------------------------------------------------------------
def _reupscale_mse(g, nx, ny, phx, phy):
    """Sample an nx*ny grid (cell centres) out of grayscale ``g``, blow it
    back up NEAREST to the source size, and return the MSE against ``g``.

    If (nx, ny, phase) match the art's real pixel grid, every source pixel
    falls inside exactly one cell and the round-trip is near lossless -> a
    sharp MSE minimum. A grid that is slightly too fine or mis-phased
    splits real pixels across two cells and the error jumps.
    """
    h, w = g.shape
    px, py = w / nx, h / ny
    xs = np.clip(phx + (np.arange(nx) + 0.5) * px, 0, w - 1).astype(int)
    ys = np.clip(phy + (np.arange(ny) + 0.5) * py, 0, h - 1).astype(int)
    small = g[np.ix_(ys, xs)]
    ux = np.clip(((np.arange(w) - phx) / px), 0, nx - 1).astype(int)
    uy = np.clip(((np.arange(h) - phy) / py), 0, ny - 1).astype(int)
    return float(np.mean((small[np.ix_(uy, ux)] - g) ** 2))


def detect_native_grid(panel_rgb, cell_lo=2.4, cell_hi=6.0, phase_step=0.1):
    """Find the native pixel-art grid of one panel.

    _reupscale_mse falls monotonically as the grid gets finer (at
    native == source it is zero), so its absolute minimum is useless. But
    the *true* grid still shows up as a local dip: at the right cell count
    every source pixel lands cleanly in one cell, at count +/-1 they smear
    and the error rises. So score each candidate by how far it sits below
    the local median of the error curve ("prominence") and take the
    strongest dip. Sub-pixel phase is then refined at that count.

    Returns (ncols, nrows, cell_w, cell_h, phase_x, phase_y, detail);
    ``detail`` is the panel's gradient energy -- a near-flat night frame
    scores low, so a multi-panel caller picks the detailed panel as the
    grid reference and reuses its cell size for the rest.
    """
    g = np.asarray(panel_rgb.convert("RGB"), dtype=np.float32).mean(axis=2)
    h, w = g.shape

    counts = np.arange(max(2, round(w / cell_hi)), round(w / cell_lo) + 1)
    err = np.array([_reupscale_mse(g, nx, max(1, round(h * nx / w)), 0.0, 0.0)
                    for nx in counts])
    k = 7
    base = np.array([np.median(err[max(0, j - k):j + k + 1])
                     for j in range(len(err))])
    prominence = (base - err) / np.maximum(base, 1e-6)
    nx = int(counts[int(np.argmax(prominence))])
    ny = max(1, round(h * nx / w))
    cell = w / nx

    bph = (_reupscale_mse(g, nx, ny, 0.0, 0.0), 0.0, 0.0)
    for phx in np.arange(0.0, cell, phase_step):
        for phy in np.arange(0.0, cell, phase_step):
            mse = _reupscale_mse(g, nx, ny, phx, phy)
            if mse < bph[0]:
                bph = (mse, float(phx), float(phy))

    detail = float(np.abs(np.diff(g, axis=1)).sum() + np.abs(np.diff(g, axis=0)).sum())
    return nx, ny, w / nx, h / ny, bph[1], bph[2], detail


# --------------------------------------------------------------------------
# 3. snap to native resolution
# --------------------------------------------------------------------------
def snap_to_native(panel_rgb, ncols, nrows, phase_x=0.0, phase_y=0.0,
                   sample_frac=0.5):
    """Reduce the panel to an ncols x nrows grid. Each output pixel is the
    median of only the central ``sample_frac`` of its source cell, so the
    anti-aliased / JPEG-fuzzed seam between two neighbouring art-pixels is
    never averaged in -- that seam bleed is what fattens a 1px outline into
    a soft 2-3px band. ``phase_*`` shifts the cell lattice to sit on the
    art's real pixel boundaries."""
    arr = np.asarray(panel_rgb.convert("RGB"))
    sh, sw, _ = arr.shape
    px, py = sw / ncols, sh / nrows
    m = (1.0 - sample_frac) / 2.0

    out = np.zeros((nrows, ncols, 3), dtype=np.uint8)
    for oy in range(nrows):
        cy = phase_y + oy * py
        y0 = max(0, min(int(np.floor(cy + m * py)), sh - 1))
        y1 = max(y0 + 1, min(int(np.ceil(cy + (1.0 - m) * py)), sh))
        for ox in range(ncols):
            cx = phase_x + ox * px
            x0 = max(0, min(int(np.floor(cx + m * px)), sw - 1))
            x1 = max(x0 + 1, min(int(np.ceil(cx + (1.0 - m) * px)), sw))
            block = arr[y0:y1, x0:x1].reshape(-1, 3)
            out[oy, ox] = np.median(block, axis=0).astype(np.uint8)
    return Image.fromarray(out)


def snap_palette(img_rgb, n_colors):
    """Collapse near-duplicate colours (the soft ramp around every edge)
    onto a fixed flat palette, no dithering -- so a dark outline becomes
    one solid colour instead of a 2-3 step gradient. n_colors <= 0 = off."""
    if not n_colors or n_colors <= 0:
        return img_rgb
    q = img_rgb.quantize(colors=n_colors, method=Image.MEDIANCUT,
                         dither=Image.Dither.NONE)
    return q.convert("RGB")


# --------------------------------------------------------------------------
# 4. upscale to target
# --------------------------------------------------------------------------
def upscale_to_target(native_rgb, target_w, target_h, fit="crop",
                      anchor="center", pad_rgb=None):
    """NEAREST-upscale by an integer factor, then crop-to-fill or fit+pad
    to exactly (target_w, target_h). ``anchor`` (top/center/bottom) picks
    which band survives when crop-to-fill discards vertical overflow."""
    nw, nh = native_rgb.size
    if fit == "crop":
        factor = max(1, int(np.ceil(max(target_w / nw, target_h / nh))))
    else:  # pad / fit: whole panel visible
        factor = max(1, int(np.floor(min(target_w / nw, target_h / nh))) or 1)

    big = native_rgb.resize((nw * factor, nh * factor), Image.NEAREST)
    bw, bh = big.size

    if fit == "crop":
        left = (bw - target_w) // 2
        top = {"top": 0, "bottom": bh - target_h}.get(anchor, (bh - target_h) // 2)
        top = max(0, min(top, bh - target_h))
        return big.crop((left, top, left + target_w, top + target_h))

    if pad_rgb is None:
        edge = np.asarray(big)
        pad_rgb = tuple(int(v) for v in np.median(
            np.concatenate([edge[0], edge[-1], edge[:, 0], edge[:, -1]]), axis=0))
    canvas = Image.new("RGB", (target_w, target_h), pad_rgb)
    canvas.paste(big, ((target_w - bw) // 2, (target_h - bh) // 2))
    return canvas


# --------------------------------------------------------------------------
def process(input_path, out_dir, target_w, target_h, fit, anchor="center",
            force_cols=None, force_rows=None, sample_frac=0.5, colors=0,
            scale=None, preview=True):
    os.makedirs(out_dir, exist_ok=True)
    src = Image.open(input_path).convert("RGB")
    arr = np.asarray(src)
    boxes = find_panels(arr)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    multi = len(boxes) > 1
    panels = [src.crop((x0, y0, x1 + 1, y1 + 1)) for (x0, x1, y0, y1) in boxes]
    print(f"{os.path.basename(input_path)}: {len(boxes)} panel(s) detected")

    # One cell size for the whole sheet: detect on every panel but trust the
    # one with the most edge detail (a near-black night frame carries almost
    # no signal of its own), then reuse its cell size + phase for the rest.
    ref = None
    if not (force_cols and force_rows):
        grids = [detect_native_grid(p) for p in panels]
        ref_i = max(range(len(grids)), key=lambda i: grids[i][6])
        ref = grids[ref_i]
        print(f"  reference grid from panel {ref_i}: cell "
              f"{ref[2]:.3f}x{ref[3]:.3f}px  phase {ref[4]:.2f},{ref[5]:.2f}")

    for i, panel in enumerate(panels):
        pw, ph = panel.size
        if force_cols and force_rows:
            ncols, nrows = force_cols, force_rows
            phx = phy = 0.0
        else:
            _, _, cw, ch, phx, phy, _ = ref
            ncols = max(1, round(pw / cw))
            nrows = max(1, round(ph / ch))
        print(f"  panel {i}: content {pw}x{ph} -> native {ncols}x{nrows} "
              f"(cell {pw / ncols:.3f}x{ph / nrows:.3f}, phase {phx:.2f},{phy:.2f})")

        native = snap_to_native(panel, ncols, nrows, phx, phy, sample_frac)
        native = snap_palette(native, colors)
        if scale:
            final = native.resize((ncols * scale, nrows * scale), Image.NEAREST)
            fw, fh = final.size
        else:
            final = upscale_to_target(native, target_w, target_h, fit, anchor)
            fw, fh = target_w, target_h

        tag = f"_{i}" if multi else ""
        native_path = os.path.join(out_dir, f"{stem}{tag}_native_{ncols}x{nrows}.png")
        final_path = os.path.join(out_dir, f"{stem}{tag}_{fw}x{fh}.png")
        native.save(native_path)
        final.save(final_path)
        print(f"    {native_path}")
        print(f"    {final_path}")
        if preview:
            prev = native.resize((ncols * 4, nrows * 4), Image.NEAREST)
            prev.save(os.path.join(out_dir, f"{stem}{tag}_native_preview4x.png"))

    print(f"Done -> {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Background image (single panel or a comparison sheet)")
    ap.add_argument("--out-dir", default=None,
                    help="Output folder (default: <input dir>/snapped)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--scale", type=int, default=None,
                    help="Just NEAREST-upscale the snapped native image by "
                         "this integer factor and stop -- no fit, crop, pad "
                         "or target resolution. Overrides --width/--height/--fit.")
    ap.add_argument("--fit", choices=["crop", "pad"], default="crop",
                    help="crop = scale-and-crop to fill; pad = fit whole panel, pad sides")
    ap.add_argument("--anchor", choices=["top", "center", "bottom"], default="center",
                    help="which vertical band to keep when --fit crop discards overflow")
    ap.add_argument("--native-cols", type=int, default=None,
                    help="Override detected native grid width")
    ap.add_argument("--native-rows", type=int, default=None,
                    help="Override detected native grid height")
    ap.add_argument("--sample-frac", type=float, default=0.5,
                    help="Fraction of each source cell (centred) to average "
                         "for its colour; lower = crisper edges, less seam "
                         "bleed (0.35-0.6 is a good range)")
    ap.add_argument("--colors", type=int, default=0,
                    help="Snap the native image to this many flat colours, "
                         "no dithering (e.g. 48) so outlines stop being a "
                         "soft gradient; 0 = keep all colours")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.input)) or ".", "snapped")
    process(args.input, out_dir, args.width, args.height, args.fit, args.anchor,
            args.native_cols, args.native_rows, args.sample_frac, args.colors,
            args.scale, preview=not args.no_preview)
