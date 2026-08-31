#!/usr/bin/env python3
"""
Pixel-art downscaling with NO background bleed and NO dithering.

The naive approaches both fail on a flat-chroma-background source:
  - NEAREST downscale: point-samples one source pixel per block -> picks up
    stray anti-aliasing noise from the source at random.
  - Plain BOX (area-average) downscale: averages ALL pixels in a block
    including background ones, so any block straddling the character's edge
    blends in green -> halo/fringe pixels that are neither a real character
    color nor the background color.

Correct method (coverage-aware foreground reduction):
  For each output pixel, look at its source block. Classify each source
  pixel in that block as foreground or background using the flat-color
  background test. If the foreground pixel *count* clears a coverage
  threshold, the output pixel is reduced from ONLY the foreground pixels
  in that block (background samples excluded entirely -- no blending),
  otherwise it is the exact background color -- a crisp, binary-clean
  silhouette boundary with zero contaminated colors.

  How those foreground pixels reduce to one color is `method` (see
  _block_color): the default `median` rejects the minority side of an
  interior edge so outlines/creases stay sharp; `mean` (the old default)
  averages them and looks soft; `center` medians only the block's middle.

After downscaling, palette quantization uses dither=NONE (hard nearest-color
snap, no ordered/error-diffusion dithering pattern) so every output pixel is
exactly one of a small, fixed set of flat colors.
"""
import numpy as np
from PIL import Image

BG_EXACT = (0, 255, 0)


# Chroma-key. The source's "green" is rarely a clean (0,255,0): AI renders
# and JPEG leave it anywhere from bright lime to a murky dark green along
# edges, so the key is deliberately wide -- any pixel where green clearly
# leads the other two channels.
def is_background(arr, g_thresh=115, margin=42):
    r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
    return (g > g_thresh) & (g > r + margin) & (g > b + margin)


def looks_greenish(arr, margin=14, dark_level=48, dark_margin=4):
    """Looser still than is_background, for the CONTAMINATED-COLOR test only
    (never for the silhouette). Two rules:

      * green leads r and b by `margin`, OR
      * the pixel is very dark (all channels < dark_level) and green merely
        leads by `dark_margin` -- a near-black outline pixel that picked up
        a faint green cast from the backdrop, e.g. (8, 17, 4).

    Such pixels are too dark for is_background so they stay in the
    silhouette, but their color must not feed the block average or the
    palette or a green rim / green-tinted dark entry survives (very visible
    once --n-colors is high)."""
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    lead = (g > r + margin) & (g > b + margin)
    dark = (np.maximum(np.maximum(r, g), b) < dark_level)
    dark_green = dark & (g >= r + dark_margin) & (g >= b + dark_margin) & (g > r) & (g > b)
    return lead | dark_green


def _block_color(block, clean, block_fg_mask, method, sample_frac):
    """Pick one output color for a source block from its foreground pixels.

    method:
      mean    per-channel mean of the clean foreground pixels. Softest --
              a block straddling an outline and skin becomes muddy tan.
      median  per-channel median instead -- rejects the minority side of
              an edge, so outlines stay dark instead of bleeding.
      center  per-channel median over only the central `sample_frac` of
              the block -- the part least likely to straddle an edge.
              Crispest; a small `sample_frac` can add speckle in noisy
              regions (hair), so tune it if so.
    """
    pick = clean if clean.any() else block_fg_mask

    if method == "mean":
        return block[pick].mean(axis=0)

    if method == "center":
        bh, bw = block.shape[:2]
        m = (1.0 - sample_frac) / 2.0
        cy0, cy1 = int(np.floor(bh * m)), int(np.ceil(bh * (1.0 - m)))
        cx0, cx1 = int(np.floor(bw * m)), int(np.ceil(bw * (1.0 - m)))
        csel = np.zeros_like(pick)
        csel[max(cy0, 0):max(cy1, cy0 + 1), max(cx0, 0):max(cx1, cx0 + 1)] = True
        csel &= pick
        pool = block[csel] if csel.any() else block[pick]
        return np.median(pool, axis=0)

    # median
    return np.median(block[pick], axis=0)


def coverage_aware_downscale(img_rgb, target_size, coverage_thresh=0.35,
                             method="median", sample_frac=0.6):
    """Downscale to (target_size, target_size) [or (tw, th) tuple] without
    letting background bleed into edge pixels. `method` / `sample_frac`
    control how each block's color is chosen -- see _block_color."""
    if isinstance(target_size, int):
        tw = th = target_size
    else:
        tw, th = target_size

    arr = np.array(img_rgb.convert("RGB"))
    sh, sw, _ = arr.shape
    fg_mask = ~is_background(arr)
    # pixels that shape the silhouette but whose color is green-contaminated
    color_mask = fg_mask & ~looks_greenish(arr)

    # block boundaries via linspace so source dims that don't divide evenly
    # (e.g. 910 / 64) still tile the full image with no gaps/overlaps
    x_bounds = np.linspace(0, sw, tw + 1).astype(int)
    y_bounds = np.linspace(0, sh, th + 1).astype(int)

    out = np.full((th, tw, 3), BG_EXACT, dtype=np.uint8)

    for oy in range(th):
        y0, y1 = y_bounds[oy], y_bounds[oy + 1]
        if y1 <= y0:
            y1 = y0 + 1
        for ox in range(tw):
            x0, x1 = x_bounds[ox], x_bounds[ox + 1]
            if x1 <= x0:
                x1 = x0 + 1

            block_fg_mask = fg_mask[y0:y1, x0:x1]
            coverage = block_fg_mask.mean()

            if coverage >= coverage_thresh:
                block = arr[y0:y1, x0:x1]
                clean = color_mask[y0:y1, x0:x1]
                out[oy, ox] = _block_color(block, clean, block_fg_mask,
                                           method, sample_frac).astype(np.uint8)
            # else: leave as BG_EXACT (already set)

    return Image.fromarray(out, "RGB")


def quantize_locked(img_rgb, n_colors=12, fixed_palette_rgb=None):
    """Snap the image to a small flat palette with zero dithering.

    If fixed_palette_rgb is given (list of (r,g,b) tuples), pixels are
    hard-assigned to the nearest color in that fixed set (used to keep an
    animation's frames color-consistent). Otherwise a new palette of
    n_colors is derived from this image's own foreground pixels.
    """
    arr = np.array(img_rgb.convert("RGB"))
    mask_bg = np.all(arr == BG_EXACT, axis=-1)
    fg = arr[~mask_bg]

    if fixed_palette_rgb is None:
        fg_img = Image.fromarray(fg.reshape(1, -1, 3).astype("uint8"))
        quant = fg_img.quantize(colors=n_colors, method=Image.MEDIANCUT,
                                 dither=Image.Dither.NONE)
        palette = np.array(quant.getpalette()[:n_colors * 3]).reshape(-1, 3)
        idx = np.array(quant).flatten()
        fg_mapped = palette[idx]
    else:
        palette = np.array(fixed_palette_rgb)
        # hard nearest-color assignment, no dithering
        dists = ((fg[:, None, :].astype(int) - palette[None, :, :].astype(int)) ** 2).sum(axis=2)
        idx = dists.argmin(axis=1)
        fg_mapped = palette[idx]

    result = arr.copy()
    result[~mask_bg] = fg_mapped.astype(np.uint8)
    result[mask_bg] = BG_EXACT
    return Image.fromarray(result.astype("uint8"), "RGB"), palette


def extract_palette(img_rgb, n_colors=12):
    """Get just the dominant palette from an image's foreground, without
    producing an output image -- used to build a SHARED palette across
    multiple animation frames before quantizing each one."""
    arr = np.array(img_rgb.convert("RGB"))
    mask_bg = is_background(arr)
    fg = arr[~mask_bg]
    if len(fg) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    fg_img = Image.fromarray(fg.reshape(1, -1, 3).astype("uint8"))
    quant = fg_img.quantize(colors=n_colors, method=Image.MEDIANCUT,
                             dither=Image.Dither.NONE)
    return np.array(quant.getpalette()[:n_colors * 3]).reshape(-1, 3)


def pixelate(img_rgb, target_size=64, n_colors=12, fixed_palette_rgb=None,
             coverage_thresh=0.35, method="median", sample_frac=0.6):
    """Full pipeline: coverage-aware downscale -> dither-free quantize."""
    down = coverage_aware_downscale(img_rgb, target_size, coverage_thresh,
                                    method, sample_frac)
    out, palette = quantize_locked(down, n_colors, fixed_palette_rgb)
    return out, palette


def _consolidate_palette(palette, weights, merge_dist):
    """Greedily drop palette entries that sit within `merge_dist` (RGB
    Euclidean) of an already-kept, more-populous entry.

    Why: a big MEDIANCUT palette (say --n-colors 256) on a smoothly shaded
    source ends up with clusters of near-identical entries. Per-frame
    nearest-color assignment then flip-flops boundary pixels between two of
    them as the downscale sampling shifts a hair between frames -> a subtle
    color shimmer that is invisible at low --n-colors (entries far apart)
    but shows up high. Keeping one entry per perceptible color removes the
    thing that can flip. merge_dist <= 0 disables it."""
    if merge_dist is None or merge_dist <= 0 or len(palette) <= 1:
        return palette
    order = np.argsort(weights)[::-1]
    kept = []
    for i in order:
        c = palette[i].astype(int)
        if all(((c - palette[k].astype(int)) ** 2).sum() > merge_dist ** 2
               for k in kept):
            kept.append(i)
    return palette[sorted(kept)]


def build_shared_palette(fg_pixel_arrays, n_colors=12, merge_dist=10):
    """Build one shared palette from foreground pixels pooled across several
    images (e.g. all frames of an animation), so quantizing each frame
    against this same palette keeps colors consistent frame-to-frame
    instead of drifting/flickering."""
    pooled = np.concatenate([p for p in fg_pixel_arrays if len(p) > 0], axis=0)
    # drop any lingering green-contaminated edge pixels before the palette
    # is derived, so no palette slot is spent on a green rim (matters most
    # at high --n-colors, where such a slot would otherwise survive)
    clean = pooled[~looks_greenish(pooled.reshape(-1, 1, 3)).ravel()]
    if len(clean) >= max(8, n_colors):
        pooled = clean
    pooled_img = Image.fromarray(pooled.reshape(1, -1, 3).astype("uint8"))
    quant = pooled_img.quantize(colors=n_colors, method=Image.MEDIANCUT,
                                 dither=Image.Dither.NONE)
    # trim to the palette entries actually used -- when the pool has fewer
    # than n_colors distinct colors, getpalette() pads the tail with zeros
    # and those would show up as spurious black swatches / palette slots.
    idx = np.array(quant).ravel()
    n_used = min(int(idx.max()) + 1, n_colors)
    palette = np.array(quant.getpalette()[:n_used * 3]).reshape(-1, 3)
    counts = np.bincount(idx, minlength=n_used)[:n_used]
    return _consolidate_palette(palette, counts, merge_dist)


def foreground_pixels_of_downscaled(img_rgb):
    """Foreground-only pixels of an already-downscaled (coverage_aware_downscale)
    image, for pooling into build_shared_palette."""
    arr = np.array(img_rgb.convert("RGB"))
    mask_bg = np.all(arr == BG_EXACT, axis=-1)
    return arr[~mask_bg]



def verify_clean(img_rgb):
    """Run integrity checks. Returns dict of results."""
    arr = np.array(img_rgb.convert("RGB"))
    colors = np.unique(arr.reshape(-1, 3), axis=0)

    # any color that's "greenish but not exact BG" indicates leftover bleed
    r, g, b = colors[:, 0].astype(int), colors[:, 1].astype(int), colors[:, 2].astype(int)
    greenish_not_exact = ((g > r + 30) & (g > b + 30) &
                           ~((r == 0) & (g == 255) & (b == 0)))
    n_bleed = greenish_not_exact.sum()

    return {
        "total_colors": len(colors),
        "suspected_bleed_colors": int(n_bleed),
        "bleed_examples": [tuple(int(v) for v in c) for c in colors[greenish_not_exact][:5]],
    }


if __name__ == "__main__":
    import sys
    src = Image.open(sys.argv[1] if len(sys.argv) > 1 else "01_cropped.png")
    out, palette = pixelate(src, target_size=64, n_colors=12)
    out.save("06_pixelated_clean.png")
    out.resize((512, 512), Image.NEAREST).save("07_preview_clean_512.png")
    report = verify_clean(out)
    print("Verification:", report)
