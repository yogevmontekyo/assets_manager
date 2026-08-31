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

Correct method (coverage-aware foreground averaging):
  For each output pixel, look at its source block. Classify each source
  pixel in that block as foreground or background using the flat-color
  background test. If the foreground pixel *count* clears a coverage
  threshold, the output pixel = the average of ONLY the foreground pixels
  in that block (background samples excluded entirely -- no blending).
  Otherwise the output pixel = the exact background color. This yields a
  crisp, binary-clean silhouette boundary with zero contaminated colors.

After downscaling, palette quantization uses dither=NONE (hard nearest-color
snap, no ordered/error-diffusion dithering pattern) so every output pixel is
exactly one of a small, fixed set of flat colors.
"""
import numpy as np
from PIL import Image

BG_EXACT = (0, 255, 0)


def is_background(arr, g_thresh=140, margin=60):
    r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
    return (g > g_thresh) & (g > r + margin) & (g > b + margin)


def looks_greenish(arr, margin=24):
    """Much looser than is_background: any pixel whose green channel clearly
    leads red and blue. These are the source's own anti-aliased edge pixels
    -- too dark to pass is_background (g <= 140) so they count as part of
    the silhouette, but their color is contaminated chroma-green and must
    be kept out of the averaged output. Invisible at low --n-colors (they
    get lumped into a brown/black bucket); at high --n-colors they'd get
    their own slot and show as a green rim."""
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    return (g > r + margin) & (g > b + margin)


def coverage_aware_downscale(img_rgb, target_size, coverage_thresh=0.35):
    """Downscale to (target_size, target_size) [or (tw, th) tuple] without
    letting background bleed into edge pixels."""
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
                # average only the clean (non-green-contaminated) silhouette
                # pixels; fall back to all silhouette pixels if a block is
                # entirely fringe
                clean = color_mask[y0:y1, x0:x1]
                pick = clean if clean.any() else block_fg_mask
                out[oy, ox] = block[pick].mean(axis=0).astype(np.uint8)
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
             coverage_thresh=0.35):
    """Full pipeline: coverage-aware downscale -> dither-free quantize."""
    down = coverage_aware_downscale(img_rgb, target_size, coverage_thresh)
    out, palette = quantize_locked(down, n_colors, fixed_palette_rgb)
    return out, palette


def build_shared_palette(fg_pixel_arrays, n_colors=12):
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
    n_used = int(np.array(quant).max()) + 1
    n_used = min(n_used, n_colors)
    return np.array(quant.getpalette()[:n_used * 3]).reshape(-1, 3)


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
