#!/usr/bin/env python3
"""
Catalogue every tile on an AI-generated style-sheet.

Analogous to ``extract_frames.py`` for sprites: instead of hand-marking rects
for the four atlas columns, this finds *every* discrete piece of art on a
megasheet (terrain tiles, slopes, platforms, columns, trees, bushes, rocks,
flowers, vines...), snaps each to the sheet's native art-pixel grid, and
writes them out as individually indexed PNGs plus a contact sheet, a montage
and an ``index.json``. Later you eyeball the montage, pick the ids you want,
and point ``atlas_spec.json`` at their rects.

Method
------
1. Foreground mask: the sheet's alpha channel if it has a real one
   (style_sheet.png), else a chroma key (magenta / green / a border colour).
2. Connected components (8-connectivity) on the mask.
3. Drop non-art: too small / too large, label pills and the NOTES box
   (wide + short, or mostly navy+white), the magenta chroma-key swatch,
   flat low-variety blobs.
4. Sort into reading order (row bands top->bottom, left->right within a row).
5. Snap the whole sheet to its native grid (snap_background) once, map each
   component's source rect into native space, crop -> a crisp tile.

Nothing here writes into a delivery; it only produces a catalogue under
``tiles/<biome>/catalog/<sheet>/``.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

import snap_background as sb


# --------------------------------------------------------------------------
# foreground mask
# --------------------------------------------------------------------------
def _has_real_alpha(img):
    if img.mode not in ("RGBA", "LA", "PA"):
        return False
    a = np.asarray(img.convert("RGBA"))[..., 3]
    return a.min() < 250 and (a < 16).mean() > 0.02


def key_bg_mask(rgb, key_rgb, key_tol=60, key_margin=40):
    """Bool mask, True where a pixel is background for chroma key ``key_rgb``.

    Two rules OR'd, so the whole anti-aliased ramp from the key into the art
    is caught, not just pixels near the exact key value:

      * sum |px - key| <= ``key_tol``  (near the key), OR
      * the pixel still carries the key's channel signature: its keyed-HIGH
        channels stay bright, its keyed-LOW channels stay dark, and high leads
        low by ``key_margin``. For a magenta key (R,B high, G low):
        ``R,B > hi_floor  and  G < lo_cap  and  R > G+m  and  B > G+m`` with
        ``hi_floor = 200 - key_margin`` and ``lo_cap = 96 + key_margin``.

    The bright/dark caps stop the rule from eating hue-adjacent art: lavender
    mountains against a magenta key are ``R~B > G`` too, but their G (~150)
    clears ``lo_cap`` and their shadows drop below ``hi_floor``, so they
    survive; a half-keyed pink fringe pixel like (251, 130, 252) still keys.
    Raise ``key_margin`` to also eat whiter fringe (at more risk to purple art).
    """
    rgb = rgb.astype(int)
    k = np.array(key_rgb, dtype=int)
    near = np.abs(rgb - k).sum(axis=2) <= key_tol

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    kr, kg, kb = int(k[0]), int(k[1]), int(k[2])
    hi = [c for c, v in zip("rgb", (kr, kg, kb)) if v >= 200]
    lo = [c for c, v in zip("rgb", (kr, kg, kb)) if v <= 120]
    ch = {"r": r, "g": g, "b": b}
    hi_floor = 200 - key_margin
    lo_cap = 96 + key_margin
    sig = np.ones(r.shape, dtype=bool) if (hi and lo) else np.zeros(r.shape, dtype=bool)
    for h in hi:
        sig &= ch[h] > hi_floor
    for l in lo:
        sig &= ch[l] < lo_cap
    for h in hi:
        for l in lo:
            sig &= ch[h] > ch[l] + key_margin
    return near | sig


def foreground_mask(img, alpha_thresh=64, key=None, key_tol=60, key_margin=40):
    """Return ``(mask uint8{0,1}, tag str, key_rgb|None)`` -- ``mask`` is 1
    where the pixel is art. Uses alpha when the image has a genuine alpha
    channel, otherwise keys out ``key`` (a colour name / #rrggbb / r,g,b) or,
    if ``key`` is None, the median border colour, via ``key_bg_mask``."""
    if _has_real_alpha(img):
        a = np.asarray(img.convert("RGBA"))[..., 3]
        return (a > alpha_thresh).astype(np.uint8), "alpha>%d" % alpha_thresh, None
    rgb = np.asarray(img.convert("RGB"))
    if key is not None:
        k = sb.parse_key_color(key)
        tag = "key %s" % (k,)
    else:
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(int)
        k = tuple(int(v) for v in np.median(border, axis=0))
        tag = "border-colour %s" % (k,)
    bg = key_bg_mask(rgb, k, key_tol, key_margin)
    return (~bg).astype(np.uint8), tag, tuple(int(v) for v in k)


# --------------------------------------------------------------------------
# non-art rejection
# --------------------------------------------------------------------------
def _is_label_or_text(crop_rgb, comp_mask):
    """A section label pill / the NOTES box / stray text: the component's
    pixels are overwhelmingly dark-navy + white (+ leftover key magenta).

    Navy here means blue actually leads red (b >= r) -- so it doesn't catch
    dark brown cave rock, which is red-dominant."""
    px = crop_rgb[comp_mask]
    if len(px) == 0:
        return True
    r, g, b = px[:, 0].astype(int), px[:, 1].astype(int), px[:, 2].astype(int)
    navy = (r < 80) & (g < 85) & (b < 130) & (b >= r)
    white = (r > 195) & (g > 195) & (b > 195)
    mag = (r > 170) & (b > 150) & (g < r - 35) & (g < b - 20)
    return float((navy | white | mag).mean()) > 0.72


def _is_magenta(crop_rgb, comp_mask):
    px = crop_rgb[comp_mask]
    if len(px) == 0:
        return False
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    return float(((r > 170) & (b > 150) & (g < r - 35) & (g < b - 20)).mean()) > 0.35


def _low_variety(crop_rgb, comp_mask, min_colors=5):
    px = crop_rgb[comp_mask]
    if len(px) < 16:
        return True
    q = (px // 24)
    return len(np.unique(q, axis=0)) < min_colors


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------
def segment(img, *, alpha_thresh=64, key=None, key_tol=60, key_margin=40,
            min_area=150, min_dim=6, max_w_frac=0.30, max_h_frac=0.55,
            pad=1, close_px=0, open_px=0,
            tile_px=None, grid_split=False, split_min_cov=0.15):
    """Return (list[component dict], mask_tag). Each dict:
        src_rect  [x, y, w, h]  in ORIGINAL source pixels (padded, clipped)
        area      foreground pixel count
        fill      area / bbox area
    in reading order.

    ``tile_px`` (w, h) in source px -- one tile's footprint. When given and
    ``grid_split`` is on, a connected component more than ~1.6 tiles wide or
    tall (a strip of tiles butted together with no key gap between them, e.g.
    the cave sheets) is sliced into a ``round(w/tile_w) x round(h/tile_h)``
    grid; cells with < ``split_min_cov`` foreground are dropped. Components
    that are already ~one tile pass through untouched (the alpine sheets).
    """
    import cv2

    W, H = img.size
    mask, tag, _key = foreground_mask(img, alpha_thresh, key, key_tol, key_margin)
    if close_px > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((close_px, close_px), np.uint8))
    if open_px > 0:
        # erode+dilate: snaps the thin bridges where tiles touch at a corner
        # / share a 1px seam, so connected components separates butted tiles.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((open_px, open_px), np.uint8))

    rgb = np.asarray(img.convert("RGB"))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    tpw = tile_px[0] if tile_px else None
    tph = tile_px[1] if tile_px else None
    raw = []          # (x, y, w, h) boxes, big strips already split
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area or w < min_dim or h < min_dim:
            continue
        if w >= 2.8 * h and h <= 44:                       # label pill / title / rule
            continue
        cm = (lab[y:y + h, x:x + w] == i)

        # A strip of tiles butted together with no key gap -> slice on the
        # tile grid. Content filters (text / low-variety) are skipped for it:
        # a whole butted row of flat dark rock legitimately looks "low variety".
        big = grid_split and tpw and (w > 1.6 * tpw or h > 1.6 * tph)
        if big:
            nx = max(1, int(round(w / tpw)))
            ny = max(1, int(round(h / tph)))
            if nx * ny >= 2:
                xs = np.linspace(x, x + w, nx + 1).round().astype(int)
                ys = np.linspace(y, y + h, ny + 1).round().astype(int)
                for gy in range(ny):
                    for gx in range(nx):
                        cx0, cx1 = xs[gx], xs[gx + 1]
                        cy0, cy1 = ys[gy], ys[gy + 1]
                        cell = cm[cy0 - y:cy1 - y, cx0 - x:cx1 - x]
                        if cell.size and cell.mean() >= split_min_cov:
                            raw.append((cx0, cy0, cx1 - cx0, cy1 - cy0))
                continue

        cr = rgb[y:y + h, x:x + w]
        if _is_label_or_text(cr, cm) or _is_magenta(cr, cm) or _low_variety(cr, cm):
            continue
        raw.append((x, y, w, h))

    comps = []
    for x, y, w, h in raw:
        if w > max_w_frac * W or h > max_h_frac * H:
            continue
        sub = mask[y:y + h, x:x + w]
        area = int(sub.sum())
        if area < min_area:
            continue
        px0, py0 = max(0, x - pad), max(0, y - pad)
        px1, py1 = min(W, x + w + pad), min(H, y + h + pad)
        comps.append({"src_rect": [px0, py0, px1 - px0, py1 - py0],
                      "area": area, "fill": round(area / (w * h), 3)})

    comps = _reading_order(comps)
    for idx, c in enumerate(comps):
        c["id"] = idx
    return comps, tag


def _reading_order(comps):
    """Row-band top->bottom, then left->right within a band."""
    if not comps:
        return comps
    bys = sorted(comps, key=lambda c: c["src_rect"][1])
    rows, cur, cy = [], [], None
    for c in bys:
        y, h = c["src_rect"][1], c["src_rect"][3]
        if cy is None or y < cy + max(14, h * 0.5):
            cur.append(c)
            cy = y if cy is None else cy
        else:
            rows.append(cur)
            cur, cy = [c], y
    if cur:
        rows.append(cur)
    return [c for r in rows for c in sorted(r, key=lambda c: c["src_rect"][0])]


# --------------------------------------------------------------------------
# section tagging (optional)
# --------------------------------------------------------------------------
def load_sections(path):
    """A sibling ``sections.json``: [{"name": "terrain", "rect": [x,y,w,h]}, ...]
    in source pixels. Each tile is tagged by which rect holds its centre."""
    if not path or not os.path.isfile(path):
        return []
    with open(path) as f:
        return json.load(f)


def _section_of(src_rect, sections):
    cx = src_rect[0] + src_rect[2] / 2
    cy = src_rect[1] + src_rect[3] / 2
    for s in sections:
        x, y, w, h = s["rect"]
        if x <= cx <= x + w and y <= cy <= y + h:
            return s["name"]
    return None


# --------------------------------------------------------------------------
# rendering + catalogue
# --------------------------------------------------------------------------
def _native(img, override=None, sample_frac=0.5):
    W, H = img.size
    override = override or {}
    if override.get("native_grid"):
        nx, ny = (int(v) for v in override["native_grid"])
        px, py = (float(v) for v in override.get("phase", (0.0, 0.0)))
    else:
        nx, ny, _cw, _ch, px, py, _ = sb.detect_native_grid(img.convert("RGB"))
    return sb.snap_to_native(img.convert("RGB"), nx, ny, float(px), float(py),
                             sample_frac), W / nx, H / ny


def write_catalog(img, out_dir, *, sheet_name="sheet", sections=None,
                  native_override=None, sample_frac=0.5, upscale=6, **seg_kw):
    """Segment ``img`` and write ``out_dir``:
        NNN.png            native-resolution crisp tile (RGBA, cut to its mask)
        _contact_sheet.png numbered boxes on the source
        _montage.png       every extracted tile in an indexed grid
        index.json         [{id, section, src_rect, art_rect, art_size, colors}]
    Returns the index list.

    Chroma-key clean-up knobs in ``seg_kw`` (see ``key_bg_mask``):
      ``key_tol`` / ``key_margin``  base keying width.
      ``magenta_peel``  int rounds -- grow the background into key-hue pixels
        that touch it, stripping the last blended rim. Erodes a purple
        parallax mountain's skyline by ~that many px; set 0 for such a sheet.
      ``magenta_kill``  bool -- also drop key-hue pixels anywhere in the tile.
        Only for art with no pink/violet of its own (water, clouds); it would
        erase the purple mountains.
    """
    import cv2

    peel = int(seg_kw.pop("magenta_peel", 2))
    mag_kill = bool(seg_kw.pop("magenta_kill", False))
    tile_art = int(seg_kw.pop("tile_art", 16))
    os.makedirs(out_dir, exist_ok=True)

    native, cw, ch = _native(img, native_override, sample_frac)
    seg_kw.setdefault("tile_px", (tile_art * cw, tile_art * ch))
    comps, tag = segment(img, **seg_kw)
    sections = sections or []

    # One native-resolution RGBA version with the background cut out, so both
    # the saved tiles and the montage carry no leftover key colour / matte.
    at = int(seg_kw.get("alpha_thresh", 64))
    if _has_real_alpha(img):
        am = np.asarray(sb.snap_to_native(
            Image.fromarray(np.asarray(img.convert("RGBA"))[..., 3]),
            native.width, native.height, 0.0, 0.0, sample_frac))
        cut = (am[..., 0] if am.ndim == 3 else am) > at
    else:
        ktol = int(seg_kw.get("key_tol", 60))
        kmarg = int(seg_kw.get("key_margin", 40))
        _m, _t, key_rgb = foreground_mask(img, at, seg_kw.get("key"), ktol, kmarg)
        na = np.asarray(native.convert("RGB"))
        # The alpha CUT can key harder than the segmentation mask: a wider
        # near-band mops up the fringe blend between the key and light art
        # (pink rim on white water) while pure hues like the purple mountains
        # stay well outside it. Segmentation used the tighter ``ktol`` so
        # component splitting isn't affected.
        cut = ~key_bg_mask(na, key_rgb, ktol + 100, kmarg)

        # `fam` = pixels that still read as the key's hue family (the key's
        # high channels lead its low channels by >24) -- i.e. any magenta /
        # pink / violet contamination left after keying.
        if max(key_rgb) >= 200 and min(key_rgb) <= 120:
            hich = [i for i, v in enumerate(key_rgb) if v >= 200]
            loch = [i for i, v in enumerate(key_rgb) if v <= 120]
            fam = np.ones(na.shape[:2], bool)
            for hi in hich:
                for lo in loch:
                    fam &= na[..., hi].astype(int) > na[..., lo].astype(int) + 24

            # Guarded peel: grow the background inward, but only into `fam`
            # pixels that touch the background -- strips the last blended rim
            # the colour tests can't safely reach without eating interior art.
            # Bounded by `magenta_peel` rounds (a purple parallax mountain
            # loses ~that many px off its skyline; set 0 for that sheet).
            if peel > 0:
                bg = ~cut
                cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
                for _ in range(peel):
                    adj = cv2.dilate(bg.astype(np.uint8), cross) > 0
                    bg |= adj & fam
                cut = ~bg

            # `magenta_kill`: also drop `fam` pixels anywhere (not just the
            # rim) -- for sheets whose art has no pink/violet of its own
            # (waterfall = blue/white/grey, clouds). DON'T set it for the
            # purple mountains: it would erase them.
            if mag_kill:
                cut &= ~fam
    native_cut = np.dstack([np.asarray(native.convert("RGB")),
                            cut.astype(np.uint8) * 255])
    native_cut = Image.fromarray(native_cut, "RGBA")

    index = []
    for c in comps:
        x, y, w, h = c["src_rect"]
        ax, ay = round(x / cw), round(y / ch)
        aw, ah = max(1, round(w / cw)), max(1, round(h / ch))
        ax, ay = min(ax, native.width - 1), min(ay, native.height - 1)
        aw, ah = min(aw, native.width - ax), min(ah, native.height - ay)
        tile = native_cut.crop((ax, ay, ax + aw, ay + ah))
        tile.save(os.path.join(out_dir, f"{c['id']:03d}.png"))
        arr = np.asarray(tile)
        alpha = arr[..., 3] > 0
        vis = arr[alpha][:, :3]
        # "full rectangular tile" = opaque pixels reach all four edges (a
        # terrain cell); a prop/decoration silhouette (stalactite, cloud,
        # crystal) only touches one edge or floats. Measured 1px in so the
        # transparent pad frame doesn't count.
        k = 1 if min(alpha.shape) > 4 else 0
        edges = (alpha[k].mean(), alpha[-1 - k].mean(),
                 alpha[:, k].mean(), alpha[:, -1 - k].mean())
        index.append({
            "id": c["id"],
            "section": _section_of(c["src_rect"], sections),
            "src_rect": [int(v) for v in c["src_rect"]],
            "art_rect": [int(ax), int(ay), int(aw), int(ah)],
            "art_size": [int(aw), int(ah)],
            "colors": int(len(np.unique(vis, axis=0))) if len(vis) else 0,
            "fill": c["fill"],
            "opaque_frac": round(float(alpha.mean()), 3),
            "shaped": bool(min(edges) < 0.4),
        })

    _contact_sheet(img, comps, os.path.join(out_dir, "_contact_sheet.png"))
    _montage(native_cut, cw, ch, comps, os.path.join(out_dir, "_montage.png"),
             upscale)
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"sheet": sheet_name, "mask": tag, "count": len(index),
                   "art_pixel_src_px": [round(cw, 4), round(ch, 4)],
                   "tiles": index}, f, indent=2)
        f.write("\n")
    return index, tag


def _contact_sheet(img, comps, path):
    disp = img.convert("RGB").copy()
    d = ImageDraw.Draw(disp)
    for c in comps:
        x, y, w, h = c["src_rect"]
        d.rectangle([x, y, x + w, y + h], outline=(255, 0, 0))
        d.text((x + 1, y + 1), str(c["id"]), fill=(255, 255, 0))
    disp.save(path)


def _montage(native, cw, ch, comps, path, upscale, cols=12):
    """Every extracted tile in a fixed grid, id in the corner. Each thumb is
    NEAREST-scaled up to fill its cell (so a 14px tile is actually visible),
    keeping aspect."""
    cell = 84
    lab = 12
    rows = (len(comps) + cols - 1) // cols or 1
    m = Image.new("RGB", (cols * cell, rows * cell), (18, 18, 20))
    d = ImageDraw.Draw(m)
    inner = cell - lab - 4
    for i, c in enumerate(comps):
        x, y, w, h = c["src_rect"]
        ax, ay = round(x / cw), round(y / ch)
        aw, ah = max(1, round(w / cw)), max(1, round(h / ch))
        t = native.crop((ax, ay, ax + aw, ay + ah))
        f = max(1, int(min(inner / max(1, t.width), inner / max(1, t.height))))
        t = t.resize((t.width * f, t.height * f), Image.NEAREST)
        if t.width > inner or t.height > inner:
            t.thumbnail((inner, inner), Image.NEAREST)
        gx, gy = (i % cols) * cell, (i // cols) * cell
        d.rectangle([gx, gy, gx + cell - 1, gy + cell - 1], outline=(40, 40, 46))
        pos = (gx + (cell - t.width) // 2, gy + lab + (inner - t.height) // 2)
        m.paste(t, pos, t if t.mode == "RGBA" else None)
        d.text((gx + 2, gy + 2), str(c["id"]), fill=(120, 255, 120))
    m.save(path)
