#!/usr/bin/env python3
"""
Tile-atlas delivery builder.

Crops hand-marked regions out of an AI-generated "style sheet" megasheet,
snaps each to a clean ``grid x grid`` pixel-art tile, and packs them
left-to-right into the single-row atlas PNG + ``manifest.json`` that the
game's asset pipeline ingests (see ``2026_08_28/docs/asset_pipeline.md`` and
``tile_delivery_spec.md``).

The mandatory atlas is 4 cells wide, column order fixed:

    0 ground_sub   1 ground_top   2 platform   3 block

Columns 4+ are free. ``validate_spec`` enforces that contract (it mirrors
``tools/ingest_assets.gd`` ``_check_atlas``) so a delivery that passes here
ingests without a FAIL.

Region selection is a hand-authored spec (see ``load_spec``) -- nothing here
guesses where a tile lives. Use ``ruler_overlay`` to read pixel coordinates
off a source, and ``main_tiles.py --dry-run`` to preview every crop before
packing.
"""
import datetime
import json
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw

import snap_background as sb

MANDATORY = ["ground_sub", "ground_top", "platform", "block"]
COLLISIONS = ("full", "top", "none")


# --------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------
def load_spec(path):
    """Read an atlas spec JSON and fill defaults. Shape:

    {
      "delivery": "alpine-forest-tiles-01",  # -> delivery folder name
      "biome": "foothill_forest",            # a game biome (Stages.BIOMES)
      "stage": null,                         # null | 1..6
      "grid": 32,                            # output cell size, px
      "background": "opaque",                # opaque | transparent | chroma:RRGGBB
      "source_key": null,                    # colour keyed out of the SOURCE
                                             #   (magenta/green/#rrggbb/r,g,b);
                                             #   null = none. Only meaningful when
                                             #   `background` is transparent/chroma.
      "sample_frac": 0.5,                    # central fraction of each source
                                             #   block averaged per output pixel
      "palette_colors": 0,                   # flat-quantise the atlas to N; 0 = off
      "native_snap": true,                   # snap each source to its detected
                                             #   art-pixel grid BEFORE cropping,
                                             #   so tiles are crisp (see below)
      "sources": {                           # optional per-source overrides for
        "style_sheet.png": {                 #   the native-grid detector
          "native_grid": [576, 384],         #   force the art-pixel resolution
          "phase": [2, 0]                    #   force the sub-pixel lattice phase
        }
      },
      "tiles": [
        {"col": 0, "name": "ground_sub", "collision": "full",
         "src": "style_sheet.png", "rect": [x, y, w, h]},
        ...  col 0..3 mandatory, in MANDATORY order; 4+ free  ...
      ]
    }

    ``rect`` is ALWAYS ``[x, y, w, h]`` in the ORIGINAL source PNG's pixels
    (author it off ``ruler_overlay``). With ``native_snap`` on (default) the
    source is first reduced to its detected art-pixel grid -- an image model
    renders "16x16" art at some non-integer pixel size with a sub-pixel phase,
    and cropping/downscaling that directly blends neighbouring art-pixels into
    a soft mush; snapping to the true grid first makes every tile crisp. The
    rect is mapped into native space internally.
    """
    with open(path) as f:
        spec = json.load(f)
    spec.setdefault("stage", None)
    spec.setdefault("grid", 32)
    spec.setdefault("background", "opaque")
    spec.setdefault("source_key", None)
    spec.setdefault("sample_frac", 0.5)
    spec.setdefault("palette_colors", 0)
    spec.setdefault("native_snap", True)
    spec.setdefault("sources", {})
    spec.setdefault("segment", {})
    spec["_dir"] = os.path.dirname(os.path.abspath(path))
    spec["_path"] = os.path.abspath(path)
    return spec


def validate_spec(spec):
    """Return a list of hard errors (``[]`` == ok)."""
    errs = []
    grid = int(spec.get("grid", 32))
    if grid <= 0:
        errs.append(f"grid must be positive, got {grid}")
    if not spec.get("delivery"):
        errs.append("spec needs a 'delivery' name")
    if not spec.get("biome") and spec.get("stage") is None:
        errs.append("spec needs a 'biome' (or a 'stage')")
    if spec.get("stage") is not None and not (1 <= int(spec["stage"]) <= 6):
        errs.append(f"stage must be 1..6 or null, got {spec['stage']}")

    tiles = spec.get("tiles", [])
    if not tiles:
        errs.append("spec has no 'tiles'")
    cols = [t.get("col") for t in tiles]
    if cols != list(range(len(tiles))):
        errs.append(f"tile 'col' values must be 0..N-1 with no gaps, got {cols}")
    for i, name in enumerate(MANDATORY):
        if i >= len(tiles):
            errs.append(f"missing mandatory column {i} ({name})")
        elif tiles[i].get("name") != name:
            errs.append(f"column {i} must be named '{name}', got "
                        f"'{tiles[i].get('name')}'")
    for t in tiles:
        c = t.get("collision", "full")
        if c not in COLLISIONS:
            errs.append(f"col {t.get('col')}: collision '{c}' not in {COLLISIONS}")
        r = t.get("rect")
        if not (isinstance(r, (list, tuple)) and len(r) == 4
                and all(isinstance(v, (int, float)) for v in r)
                and r[2] > 0 and r[3] > 0):
            errs.append(f"col {t.get('col')}: rect must be [x, y, w>0, h>0], "
                        f"got {r!r}")
        if not t.get("src"):
            errs.append(f"col {t.get('col')}: missing 'src'")

    bg = str(spec.get("background", "opaque"))
    if bg not in ("opaque", "transparent") and not bg.startswith("chroma:"):
        errs.append(f"background '{bg}' must be opaque | transparent | chroma:RRGGBB")
    return errs


def _chroma_rgb(bg):
    """'chroma:00FF00' -> (0, 255, 0); anything else -> None."""
    if not str(bg).startswith("chroma:"):
        return None
    h = str(bg).split(":", 1)[1].lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# --------------------------------------------------------------------------
# per-source native-grid snap
# --------------------------------------------------------------------------
def native_source(img, override=None, sample_frac=0.5):
    """Reduce a source image to its detected art-pixel grid, so the fuzzy
    non-integer "pixels" an image model produces become hard, exact ones.

    Returns ``(native_img, cell_w, cell_h)`` -- ``cell_*`` is how many
    ORIGINAL pixels one art-pixel spans, i.e. the factor to divide a
    source-space rect by to land in ``native_img`` space.

    ``override`` (from ``spec['sources'][name]``) may pin ``native_grid``
    ``[nx, ny]`` and/or ``phase`` ``[px, py]`` when detection is wrong.
    """
    W, H = img.size
    override = override or {}
    if override.get("native_grid"):
        nx, ny = (int(v) for v in override["native_grid"])
        px, py = (float(v) for v in override.get("phase", (0.0, 0.0)))
    else:
        nx, ny, _cw, _ch, dpx, dpy, _ = sb.detect_native_grid(img)
        px, py = override.get("phase", (dpx, dpy))
        px, py = float(px), float(py)
    native = sb.snap_to_native(img, nx, ny, px, py, sample_frac)
    return native, W / nx, H / ny


# --------------------------------------------------------------------------
# one tile
# --------------------------------------------------------------------------
def snap_tile(native_img, cell_w, cell_h, rect_src, grid, sample_frac=0.5,
              key_rgb=None, key_tol=60, fill_rgb=None):
    """Crop a tile (``rect_src`` = [x, y, w, h] in ORIGINAL source pixels)
    out of an already-native-snapped source and return a crisp ``grid x grid``
    RGB tile.

    ``rect_src`` is mapped into native space by dividing by ``cell_*``. If the
    native tile is already <= ``grid`` on both axes it is NEAREST-upscaled
    (hard edges kept); if larger it is reduced with the central-median
    downscale (``snap_background.snap_to_native``).

    If ``key_rgb`` and ``fill_rgb`` are both given, native pixels within
    ``key_tol`` (per-channel sum) of ``key_rgb`` are repainted to ``fill_rgb``
    first, so the key can't bleed a coloured fringe and lands on an exact,
    re-keyable value.
    """
    x = int(round(rect_src[0] / cell_w))
    y = int(round(rect_src[1] / cell_h))
    w = max(1, int(round(rect_src[2] / cell_w)))
    h = max(1, int(round(rect_src[3] / cell_h)))
    crop = np.asarray(native_img.convert("RGB").crop((x, y, x + w, y + h))).copy()
    if key_rgb is not None and fill_rgb is not None:
        d = np.abs(crop.astype(int) - np.array(key_rgb)).sum(axis=2)
        crop[d <= key_tol] = fill_rgb
    tile = Image.fromarray(crop)
    if w > grid or h > grid:
        tile = sb.snap_to_native(tile, grid, grid, 0.0, 0.0, sample_frac)
    else:
        tile = tile.resize((grid, grid), Image.NEAREST)
    return tile.convert("RGB"), (w, h)


# --------------------------------------------------------------------------
# whole atlas
# --------------------------------------------------------------------------
def build_atlas(spec):
    """Return a dict:
        atlas       PIL.Image  (RGB, or RGBA when background == transparent)
        manifest    dict       schema-v1 manifest for this one delivery
        tiles       list[(tile_spec, PIL.Image grid x grid)]
        report      list[str]
    """
    grid = int(spec["grid"])
    bg = str(spec["background"])
    src_dir = spec["_dir"]
    key_rgb = (sb.parse_key_color(spec["source_key"])
               if spec.get("source_key") else None)
    fill_rgb = _chroma_rgb(bg)  # None unless chroma:*

    pal_n = int(spec["palette_colors"])
    sample_frac = float(spec["sample_frac"])
    native_snap = bool(spec.get("native_snap", True))
    src_overrides = spec.get("sources", {})
    srcs = {}          # name -> (native_img_or_source, cell_w, cell_h)
    out_tiles = []
    report = [f"delivery '{spec['delivery']}'  biome={spec.get('biome')}  "
              f"stage={spec.get('stage')}  grid={grid}  background={bg}"
              + (f"  palette={pal_n}" if pal_n > 0 else "  palette=off")
              + (f"  native_snap=on" if native_snap else "  native_snap=off")]
    if key_rgb is not None and fill_rgb is None and bg != "transparent":
        report.append("  note: 'source_key' is set but background is opaque "
                      "-- key ignored (cells are fully painted)")

    for t in spec["tiles"]:
        name = t["src"]
        if name not in srcs:
            img = Image.open(os.path.join(src_dir, name))
            if native_snap:
                nat, cw, ch = native_source(img, src_overrides.get(name),
                                            sample_frac)
                srcs[name] = (nat, cw, ch)
                report.append(f"  source {name}: {img.size[0]}x{img.size[1]} "
                              f"-> native {nat.size[0]}x{nat.size[1]}  "
                              f"(art-pixel = {cw:.3f}x{ch:.3f} src px)")
            else:
                srcs[name] = (img.convert("RGB"), 1.0, 1.0)
        nat, cw, ch = srcs[name]
        tile, (nw, nh) = snap_tile(nat, cw, ch, t["rect"], grid, sample_frac,
                                   key_rgb, 60, fill_rgb)
        out_tiles.append((t, tile))
        x, y, w, h = t["rect"]
        report.append(f"  col {t['col']}  {t['name']:<11s} <- {name} "
                      f"[{x},{y} {w}x{h}px = {nw}x{nh} art-px] -> {grid}x{grid}  "
                      f"collision={t.get('collision', 'full')}")

    n = len(out_tiles)

    # Pack into one RGB scratch atlas so a single shared palette snap covers
    # every cell (no cell-to-cell colour drift), then key/convert as needed.
    scratch = Image.new("RGB", (n * grid, grid), fill_rgb or (0, 0, 0))
    for t, tile in out_tiles:
        scratch.paste(tile, (int(t["col"]) * grid, 0))
    if pal_n > 0:
        scratch = sb.snap_palette(scratch, pal_n)
    if fill_rgb is not None:
        # snap_palette / resampling can nudge the flat key off its exact value;
        # pull anything close back to it so ingest's edge-residue check is clean
        a = np.asarray(scratch).astype(int).copy()
        d = np.abs(a - np.array(fill_rgb)).sum(axis=2)
        a[d <= 48] = fill_rgb
        scratch = Image.fromarray(a.astype(np.uint8))
    report.append(f"  packed atlas colours: {_ncolours(scratch)}")

    if bg == "transparent":
        if key_rgb is not None:
            atlas = sb.key_out(scratch, key_rgb, 60)
        else:
            atlas = scratch.convert("RGBA")
    else:
        atlas = scratch

    manifest = {
        "schema": 1,
        "delivery": spec["delivery"],
        "defaults": {"grid": grid, "background": bg},
        "assets": [{
            "file": _atlas_filename(spec),
            "role": "tile_atlas",
            "biome": spec.get("biome", ""),
            "stage": spec.get("stage"),
            "grid": grid,
            "background": bg,
            "tiles": [
                {"col": int(t["col"]), "name": t["name"],
                 "collision": t.get("collision", "full")}
                for t, _ in out_tiles
            ],
        }],
    }
    report.append(f"  atlas: {n} cols x {grid}px  ->  {n * grid}x{grid}")
    return {"atlas": atlas, "manifest": manifest, "tiles": out_tiles,
            "report": report}


def _atlas_filename(spec):
    stem = spec.get("biome") or f"stage{spec.get('stage')}"
    return f"{stem}_atlas.png"


def _ncolours(img):
    a = np.asarray(img.convert("RGB")).reshape(-1, 3)
    return len(np.unique(a, axis=0))


# --------------------------------------------------------------------------
# writing a delivery
# --------------------------------------------------------------------------
def write_delivery(out_root, spec, built, dry_run=False):
    """Write ``<out_root>/<delivery>/`` with the atlas, manifest, an 8x
    preview, and per-tile previews. ``--dry-run`` writes only the previews +
    report (no atlas / manifest), so a spec can be dialled in safely.
    Returns (dest_dir, report_lines).
    """
    dest = os.path.join(out_root, spec["delivery"])
    os.makedirs(dest, exist_ok=True)
    lines = list(built["report"])
    atlas = built["atlas"]

    for t, tile in built["tiles"]:
        pv = tile.resize((tile.width * 8, tile.height * 8), Image.NEAREST)
        pv.save(os.path.join(dest, f"_tile_{t['col']}_{t['name']}_preview.png"))
    atlas.resize((atlas.width * 8, atlas.height * 8), Image.NEAREST).save(
        os.path.join(dest, "_atlas_preview.png"))

    fname = built["manifest"]["assets"][0]["file"]
    if dry_run:
        lines.append("\n(dry run -- previews only; no atlas / manifest written)")
    else:
        atlas.save(os.path.join(dest, fname))
        with open(os.path.join(dest, "manifest.json"), "w") as f:
            json.dump(built["manifest"], f, indent=2)
            f.write("\n")
        lines.append(f"\nwrote {fname}  ({atlas.width}x{atlas.height}, {atlas.mode})")
        lines.append("wrote manifest.json")
    lines.append(f"wrote _atlas_preview.png + {len(built['tiles'])} _tile_*_preview.png")

    with open(os.path.join(dest, "_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return dest, lines


# --------------------------------------------------------------------------
# asset catalogue for the game-engine workspace
# --------------------------------------------------------------------------
_ROLE_HINT_BY_NAME = [
    ("waterfall", "water"), ("water", "water"),
    ("parallax", "parallax"), ("background", "parallax"),
    ("cloud", "parallax"), ("mountain", "parallax"), ("sky", "parallax"),
    ("formation", "decoration"), ("decor", "decoration"), ("prop", "prop"),
    ("hazard", "hazard"), ("enemy", "enemy"),
    ("terrain", "tile_atlas"), ("tile", "tile_atlas"), ("style", "tile_atlas"),
]


def _role_hint(sheet_png, spec):
    seg = spec.get("segment", {})
    if sheet_png in seg.get("role_hints", {}):
        return seg["role_hints"][sheet_png]
    per = seg.get("per_source", {}).get(sheet_png, {})
    if per.get("role"):
        return per["role"]
    low = sheet_png.lower()
    for key, role in _ROLE_HINT_BY_NAME:
        if key in low:
            return role
    return "tile_atlas"


def write_asset_catalog(spec, delivery_dir):
    """Companion to ``manifest.json`` for a workspace on the game-engine side.

    Lists every EXTRA tile segmented from this delivery's source sheets
    (``tile_segment.write_catalog`` output under ``tiles/<biome>/catalog/``),
    copies the PNGs + montages into ``<delivery>/catalog/``, and tags each
    sheet with a ``role_hint``. ``tools/ingest_assets.gd`` does not read this
    file -- it's a menu for pulling additional assets in by hand.

    Returns ``(catalog.json path, tile count)`` or ``None`` when there is no
    segmentation catalogue to fold in (run ``segment`` / ``all`` first).
    """
    cat_src = os.path.join(spec["_dir"], "catalog")
    if not os.path.isdir(cat_src):
        return None

    sheets = []
    total = 0
    for stem in sorted(os.listdir(cat_src)):
        idx_path = os.path.join(cat_src, stem, "index.json")
        if not os.path.isfile(idx_path):
            continue
        with open(idx_path) as f:
            idx = json.load(f)
        src_png = f"{stem}.png"
        dst_dir = os.path.join(delivery_dir, "catalog", stem)
        os.makedirs(dst_dir, exist_ok=True)

        tiles = []
        for t in idx.get("tiles", []):
            fn = f"{int(t['id']):03d}.png"
            src_file = os.path.join(cat_src, stem, fn)
            if not os.path.isfile(src_file):
                continue
            shutil.copy2(src_file, os.path.join(dst_dir, fn))
            tiles.append({
                "id": t["id"],
                "file": f"catalog/{stem}/{fn}",
                "section": t.get("section"),
                "art_size": t.get("art_size"),
                "src_rect": t.get("src_rect"),
                "colors": t.get("colors"),
                "shaped": bool(t.get("shaped", False)),
            })
        montage_rel = None
        m = os.path.join(cat_src, stem, "_montage.png")
        if os.path.isfile(m):
            shutil.copy2(m, os.path.join(dst_dir, "_montage.png"))
            montage_rel = f"catalog/{stem}/_montage.png"

        sheets.append({
            "source": src_png,
            "role_hint": _role_hint(src_png, spec),
            "art_pixel_src_px": idx.get("art_pixel_src_px"),
            "count": len(tiles),
            "montage": montage_rel,
            "tiles": tiles,
        })
        total += len(tiles)

    if not sheets:
        return None

    doc = {
        "schema": 1,
        "kind": "asset-catalog",
        "delivery": spec["delivery"],
        "biome": spec.get("biome", ""),
        "stage": spec.get("stage"),
        "grid": int(spec["grid"]),
        "generated": datetime.datetime.now(datetime.timezone.utc)
                     .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": total,
        "about": (
            "Optional EXTRA tiles segmented from the same source sheets as this "
            "delivery's manifest.json tile_atlas. tools/ingest_assets.gd does NOT "
            "read this file. To use a tile: copy its `file` (path relative to "
            "this folder) into assets/biomes/<biome>/<role-dir>/ or "
            "assets/stages/<id>/..., then add an entry to a manifest.json with "
            "`role` = the sheet's `role_hint` (docs/asset_pipeline.md lists the "
            "per-role fields). `art_size` is the tile's native pixel size -- on "
            "import scale it to a whole multiple of `grid`. `shaped` true = the "
            "PNG has transparency (a prop/decoration silhouette); false = a full "
            "opaque cell fit for a tile_atlas column. `src_rect` is the region "
            "in the original source sheet. These PNGs are grid-snapped but NOT "
            "palette-flattened -- quantise on import if a flat look is wanted."
        ),
        "sheets": sheets,
    }
    path = os.path.join(delivery_dir, "catalog.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    return path, total


# --------------------------------------------------------------------------
# authoring aid
# --------------------------------------------------------------------------
def ruler_overlay(src_img, step=64, out_path=None):
    """A copy of ``src_img`` with a magenta grid every ``step`` px and yellow
    axis labels, so rects for the spec can be read straight off it."""
    im = src_img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    for x in range(0, w, step):
        d.line([(x, 0), (x, h)], fill=(255, 0, 255))
        d.text((x + 2, 1), str(x), fill=(255, 255, 0))
        d.text((x + 2, h - 11), str(x), fill=(255, 255, 0))
    for y in range(0, h, step):
        d.line([(0, y), (w, y)], fill=(255, 0, 255))
        d.text((1, y + 1), str(y), fill=(255, 255, 0))
        d.text((max(0, w - 34), y + 1), str(y), fill=(255, 255, 0))
    if out_path:
        im.save(out_path)
    return im
