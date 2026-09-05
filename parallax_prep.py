#!/usr/bin/env python3
"""
Turn the magenta-keyed snapped background style-sheets into parallax layers.

Input  : background_image/snapped/<n>_<name>_native_<C>x<R>.png
         -- pixel-snapped sheets, each a scatter of discrete hill / mountain /
         cloud pieces (or a stack of full-width ridge rows) on a flat magenta
         chroma field (~#E205E6, it drifts a little per sheet).

Output : background_image/parallax/
         <layer>.png            one seamless, horizontally-tiling RGBA strip per
                                depth layer, magenta removed + de-fringed, art
                                bottom-aligned to the ground-contact line
                                (clouds: scattered in a sky band).
         parallax_catalog.json  depth-ordered metadata: 0 = closest to camera
                                (scrolls fastest), higher = farther. Per layer:
                                size, opaque bbox, ground line, piece count and
                                the recommended scroll_scale / autoscroll /
                                sprite-Y / z-index for the engine.
         manifest.json          schema-v1 ingest manifest (role "parallax") so
                                the folder drops straight into the game repo's
                                incoming/ and is picked up by tools/ingest_assets.gd.
         README.md              how a 2D engine consumes the stack, plus the
                                exact Godot-workspace wiring steps.
         _preview.png           the layers composited on a sky gradient at their
                                recommended offsets -- eyeball check only.

Every strip is periodic by construction: scatter layers place each piece once
along a period equal to the summed piece widths (+/- the overlap), wrapping the
overflow back to x=0; ridge layers are rolled so their lowest-profile column
sits on the seam. So `repeat_width` == the image width and the layer tiles with
no visible join.

Usage:
    python parallax_prep.py                       # all layers -> background_image/parallax/
    python parallax_prep.py --out-dir some/dir
    python parallax_prep.py --only trees,clouds
"""
import argparse
import datetime as _dt
import json
import os

import numpy as np
from PIL import Image

import tile_segment as ts

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN_DIR = os.path.join(_PROJECT_DIR, "background_image", "snapped")
DEFAULT_OUT_DIR = os.path.join(_PROJECT_DIR, "background_image", "parallax")

BIOME = "foothill_forest"
DELIVERY = "foothill-forest-parallax-01"

# Depth stack, closest -> farthest. `scroll` / `drift` / `y` / `z` mirror the
# game's tuned DEFAULT_LAYERS for this biome (data/stages.gd), so a strip is a
# drop-in for the procedural art/bg/<biome>_<key>.tres it replaces.
#   mode "scatter" : cut the sheet into pieces, re-pack them along one periodic
#                    baseline (bottom-aligned; clouds -> `band` of canvas height).
#   mode "ridge"   : the sheet rows are already full-width ridgelines; take one,
#                    bottom-align it, roll it to the emptiest column for the seam.
LAYERS = [
    dict(idx=0, key="trees", node="Trees",
         src="0_hill_native_480x320.png", mode="scatter",
         upscale=3, overlap=0.34, base_jitter=6, seed=70,
         scroll=[0.76, 0.47], drift=[0, 0], y=344, z=-8),
    dict(idx=1, key="mist", node="Mist",
         src="1_hill_native_576x384.png", mode="scatter",
         upscale=2, overlap=0.30, base_jitter=5, seed=71,
         scroll=[0.60, 0.34], drift=[-3, 0], y=300, z=-12),
    dict(idx=2, key="mtn_near", node="MountainsNear",
         src="2_mountain_close_native_576x384.png", mode="ridge",
         upscale=3, row="median", seed=72,
         scroll=[0.48, 0.27], drift=[0, 0], y=138, z=-20),
    dict(idx=3, key="mtn_far", node="MountainsFar",
         src="3_mountain_far_native_576x384.png", mode="ridge",
         upscale=3, row="lowest", seed=73,
         scroll=[0.34, 0.19], drift=[0, 0], y=88, z=-26),
    dict(idx=4, key="clouds", node="Clouds",
         src="4_clouds_native_521x278.png", mode="scatter",
         upscale=2, overlap=-0.22, base_jitter=0, band=(0.04, 0.72), seed=74,
         scroll=[0.14, 0.07], drift=[-6, 0], y=76, z=-32),
]

TOP_PAD = 10          # transparent rows kept above the tallest piece
GRID = 32             # game pixel grid; strip dims rounded UP to a multiple


# --------------------------------------------------------------------------
# keying
# --------------------------------------------------------------------------
def _border_key(rgb):
    """Median colour of the 1px frame -- the sheet's actual magenta."""
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(int)
    return tuple(int(v) for v in np.median(edge, axis=0))


def key_rgba(img, *, key_tol=72, key_margin=42, peel=1):
    """RGB image -> RGBA array with the magenta keyed out.

    ts.key_bg_mask catches the whole anti-aliased ramp into the key; we then
    erode the alpha by `peel` px to drop the last blended rim, and kill any
    pixel that still reads as the magenta hue family (these sheets have no
    pink/violet art of their own, so that is safe). RGB under transparent
    pixels is zeroed so a bilinear sampler can't drag magenta back.
    """
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    key = _border_key(rgb)
    bg = ts.key_bg_mask(rgb, key, key_tol, key_margin)
    a = (~bg).astype(np.uint8)
    if peel > 0:
        a = cv2.erode(a, np.ones((3, 3), np.uint8), iterations=peel)

    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    fam = (r > 150) & (b > 140) & (g < r - 40) & (g < b - 30)
    a[fam] = 0

    out = np.dstack([rgb, a * 255]).astype(np.uint8)
    out[a == 0, :3] = 0
    return out, key


# --------------------------------------------------------------------------
# pieces
# --------------------------------------------------------------------------
def _components(alpha, min_area=90, min_w=12, min_h=8):
    """(x, y, w, h) of every solid blob, big first, tiny specks dropped."""
    import cv2

    m = (alpha > 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area or w < min_w or h < min_h:
            continue
        out.append((x, y, w, h))
    return out


def _trim(rgba):
    ys, xs = np.where(rgba[..., 3] > 0)
    if len(xs) == 0:
        return rgba, (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return rgba[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


def _upscale(rgba, k):
    if k == 1:
        return rgba
    return np.asarray(Image.fromarray(rgba).resize(
        (rgba.shape[1] * k, rgba.shape[0] * k), Image.NEAREST))


def _over(dst, src, x, y):
    """Alpha-over `src` (RGBA) onto `dst` at (x, y), x wrapping modulo width."""
    H, W = dst.shape[:2]
    sh, sw = src.shape[:2]
    sa = src[..., 3:4].astype(np.float32) / 255.0
    for row in range(sh):
        ty = y + row
        if ty < 0 or ty >= H:
            continue
        cols = (x + np.arange(sw)) % W
        a = sa[row]
        dpx = dst[ty, cols].astype(np.float32)
        spx = src[row].astype(np.float32)
        outa = a + dpx[..., 3:4] / 255.0 * (1 - a)
        rgb = np.where(outa > 0, (spx[..., :3] * a + dpx[..., :3] * (dpx[..., 3:4] / 255.0) * (1 - a)) / np.maximum(outa, 1e-6), 0)
        dst[ty, cols, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        dst[ty, cols, 3] = np.clip(outa[..., 0] * 255, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
def build_scatter(pieces, cfg):
    """`pieces` = list of trimmed RGBA. Lay each once along a period equal to
    the summed widths scaled by (1 - overlap); wrap overflow to x=0 so the
    strip is exactly periodic. Bottom-aligned, unless `band` -> place the top
    edge at a random fraction of canvas height (clouds)."""
    rng = np.random.default_rng(cfg["seed"])
    order = list(rng.permutation(len(pieces)))
    ws = [pieces[i].shape[1] for i in order]
    step = 1.0 - cfg["overlap"]
    # Period rounded to the pixel grid so the strip width is grid-clean; each
    # piece then gets a slice of it proportional to its own width, so the
    # layout stays periodic (overflow wraps) with the intended overlap/gap.
    period = _ceil_grid(sum(w * step for w in ws))
    total_w = float(sum(ws))
    starts = np.concatenate([[0.0], np.cumsum(ws)[:-1]]) / total_w * period

    max_h = max(p.shape[0] for p in pieces)
    band = cfg.get("band")
    if band:
        canvas_h = _ceil_grid(max_h + int(max_h * band[0] / max(band[1] - band[0], 1e-6)) + TOP_PAD)
    else:
        canvas_h = _ceil_grid(max_h + cfg["base_jitter"] + TOP_PAD)
    canvas = np.zeros((canvas_h, period, 4), np.uint8)

    placed = []
    for oi, s in zip(order, starts):
        p = pieces[oi]
        ph, pw = p.shape[:2]
        x = int(round(s)) % period
        if band:
            top = float(rng.uniform(band[0], band[1]))
            y = int(round(top * (canvas_h - ph)))
        else:
            y = canvas_h - ph - int(rng.integers(0, cfg["base_jitter"] + 1))
        _over(canvas, p, x, y)
        placed.append([int(x), int(y), int(pw), int(ph)])
    return canvas, placed, period


def build_ridge(rgba, alpha, cfg):
    """The sheet's rows are already full-width ridgelines. Pick one, upscale,
    bottom-align on a grid canvas, feather the vertical edges and roll it so
    its emptiest column lands on the seam."""
    import cv2

    rows = _row_bands(alpha)
    pick = {"median": len(rows) // 2, "lowest": -1, "top": 0, "tallest": None}
    idx = cfg.get("row", "median")
    if idx == "tallest":
        r = max(rows, key=lambda rb: rb[1] - rb[0])
    else:
        r = rows[pick.get(idx, len(rows) // 2)]
    y0, y1 = r
    strip = rgba[y0:y1]
    strip, _bb = _trim(strip)
    strip = _upscale(strip, cfg["upscale"])

    sh, sw = strip.shape[:2]
    canvas_h = _ceil_grid(sh + TOP_PAD)
    canvas_w = _ceil_grid(sw)                       # grid-clean width
    canvas = np.zeros((canvas_h, canvas_w, 4), np.uint8)
    canvas[canvas_h - sh:, :sw] = strip            # extra columns stay transparent

    # feather 2px in from each edge of the real content
    for k, f in ((0, 0.25), (1, 0.6), (sw - 2, 0.6), (sw - 1, 0.25)):
        canvas[:, k, 3] = (canvas[:, k, 3] * f).astype(np.uint8)

    col_fill = (canvas[..., 3] > 0).sum(axis=0)
    seam = int(np.argmin(col_fill))                # lands in the transparent pad
    canvas = np.roll(canvas, -seam, axis=1)
    return canvas, seam


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _ceil_grid(v, g=GRID):
    return int(np.ceil(v / g) * g)


def _row_bands(alpha, gap=6):
    """Split an alpha sheet into horizontal bands of content (its ridge rows)."""
    rowsum = (alpha > 0).sum(axis=1)
    on = rowsum > alpha.shape[1] * 0.02
    bands, start = [], None
    run = 0
    for y, v in enumerate(on):
        if v:
            if start is None:
                start = y
            run = 0
        else:
            if start is not None:
                run += 1
                if run >= gap:
                    bands.append((start, y - run + 1))
                    start = None
    if start is not None:
        bands.append((start, len(on)))
    return bands or [(0, alpha.shape[0])]


def _depth_factor(scroll_x):
    """0.0 (locked to camera / infinitely far) .. 1.0 (locked to world / at
    the play plane), straight from the horizontal scroll scale."""
    return round(float(scroll_x), 4)


# --------------------------------------------------------------------------
# preview
# --------------------------------------------------------------------------
def _sky(w, h):
    top = np.array([46, 74, 120], np.float32)
    mid = np.array([126, 160, 196], np.float32)
    hor = np.array([236, 216, 188], np.float32)
    out = np.zeros((h, w, 3), np.float32)
    for y in range(h):
        t = y / (h - 1)
        c = top + (mid - top) * (t / 0.55) if t < 0.55 else mid + (hor - mid) * (((t - 0.55) / 0.45) ** 1.4)
        out[y, :] = c
    return out


def write_preview(built, path, w=1280, h=648):
    base = _sky(w, h)
    canvas = np.dstack([base, np.full((h, w), 255, np.float32)])
    for lay in sorted(built, key=lambda d: d["cfg"]["z"]):
        strip = lay["img"]
        cfg = lay["cfg"]
        tiled = np.concatenate([strip] * (w // strip.shape[1] + 2), axis=1)[:, :w]
        y = int(cfg["y"])
        sh = min(tiled.shape[0], h - max(y, 0))
        if sh <= 0:
            continue
        seg = tiled[:sh].astype(np.float32)
        a = seg[..., 3:4] / 255.0
        canvas[y:y + sh, :, :3] = seg[..., :3] * a + canvas[y:y + sh, :, :3] * (1 - a)
    Image.fromarray(np.clip(canvas[..., :3], 0, 255).astype(np.uint8)).save(path)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def process(in_dir, out_dir, only=None):
    os.makedirs(out_dir, exist_ok=True)
    built = []
    for cfg in LAYERS:
        if only and cfg["key"] not in only:
            continue
        src = os.path.join(in_dir, cfg["src"])
        if not os.path.isfile(src):
            print(f"  ! {cfg['key']}: source missing {cfg['src']} -- skipped")
            continue
        img = Image.open(src)
        rgba, key = key_rgba(img)
        alpha = rgba[..., 3]

        if cfg["mode"] == "ridge":
            strip, seam = build_ridge(rgba, alpha, cfg)
            placed, period, seam_info = None, strip.shape[1], {"rolled_to_col": seam}
            piece_count = len(_row_bands(alpha))
        else:
            comps = _components(alpha)
            pieces = []
            for (x, y, w, h) in comps:
                pc, _bb = _trim(rgba[y:y + h, x:x + w])
                pieces.append(_upscale(pc, cfg["upscale"]))
            strip, placed, period = build_scatter(pieces, cfg)
            seam_info = {"periodic": True, "pieces_wrapped": True}
            piece_count = len(pieces)

        _tr, bbox = _trim(strip)
        H, W = strip.shape[:2]
        out_png = os.path.join(out_dir, f"{cfg['key']}.png")
        Image.fromarray(strip).save(out_png)
        print(f"  {cfg['key']:9s} {W}x{H}  pieces={piece_count}  "
              f"key={key}  -> {os.path.basename(out_png)}")

        built.append(dict(cfg=cfg, img=strip, entry={
            "depth_index": cfg["idx"],
            "name": cfg["key"],
            "engine_node": cfg["node"],
            "file": f"{cfg['key']}.png",
            "source_sheet": cfg["src"],
            "mode": cfg["mode"],
            "image_size": [W, H],
            "repeat_width": W,
            "opaque_bbox": [int(v) for v in bbox],
            "ground_y": H,
            "horizon_y": int(bbox[1]),
            "upscale": cfg["upscale"],
            "piece_count": piece_count,
            "seam": seam_info,
            "recommended": {
                "scroll_scale": cfg["scroll"],
                "autoscroll": cfg["drift"],
                "sprite_y": cfg["y"],
                "z_index": cfg["z"],
                "anchor": "top_left",
            },
            "depth_factor": _depth_factor(cfg["scroll"][0]),
        }))

    if not built:
        raise SystemExit("nothing built")

    today = _dt.date.today().isoformat()
    catalog = {
        "schema": "parallax-catalog/1",
        "generated": today,
        "biome": BIOME,
        "source": "background_image/snapped/",
        "layer_count": len(built),
        "note": ("depth_index 0 = closest to the camera (parallax moves fastest); "
                 "higher index = farther away. This is a 5-layer stack; a scene "
                 "may use fewer. A procedural sky sits behind index "
                 f"{built[-1]['entry']['depth_index']} and is not delivered here."),
        "conventions": {
            "scroll_scale": "[x,y] screen px moved per 1 px of camera travel. "
                            "1 = pinned to the world, 0 = pinned to the camera. "
                            "Smaller = farther.",
            "autoscroll": "[x,y] constant px/second drift, independent of the camera.",
            "anchor": "sprite top-left is placed at [0, sprite_y]; the texture "
                      "repeats horizontally every repeat_width px.",
            "ground_y": "y of the ground-contact line inside the texture "
                        "(= image height; art is bottom-aligned). clouds float.",
        },
        "layers": [b["entry"] for b in built],
    }
    with open(os.path.join(out_dir, "parallax_catalog.json"), "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
        f.write("\n")

    manifest = {
        "schema": 1,
        "delivery": DELIVERY,
        "generated": today,
        "defaults": {"grid": GRID, "background": "transparent"},
        "assets": [
            {
                "file": b["entry"]["file"],
                "role": "parallax",
                "biome": BIOME,
                "layer": b["cfg"]["key"],
                "scroll": b["cfg"]["scroll"],
                "drift": b["cfg"]["drift"],
            }
            for b in sorted(built, key=lambda d: d["cfg"]["z"])
        ],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    write_preview(built, os.path.join(out_dir, "_preview.png"))
    write_readme(os.path.join(out_dir, "README.md"), catalog, built)
    print(f"\n-> {out_dir}/  ({len(built)} layers + catalog + manifest + README + preview)")


def write_readme(path, catalog, built):
    L = []
    w = L.append
    w(f"# {DELIVERY} — parallax background layers")
    w("")
    w(f"Biome **`{BIOME}`**. {len(built)} horizontally-tiling RGBA strips cut from "
      "the magenta-keyed snapped style-sheets in `background_image/snapped/`, "
      "background removed and de-fringed, art bottom-aligned to the "
      "ground-contact line (clouds scattered in a sky band).")
    w("")
    w("Regenerate with `python parallax_prep.py`.")
    w("")
    w("## Layer stack")
    w("")
    w("`depth 0` = closest to the camera (scrolls fastest); higher = farther. "
      "A procedural sky sits behind the last layer and is **not** in this delivery.")
    w("")
    w("| depth | file | node | size (px) | repeat_width | scroll_scale | autoscroll | sprite_y | z | source sheet |")
    w("|------:|------|------|-----------|-------------:|--------------|------------|---------:|--:|--------------|")
    for b in sorted(built, key=lambda d: d["cfg"]["idx"]):
        e = b["entry"]
        r = e["recommended"]
        w(f"| {e['depth_index']} | `{e['file']}` | `{e['engine_node']}` | "
          f"{e['image_size'][0]}×{e['image_size'][1]} | {e['repeat_width']} | "
          f"{r['scroll_scale']} | {r['autoscroll']} | {r['sprite_y']} | {r['z_index']} | "
          f"`{e['source_sheet']}` |")
    w("")
    w("Full per-layer metadata (opaque bbox, horizon line, piece count, seam "
      "handling) is in [`parallax_catalog.json`](parallax_catalog.json).")
    w("")
    w("## How a 2D engine draws it")
    w("")
    w("Each layer is one sprite that **tiles horizontally** every `repeat_width` "
      "px and never tiles vertically. Its top-left is anchored at "
      "`(0, sprite_y)` in screen space, then offset against the camera:")
    w("")
    w("```")
    w("for layer in layers:                       # far -> near")
    w("    off_x = -camera.x * layer.scroll_scale.x + time * layer.autoscroll.x")
    w("    off_y = -camera.y * layer.scroll_scale.y + layer.sprite_y")
    w("    draw_tiled_x(layer.texture, off_x mod layer.repeat_width, off_y)")
    w("```")
    w("")
    w("- `scroll_scale` `[x,y]`: screen px moved per px of camera travel. "
      "`1` = pinned to the world, `0` = pinned to the camera. Smaller ⇒ farther.")
    w("- `autoscroll` `[x,y]`: constant px/second drift (wind on the clouds and "
      "mist), independent of the camera.")
    w("- Strips are **seamless**: scatter layers are periodic (each piece placed "
      "once per `repeat_width`, overflow wrapped to x=0); ridge layers are "
      "rolled so their emptiest column is on the join and the vertical edges "
      "are feathered.")
    w("- Art is **bottom-aligned**: `ground_y` (= image height) is the "
      "ground-contact line. Sit that on your horizon. Clouds are the exception "
      "— they float, so `ground_y` is nominal.")
    w("")
    w("## Godot workspace (`../2026_08_28`)")
    w("")
    w("This matches the `role: \"parallax\"` ingest contract "
      "(`docs/asset_pipeline.md`). Layer keys map to the `Parallax2D` nodes "
      "`gen_stages.gd` builds, and `scroll` / `drift` here equal the tuned "
      "`DEFAULT_LAYERS` values in `data/stages.gd`, so each strip is a drop-in "
      "for the procedurally-painted `art/bg/{biome}_{key}.tres` it replaces.")
    w("")
    w("```sh")
    w(f"cp -r background_image/parallax  ../2026_08_28/incoming/{DELIVERY}")
    w("cd ../2026_08_28")
    w(f"godot --headless -s res://tools/ingest_assets.gd --path . -- --delivery {DELIVERY}")
    w("godot --headless -s res://tools/gen_background.gd --path .   # wrap into art/bg/*.tres")
    w("godot --headless -s res://tools/gen_stages.gd     --path .   # rebuild stage scenes")
    w("```")
    w("")
    w("Notes:")
    w("")
    w("- `ingest_assets.gd` re-keys on `background: \"transparent\"` (a no-op "
      "here) and checks the pixel grid. Every strip is sized to a whole "
      f"multiple of {GRID} px on both axes, so the (soft) parallax grid check "
      "passes clean.")
    w("- These strips are **shorter** than the procedural layers (real "
      "pixel-art ridgelines, not full-height haze). `gen_stages.gd` reads "
      "`repeat_size` from the texture width automatically, but the per-layer "
      "`sprite_y` in `DEFAULT_LAYERS` was tuned for the tall painted art. If a "
      "horizon sits too high after ingest, nudge that layer's `y` — the "
      "`sprite_y` column above is the starting point.")
    w("- A biome with every parallax slot **and** a tile atlas imported "
      "(`Assets.is_complete`) drops `BgPainter` entirely. `sky` is still "
      "painted; deliver a `sky` layer too if you want it replaced.")
    w("")
    w("## Files")
    w("")
    for b in sorted(built, key=lambda d: d["cfg"]["idx"]):
        w(f"- `{b['entry']['file']}` — depth {b['entry']['depth_index']} "
          f"({b['entry']['engine_node']}), {b['entry']['image_size'][0]}×"
          f"{b['entry']['image_size'][1]}")
    w("- `parallax_catalog.json` — engine-agnostic depth-ordered metadata")
    w("- `manifest.json` — schema-v1 ingest manifest for the Godot pipeline")
    w("- `_preview.png` — layers composited on a sky gradient (check only)")
    w("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=DEFAULT_IN_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--only", default=None,
                    help="comma list of layer keys to (re)build, e.g. trees,clouds")
    args = ap.parse_args()
    only = set(s.strip() for s in args.only.split(",")) if args.only else None
    process(args.in_dir, args.out_dir, only)
