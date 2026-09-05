#!/usr/bin/env python3
"""
Tile-asset pipeline: catalogue a style-sheet and/or build a tile-atlas delivery.

One AI-generated "style sheet" megasheet per biome under ``tiles/<biome>/`` plus
a hand-authored ``atlas_spec.json``. Two things you do with them:

  segment   Find EVERY discrete tile on each source sheet, snap each to the
            sheet's native art-pixel grid, and write an indexed catalogue under
            ``tiles/<biome>/catalog/<sheet>/`` (NNN.png, _montage.png,
            _contact_sheet.png, index.json). Pick ids from the montage later.

  build     Crop the rects named in ``atlas_spec.json`` into the single-row
            atlas PNG + schema-v1 ``manifest.json`` the game ingests
            (``2026_08_28/docs/tile_delivery_spec.md``). Columns are fixed:
            ``0 ground_sub, 1 ground_top, 2 platform, 3 block`` (4+ free).

Every source is snapped to its detected art-pixel grid (same native-grid + phase
detection as ``snap_background.py``) before any crop, so an image model's fuzzy
non-integer "pixels" become hard ones. Pin it per source with
``spec['sources'][name]``; disable with ``"native_snap": false``.

Usage
-----
    python main_tiles.py                    # 'all': segment + build, every spec
    python main_tiles.py all [spec]
    python main_tiles.py segment [spec]
    python main_tiles.py build [spec] [--dry-run] [--incoming ../2026_08_28/incoming]
    python main_tiles.py ruler [spec]       # coordinate-ruler + native previews

``spec`` is an ``atlas_spec.json``, a folder holding one, or omitted for every
``tiles/*/atlas_spec.json``.

build outputs (in ``--out-dir/<delivery>/`` or ``--incoming/<delivery>/``):
    <biome>_atlas.png, manifest.json, _atlas_preview.png,
    _tile_<col>_<name>_preview.png, _report.txt
"""
import argparse
import glob
import os
import sys

from PIL import Image

import tile_atlas as ta
import tile_segment as tseg

_ART_ARTIFACTS = ("_ruler", "_native", "_contact_sheet", "_montage")

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPEC_GLOB = os.path.join(_PROJECT_DIR, "tiles", "*", "atlas_spec.json")
DEFAULT_OUT_DIR = os.path.join(_PROJECT_DIR, "Deliveries")

COMMANDS = ("all", "segment", "build", "ruler")


def resolve_specs(arg):
    """arg None -> every tiles/*/atlas_spec.json; a file -> just it;
    a dir -> <dir>/atlas_spec.json."""
    if arg is None:
        hits = sorted(glob.glob(DEFAULT_SPEC_GLOB))
        if not hits:
            raise SystemExit(f"no specs found ({DEFAULT_SPEC_GLOB})")
        return hits
    if os.path.isdir(arg):
        p = os.path.join(arg, "atlas_spec.json")
        if not os.path.isfile(p):
            raise SystemExit(f"no atlas_spec.json in {arg}")
        return [p]
    if os.path.isfile(arg):
        return [arg]
    raise SystemExit(f"spec not found: {arg}")


# --------------------------------------------------------------------------
# ruler
# --------------------------------------------------------------------------
def run_ruler(spec, step):
    """For every source the spec references, write next to the spec:
      <src>_ruler<step>.png   the ORIGINAL with a coordinate grid (rects are
                              authored in these original-pixel coordinates)
      <src>_native.png        the art-pixel-grid snap the builder crops from
                              (eyeball the detector; pin it in
                              spec['sources'][name] if it's wrong)
    """
    seen = set()
    for t in spec.get("tiles", []):
        name = t.get("src")
        if not name or name in seen:
            continue
        seen.add(name)
        src_path = os.path.join(spec["_dir"], name)
        if not os.path.isfile(src_path):
            print(f"  ruler: source missing, skipped: {name}")
            continue
        stem = os.path.splitext(name)[0]
        img = Image.open(src_path)
        out = os.path.join(spec["_dir"], f"{stem}_ruler{step}.png")
        ta.ruler_overlay(img, step, out)
        print(f"  ruler  -> {os.path.relpath(out, _PROJECT_DIR)}")
        if spec.get("native_snap", True):
            nat, cw, ch = ta.native_source(img, spec.get("sources", {}).get(name),
                                           float(spec.get("sample_frac", 0.5)))
            np_out = os.path.join(spec["_dir"], f"{stem}_native.png")
            nat.save(np_out)
            print(f"  native -> {os.path.relpath(np_out, _PROJECT_DIR)}   "
                  f"({nat.size[0]}x{nat.size[1]}, art-pixel = {cw:.3f}x{ch:.3f} src px)")


# --------------------------------------------------------------------------
# segment
# --------------------------------------------------------------------------
def _segment_sources(spec):
    """Which sheets to catalogue: spec['segment']['sources'] if given, else
    every *.png in the spec dir that isn't a generated artifact."""
    seg = spec.get("segment", {})
    if seg.get("sources"):
        return list(seg["sources"])
    out = []
    for f in sorted(os.listdir(spec["_dir"])):
        if not f.lower().endswith(".png"):
            continue
        stem = os.path.splitext(f)[0]
        if stem.startswith("_") or any(k in stem for k in _ART_ARTIFACTS):
            continue
        out.append(f)
    return out


_SEG_DEFAULTS = dict(alpha_thresh=64, key_tol=60, key_margin=40, magenta_peel=2,
                     magenta_kill=False, min_area=150, min_dim=6,
                     max_w_frac=0.30, max_h_frac=0.55, pad=1, close_px=0,
                     tile_art=16, grid_split=False, split_min_cov=0.15,
                     size_table=False, alpha_bin=128, chroma=None,
                     extra_large=False, key_rim_peel=0, shadow_detint=0.0,
                     trim_stragglers=True)


def run_segment(spec, min_area=None):
    """Catalogue every tile on each source sheet -> tiles/<biome>/catalog/<sheet>/.

    ``spec['segment']`` holds the defaults; ``spec['segment']['per_source']``
    maps a source filename to a dict that overrides them for just that sheet
    (e.g. ``{"background_mountains.png": {"magenta_peel": 1}}`` to keep the
    purple ridgeline; ``{"waterfall_tiles.png": {"magenta_kill": true}}`` to
    strip every magenta/violet pixel from art that has none of its own).
    """
    seg = spec.get("segment", {})
    keys = seg.get("keys", {})
    per_source = seg.get("per_source", {})
    base = {k: seg.get(k, v) for k, v in _SEG_DEFAULTS.items()}
    if min_area is not None:
        base["min_area"] = min_area
    sample_frac = float(spec.get("sample_frac", 0.5))
    cat_root = os.path.join(spec["_dir"], "catalog")

    for name in _segment_sources(spec):
        src_path = os.path.join(spec["_dir"], name)
        if not os.path.isfile(src_path):
            print(f"  segment: source missing, skipped: {name}")
            continue
        stem = os.path.splitext(name)[0]
        img = Image.open(src_path)
        sec_path = os.path.join(spec["_dir"], f"{stem}_sections.json")
        sections = tseg.load_sections(sec_path)
        out_dir = os.path.join(cat_root, stem)
        kw = {**base, **per_source.get(name, {})}
        role = ta._role_hint(name, spec)
        if role == "parallax":
            kw["size_table"] = False      # big scrolling layers are never cell-fit
        index, tag = tseg.write_catalog(
            img, out_dir, sheet_name=stem, sections=sections,
            native_override=spec.get("sources", {}).get(name),
            sample_frac=sample_frac, key=keys.get(name),
            grid=int(spec.get("grid", 32)), role=role, **kw)
        secs = sorted({t["section"] for t in index if t["section"]})
        print(f"  {name}: {len(index)} tiles  (mask: {tag})"
              + (f"  sections: {', '.join(secs)}" if secs else "")
              + (f"  [+{stem}_sections.json]" if sections else ""))
        print(f"    -> {os.path.relpath(out_dir, _PROJECT_DIR)}/  "
              f"(_montage.png, _contact_sheet.png, index.json, {len(index)} PNGs)")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def run_build(spec, out_root, dry_run=False):
    """Validate the spec and write the atlas delivery. Returns 0 / 1."""
    errs = ta.validate_spec(spec)
    if errs:
        print("  SPEC INVALID:")
        for e in errs:
            print(f"    - {e}")
        return 1
    built = ta.build_atlas(spec)
    dest, lines = ta.write_delivery(out_root, spec, built, dry_run=dry_run)
    print("\n".join(lines))
    if not dry_run:
        cat = ta.write_asset_catalog(spec, dest)
        doc = None
        if cat:
            _path, doc = cat
            print(f"wrote catalog.json  ({doc['count']} extra tiles + PNGs under catalog/)")
        else:
            print("catalog.json skipped (no segmentation catalogue -- run `segment` first)")
        ta.write_delivery_readme(spec, dest, built["manifest"], doc)
        print("wrote README.md")
    print(f"-> {dest}/")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _build_parser():
    ap = argparse.ArgumentParser(
        prog="main_tiles.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", metavar="{all,segment,build,ruler}")

    def spec_arg(p):
        p.add_argument("spec", nargs="?", default=None,
                       help="atlas_spec.json, a folder holding one, or omit for "
                            "every tiles/*/atlas_spec.json")

    p_all = sub.add_parser("all", help="segment then build (the default)")
    spec_arg(p_all)
    p_all.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p_all.add_argument("--incoming", default=None,
                       help="write <delivery>/ under this path instead of --out-dir")
    p_all.add_argument("--dry-run", action="store_true",
                       help="build step: previews + report only, no atlas/manifest")
    p_all.add_argument("--min-area", type=int, default=None,
                       help="segment step: min component area in source px (default 150)")

    p_seg = sub.add_parser("segment", help="catalogue every tile on each source sheet")
    spec_arg(p_seg)
    p_seg.add_argument("--min-area", type=int, default=None,
                       help="min component area in source px (default 150; raise "
                            "to drop specks, lower to keep tiny decorations)")

    p_bld = sub.add_parser("build", help="build the tile-atlas delivery")
    spec_arg(p_bld)
    p_bld.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                       help="where <delivery>/ folders go (default: Deliveries/)")
    p_bld.add_argument("--incoming", default=None,
                       help="write <delivery>/ under this path instead of --out-dir "
                            "(point it at the game repo's incoming/)")
    p_bld.add_argument("--dry-run", action="store_true",
                       help="previews + report only; no atlas PNG, no manifest.json")

    p_rul = sub.add_parser("ruler", help="coordinate-ruler + native-snap previews")
    spec_arg(p_rul)
    p_rul.add_argument("--step", type=int, default=64,
                       help="ruler grid spacing in px (default 64)")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    first = argv[0] if argv else None
    if first not in COMMANDS and first not in ("-h", "--help"):
        argv = ["all"] + argv          # default subcommand (also for bare flags)

    args = _build_parser().parse_args(argv)
    specs = resolve_specs(args.spec)
    print(f"[{args.cmd}] {len(specs)} spec(s): "
          f"{[os.path.relpath(p, _PROJECT_DIR) for p in specs]}")

    rc = 0
    for spec_path in specs:
        spec = ta.load_spec(spec_path)
        print(f"\n=== {os.path.relpath(spec_path, _PROJECT_DIR)} ===")

        if args.cmd == "ruler":
            run_ruler(spec, args.step)
        elif args.cmd == "segment":
            run_segment(spec, args.min_area)
        elif args.cmd == "build":
            rc |= run_build(spec, args.incoming or args.out_dir, args.dry_run)
        else:  # all
            run_segment(spec, args.min_area)
            rc |= run_build(spec, args.incoming or args.out_dir, args.dry_run)

    return rc


if __name__ == "__main__":
    sys.exit(main())
