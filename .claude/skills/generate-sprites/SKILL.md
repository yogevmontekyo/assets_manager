---
name: generate-sprites
description: Run this project's sprite pipeline (main.py) to turn an AI-generated character image or a multi-row animation sheet on a flat chroma-green background into clean, palette-locked pixel-art frames. Use when asked to "generate sprites", "make a sprite sheet", "pixelate a character", "extract animation frames", or to process an image dropped in Input_Generated_Character/.
---

# Generate Sprites

Pipeline: source PNG (green-screen background) → per-frame crop → cross-frame centering → aspect-preserving coverage-aware downscale (per-block **median**, keeps outlines crisp) → one shared locked palette (`--n-colors` cap, near-duplicates merged) → dither-free pixel-art frames + QA report.

The green key is deliberately **wide** — the source "green" ranges from bright lime to a murky dark green along edges — and green-contaminated edge pixels (including faint-green near-black outline pixels) are kept out of both the block color and the palette (`looks_greenish` in [pixelate.py](../../../pixelate.py)). The shared palette also drops near-duplicate colors (`--palette-merge-dist`) so frames don't shimmer at high `--n-colors`.

`--target-size` is the output **height**. Width is derived per state from that state's padded-crop aspect ratio, so frames keep their proportions instead of being squashed square. All frames of one state share the same dimensions; different states may have different widths.

## Requirements

- Python 3; `pip install -r requirements.txt` (`numpy`, `Pillow`, `opencv-python-headless`). OpenCV is only for the default `--center-method feature`; without it, centering auto-falls back to the pure-NumPy `centroid` method.
- **Source background** — either a green screen or a real alpha channel:
  - Green screen: `is_background()` (same in [extract_frames.py](../../../extract_frames.py) and [pixelate.py](../../../pixelate.py), keep in sync) tests `g>115 and g>r+42 and g>b+42`, tolerating a lime→dark-green range, not just `(0,255,0)`. The character must contain no strongly-green pixels.
  - Already transparent (RGBA/PA source, no green): `main.py` auto-detects it and paints pixels with `alpha < --alpha-thresh` (default 64) onto the green sentinel, then runs normally. The default drops faint halos / motion-blur trails that would otherwise split frame detection into slivers; lower to ~16 to keep faint translucency.
  - Anti-aliased edges are fine either way; fringe is re-cleaned.
- Layout the pipeline understands with no special-casing:
  - One full-body character on green → detected as 1 state / 1 frame.
  - Multi-row sheet: one animation state per row, frames left-to-right within the row.

## Steps

1. Put the source image(s) in [Input_Generated_Character/](../../../Input_Generated_Character/), or note an explicit path the user gave.
2. Decide state names. One per detected row, top to bottom (e.g. `walk jump`). If unknown, omit `--state-names` and the pipeline uses `state0, state1, …`. Passing the wrong count is ignored with a warning. Note: `--state-names` applies to *every* image in a batch, so only pass it when all inputs share the same row layout.
3. Run from the project root:
   ```
   # process every image in Input_Generated_Character/
   python main.py --target-size 96 --state-names <names...>

   # or one specific file / folder
   python main.py <path> --target-size 96 --n-colors 32 --state-names <names...>
   ```
   - `input` is optional; omitted → every image (`.png/.jpg/.jpeg/.bmp/.webp`) in `Input_Generated_Character/`.
   - `--out-dir` defaults to the project's `Output_Sprite_Sheet/`; pass it only to write elsewhere.
   - Folder source (the default) → each image's outputs land in `Output_Sprite_Sheet/<image-name>/`. A single explicit file writes straight into `Output_Sprite_Sheet/`.
4. For each output folder, read `_report.txt`. It lists each state's derived output `WxH` (aspect check) and one line per frame; every frame line must end in `[OK]`. `[FAIL]` means green bled into the palette — see Troubleshooting.
5. Show the user each `_contact_sheet.png` (detected row/frame boundaries on the source) and a few `*_preview.png` files to confirm detection and quality.

## Watching the result (centering / snap QA)

[preview_anim.py](../../../preview_anim.py) turns output frames into things you can actually watch:

```
python preview_anim.py                              # every state under Output_Sprite_Sheet/
python preview_anim.py Output_Sprite_Sheet/Player1  # one character
python preview_anim.py <folder> --kind raw --fps 6  # pre-downscale frames only
```

Per state it writes `<state>_anim.gif` (final frames, looping, NEAREST-scaled), `<state>_raw_anim.gif` (post-centering / **pre-downscale** — isolates a centering fault from a snap fault), and `<state>_onion.png` (all frames mean-blended: a well-centered anim has a **crisp head and torso** with only the limbs/cape blurred; if the head smears, centering is off). A magenta line marks frame center in every view. `--strip` adds a filmstrip. `--bg checker|gray|green|magenta`.

### Objective centering check

[make_test_sheet.py](../../../make_test_sheet.py) builds a synthetic sheet whose torso/head sit at a **fixed** x while a big cape, hair tuft, arm and jump arc move around them, and paints the torso a unique blue so drift is measurable:

```
python make_test_sheet.py
python main.py Input_Generated_Character/_test_sheet.png --state-names walk jump
python make_test_sheet.py --verify Output_Sprite_Sheet/_test_sheet   # PASS/FAIL, body-center spread in px
```

Note: on this adversarial sheet the `_report.txt` `core-jitter` line reads high (~15–20 px) because it measures the head band, which here is full of the deliberately-flapping hair/cape — that is *not* a centering failure. `--verify`, which tracks the rigid blue torso, is the real pass/fail (expect < ~1 px). `_test_sheet.png` (leading `_`) is skipped by the default folder batch; run it by name.

## Key parameters

| Flag | Default | Raise it when… | Lower it when… |
|---|---|---|---|
| `--target-size` | 64 | want taller/higher-res sprites (e.g. 96, 128) — this is the output height; width follows the aspect ratio | want chunkier pixels |
| `--n-colors` | 256 | it's a **cap** now, not a target — the real count is what the art needs after merge (often 50–150) | want a flatter, more retro look — try 12–32 |
| `--palette-merge-dist` | 10 | frames still shimmer / too many near-identical colors — try 14–20 | palette looks banded / lost shading — try 4–6, or 0 to keep every entry |
| `--transparent` | off | want RGBA output with a clear background instead of the green fill (alpha is binary — the silhouette is a hard edge) | — |
| `--coverage-thresh` | 0.35 | thin details (antennae, fingers) vanish — try 0.20–0.30 | edges look bloated / jagged — try 0.45–0.55 |
| `--downscale` | `median` | output looks soft/muddy — `median` (default) keeps outlines crisp; `center` (+ `--sample-frac`, default 0.6) is crisper still | too speckly / noisy — `mean` for the old soft look, or raise `--sample-frac` |
| `--h-pad-frac` / `--v-pad-frac` | 0.15 | character clips the frame edge | too much empty margin |
| `--center-method` | `feature` | — | use `centroid` (no OpenCV) or `bbox` (old behavior) if `feature` mis-aligns a specific sheet |

### Centering (`--center-method`)

Aligns the character to the same x in every frame of a state so it doesn't slide around in playback. The problem it solves: a billowing cape, swinging hand, flying hair or held item all move a naive center estimate.

- **`feature`** (default) — ORB keypoint matches between each frame and a reference frame; the shift is the MAD-robust median of matched-point dx. Rigid features (face, collar, belt) agree on one shift; cape/hair/hand keypoints are inconsistent and get rejected as outliers. Handles front views, side walks and jumps. If a state's raw frames barely drift (≤3 px) the ORB step is skipped and `centroid` is used, so already-aligned sheets aren't perturbed. Needs OpenCV.
- **`centroid`** — center of mass of the head+torso band (skips hair spikes and the leg/foot zone). Pure NumPy. Good for idle/walk; a large one-sided cape still pulls it a few px, and it drifts on jumps.
- **`bbox`** — midpoint of the full silhouette bbox. The original method; any one-sided appendage moves it by half its reach. Kept for comparison.

`_report.txt` logs the method used, the per-frame offsets, and `core-jitter <before>px -> <after>px` (head-centroid spread across frames — aim for ≤ ~2 px).

## Outputs (in `--out-dir`, or `--out-dir/<image-name>/` for a folder batch)

- `<state>_<NN>_raw.png` — full-res padded, cross-frame-centered crop (vertical motion preserved).
- `<state>_<NN>.png` — final pixel-art frame, `target_size` tall and aspect-derived wide (same size for all frames of a state). RGBA with a transparent background when `--transparent` is passed (QA files stay RGB).
- `<state>_<NN>_preview.png` — 8× nearest-neighbor upscale of that frame for inspection.
- `_contact_sheet.png` — detected boundaries overlaid on the source.
- `_palette_swatch.png` — the single palette shared by every frame (auto-shrinks swatch width for large palettes).
- `_report.txt` — per-state centering method/offsets/jitter + per-frame color count + bleed check.

## Troubleshooting

- **Wrong row/frame count**: `detect_rows`/`detect_cols` threshold the projection and drop sliver bands, but frames that truly touch still merge, and a long motion-blur trail can still add a phantom frame. For an alpha source, raise `--alpha-thresh` (trails are faint). Otherwise clean/split the source manually — there's no `--rows`/`--cols` override (the standalone [extract_frames.py](../../../extract_frames.py) docstring mentions one but it isn't implemented).
- **`[FAIL]` / green bleed**: the wide key + `looks_greenish` filter already strip green from the block color and the palette, so a `[FAIL]` means real contamination — a background too far from green, or `--coverage-thresh` too low. Raise the threshold, or as a last resort widen `looks_greenish` (`margin` / `dark_margin`) in [pixelate.py](../../../pixelate.py).
- **Colors shift/shimmer slightly between frames** (mostly visible at high `--n-colors`): near-duplicate palette entries that boundary pixels flip between. Raise `--palette-merge-dist` (default 10 → try 16). It only merges colors closer than that distance, so quality cost is minimal until ~20.
- **Thin features disappear**: lower `--coverage-thresh`, or raise `--target-size` so those features span more source blocks.
- **Output looks soft/blurry**: each source block's color is now a per-channel **median** of its foreground pixels (`--downscale median`, default) instead of a mean, so interior outlines/creases stay sharp. For extra crunch use `--downscale center --sample-frac 0.5`; to get the old soft look back use `--downscale mean`. Raising `--target-size` also helps (thin outlines get more output pixels).
- **A state comes out wider/narrower than expected**: output width tracks that state's padded-crop aspect ratio, which is driven by the widest frame plus `--h-pad-frac`. Trim stray background in the source or lower `--h-pad-frac` to tighten it.
- **Frames jitter horizontally in-game**: run `preview_anim.py` on that state and look at `<state>_onion.png` — if the head/torso is crisp, centering is fine and the wobble is real limb/cape motion. If the head smears, check `_report.txt`'s `core-jitter … -> Npx`; try `--center-method centroid`, then `bbox`. Beware: `core-jitter` measures the head band, so a bobbing hat/hair inflates it even when the body is steady — confirm against the onion or `make_test_sheet.py --verify`.
- **A previously-fine sheet shifted after adding centering**: force the old behavior with `--center-method bbox`.

## Backgrounds (separate tool)

This skill is for characters/sprites. To snap an AI-generated "pixel-art" **background** to its true pixel grid and upscale it crisply, use the other entrypoint:

```
python main_background.py            # every image in background_image/ -> background_image/snapped/
python main_background.py <path> --fit pad --colors 48
python main_background.py <sheet> --transparent            # key out magenta -> RGBA
python main_background.py <sheet> --key-color green        # or green / #rrggbb / r,g,b
```

[main_background.py](../../../main_background.py) is the batch wrapper; [snap_background.py](../../../snap_background.py) has the method (native-grid detection, centre-cell median sampling, optional palette snap, NEAREST upscale) and handles multi-panel comparison sheets. `--transparent` makes the key colour (default magenta, `--key-tol` slack) transparent in the RGBA output.

## Assembling a sheet

`main.py` writes individual frames, not a packed sheet. All frames of one state share the same size, so to montage a state into one horizontal strip (like the existing `_walk_final_strip.png`), concatenate its `*_preview.png` frames in order — with PIL, paste each at `x = i * frame_width` onto a `(n * frame_width, frame_height)` canvas. Do this per state; two states can have different frame widths, so don't mix them on one row without re-padding.
