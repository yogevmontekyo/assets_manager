---
name: generate-sprites
description: Run this project's sprite pipeline (main.py) to turn an AI-generated character image or a multi-row animation sheet on a flat chroma-green background into clean, palette-locked pixel-art frames. Use when asked to "generate sprites", "make a sprite sheet", "pixelate a character", "extract animation frames", or to process an image dropped in Input_Generated_Character/.
---

# Generate Sprites

Pipeline: source PNG (flat green background) → per-frame crop → cross-frame centering → aspect-preserving coverage-aware downscale → one shared locked palette (default 256 colors) → dither-free pixel-art frames + QA report.

Green-contaminated edge pixels from the source's own anti-aliasing (`looks_greenish` in [pixelate.py](../../../pixelate.py)) are kept out of both the averaged output and the palette, so a high `--n-colors` doesn't spend a palette slot on a green rim.

`--target-size` is the output **height**. Width is derived per state from that state's padded-crop aspect ratio, so frames keep their proportions instead of being squashed square. All frames of one state share the same dimensions; different states may have different widths.

## Requirements

- Python 3; `pip install -r requirements.txt` (`numpy`, `Pillow`, `opencv-python-headless`). OpenCV is only for the default `--center-method feature`; without it, centering auto-falls back to the pure-NumPy `centroid` method.
- **Source image must have a flat chroma-green background** — `is_background()` in [extract_frames.py](../../../extract_frames.py) tests `g>140 and g>r+60 and g>b+60`. The exact fill color is `(0,255,0)`. Renders with anti-aliased edges are fine; crop/edge fringe is re-cleaned automatically.
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

## Key parameters

| Flag | Default | Raise it when… | Lower it when… |
|---|---|---|---|
| `--target-size` | 64 | want taller/higher-res sprites (e.g. 96, 128) — this is the output height; width follows the aspect ratio | want chunkier pixels |
| `--n-colors` | 256 | — (256 is the max) | want a flatter, more retro look — try 12–32 |
| `--coverage-thresh` | 0.35 | thin details (antennae, fingers) vanish — try 0.20–0.30 | edges look bloated / jagged — try 0.45–0.55 |
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
- `<state>_<NN>.png` — final pixel-art frame, `target_size` tall and aspect-derived wide (same size for all frames of a state).
- `<state>_<NN>_preview.png` — 8× nearest-neighbor upscale of that frame for inspection.
- `_contact_sheet.png` — detected boundaries overlaid on the source.
- `_palette_swatch.png` — the single palette shared by every frame (no frame-to-frame color flicker).
- `_report.txt` — per-state centering method/offsets/jitter + per-frame color count + bleed check.

## Troubleshooting

- **Wrong row/frame count**: usually frames touch or there's stray noise on the background. Clean the source or split it manually; the current pipeline has no `--rows`/`--cols` override (the standalone [extract_frames.py](../../../extract_frames.py) docstring mentions one but it isn't implemented).
- **`[FAIL]` / green bleed**: the pipeline already filters greenish edge pixels from the average and the palette, so a `[FAIL]` now means real contamination — a background that isn't flat green, or `--coverage-thresh` too low letting edge blocks average in green. Raise the threshold, or tighten the source background to a solid `(0,255,0)`. As a last resort widen the green filter (`looks_greenish` margin) in [pixelate.py](../../../pixelate.py).
- **Thin features disappear**: lower `--coverage-thresh`, or raise `--target-size` so those features span more source blocks.
- **A state comes out wider/narrower than expected**: output width tracks that state's padded-crop aspect ratio, which is driven by the widest frame plus `--h-pad-frac`. Trim stray background in the source or lower `--h-pad-frac` to tighten it.
- **Frames jitter horizontally in-game**: check `_report.txt`'s `core-jitter … -> Npx` for that state. If N is large with `feature`, try `--center-method centroid`; if a cape/hair still drags it, the source frames themselves differ too much — author tighter frames. `bbox` (old method) is the most appendage-sensitive and usually worst.
- **A previously-fine sheet shifted after adding centering**: force the old behavior with `--center-method bbox`.

## Assembling a sheet

`main.py` writes individual frames, not a packed sheet. All frames of one state share the same size, so to montage a state into one horizontal strip (like the existing `_walk_final_strip.png`), concatenate its `*_preview.png` frames in order — with PIL, paste each at `x = i * frame_width` onto a `(n * frame_width, frame_height)` canvas. Do this per state; two states can have different frame widths, so don't mix them on one row without re-padding.
