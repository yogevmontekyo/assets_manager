---
name: generate-sprites
description: Run this project's sprite pipeline (main.py) to turn an AI-generated character image or a multi-row animation sheet on a flat chroma-green background into clean, palette-locked pixel-art frames. Use when asked to "generate sprites", "make a sprite sheet", "pixelate a character", "extract animation frames", or to process an image dropped in Input_Generated_Character/.
---

# Generate Sprites

Pipeline: source PNG (flat green background) → drift-corrected full-res frames → aspect-preserving coverage-aware downscale → one shared locked palette → dither-free pixel-art frames + QA report.

`--target-size` is the output **height**. Width is derived per state from that state's padded-crop aspect ratio, so frames keep their proportions instead of being squashed square. All frames of one state share the same dimensions; different states may have different widths.

## Requirements

- Python 3 with `Pillow` and `numpy` (no requirements file; install with `pip install pillow numpy` if missing).
- **Source image must have a flat chroma-green background** — `is_background()` in [extract_frames.py](../../../extract_frames.py) tests `g>140 and g>r+60 and g>b+60`. The exact fill color is `(0,255,0)`. Renders with anti-aliased edges are fine; crop/edge fringe is re-cleaned automatically.
- Layout the pipeline understands with no special-casing:
  - One full-body character on green → detected as 1 state / 1 frame.
  - Multi-row sheet: one animation state per row, frames left-to-right within the row.

## Steps

1. Find the input. Default location is [Input_Generated_Character/](../../../Input_Generated_Character/); accept an explicit path if the user gives one.
2. Decide state names. One per detected row, top to bottom (e.g. `walk jump`). If unknown, omit `--state-names` and the pipeline uses `state0, state1, …`. Passing the wrong count is ignored with a warning.
3. Run from the project root:
   ```
   python main.py <input.png> --out-dir Output_Sprite_Sheet --target-size 64 --n-colors 12 --state-names <names...>
   ```
4. Read `Output_Sprite_Sheet/_report.txt`. It lists each state's derived output `WxH` (aspect check) and one line per frame; every frame line must end in `[OK]`. `[FAIL]` means green bled into the palette — see Troubleshooting.
5. Show the user `Output_Sprite_Sheet/_contact_sheet.png` (detected row/frame boundaries on the source) and a few `*_preview.png` files to confirm detection and quality.

## Key parameters

| Flag | Default | Raise it when… | Lower it when… |
|---|---|---|---|
| `--target-size` | 64 | want taller/higher-res sprites (e.g. 96, 128) — this is the output height; width follows the aspect ratio | want chunkier pixels |
| `--n-colors` | 12 | character has more distinct shades | want a flatter, more retro look |
| `--coverage-thresh` | 0.35 | thin details (antennae, fingers) vanish — try 0.20–0.30 | edges look bloated / jagged — try 0.45–0.55 |
| `--h-pad-frac` / `--v-pad-frac` | 0.15 | character clips the frame edge | too much empty margin |

## Outputs (in `--out-dir`)

- `<state>_<NN>_raw.png` — full-res padded, horizontally drift-corrected crop (vertical motion preserved).
- `<state>_<NN>.png` — final pixel-art frame, `target_size` tall and aspect-derived wide (same size for all frames of a state).
- `<state>_<NN>_preview.png` — 8× nearest-neighbor upscale of that frame for inspection.
- `_contact_sheet.png` — detected boundaries overlaid on the source.
- `_palette_swatch.png` — the single palette shared by every frame (no frame-to-frame color flicker).
- `_report.txt` — per-frame color count + bleed check.

## Troubleshooting

- **Wrong row/frame count**: usually frames touch or there's stray noise on the background. Clean the source or split it manually; the current pipeline has no `--rows`/`--cols` override (the standalone [extract_frames.py](../../../extract_frames.py) docstring mentions one but it isn't implemented).
- **`[FAIL]` / green bleed**: background isn't flat enough, or `--coverage-thresh` is too low letting edge blocks average in green. Raise the threshold, or tighten the source background to a solid `(0,255,0)`.
- **Thin features disappear**: lower `--coverage-thresh`, or raise `--target-size` so those features span more source blocks.
- **A state comes out wider/narrower than expected**: output width tracks that state's padded-crop aspect ratio, which is driven by the widest frame plus `--h-pad-frac`. Trim stray background in the source or lower `--h-pad-frac` to tighten it.
- **Frames jitter horizontally in-game**: that's drift the pipeline already corrects on the silhouette bbox center; if it persists, the silhouette width itself changes a lot between frames (e.g. an arm extending) — accept it or author tighter source frames.

## Assembling a sheet

`main.py` writes individual frames, not a packed sheet. All frames of one state share the same size, so to montage a state into one horizontal strip (like the existing `_walk_final_strip.png`), concatenate its `*_preview.png` frames in order — with PIL, paste each at `x = i * frame_width` onto a `(n * frame_width, frame_height)` canvas. Do this per state; two states can have different frame widths, so don't mix them on one row without re-padding.
