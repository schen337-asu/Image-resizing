# Image Resizer

High-quality JPEG batch resizer with optional dynamic range and contrast enhancement.

## Features

- High-quality resize using Pillow `LANCZOS`
- Optional enhancement pipeline:
  - CLAHE on luminance
  - gentle S-curve tone mapping
  - light unsharp mask
- Recursive processing of all child folders
- Output folder naming format:

`<original folder name>+<multiplier>`

- Preserves relative subfolder structure in output
- Preserves EXIF/ICC metadata when available

## Requirements

- Supported Python: 3.10–3.12
- Recommended Python: 3.12.x
- Python 3.13+ is not recommended yet (PyTorch/torchvision compatibility may vary)
- Dependencies in `requirements.txt`

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python resizer.py
```

## How It Works

When you run `resizer.py`, the app will:
1. Open a folder picker to choose the input root directory
2. Ask for a resize multiplier (example: `0.5`, `1.25`, `2`)
3. Ask whether to enable enhancement (`Y/n`)
4. Process all `.jpg` and `.jpeg` files recursively
5. Save outputs to a sibling folder named `<input-folder>+<multiplier>`

Example:

- Input root: `/Users/name/Pictures/Holidays`
- Multiplier: `0.5`
- Output root: `/Users/name/Pictures/Holidays+0.5`

A file:

`/Users/name/Pictures/Holidays/Trips/Italy/img001.jpg`

becomes:

`/Users/name/Pictures/Holidays+0.5/Trips/Italy/img001_x0p5.jpg`

## Notes

- Only `.jpg` and `.jpeg` are processed.
- If the folder picker is unavailable/canceled, manual path input is supported.
- Existing output files with the same name are overwritten.

## Detailed Guide

For full instructions and troubleshooting, see `HOW_TO_USE.md`.

## Resizer2 (Real-ESRGAN)

`resizer2.py` is currently tracked as sub-version `v2.1`.

Recent `v2.1` changes:

- One Tkinter settings dialog for source, destination, resize ratio, model selection, blend strength, and tile size.
- Optional tile benchmark mode to test one image with multiple tile sizes before the full batch starts.
- Overall progress window replaces per-file terminal progress logs.
- Existing output files with matching names are skipped instead of being overwritten.
- Real-ESRGAN settings now support model choice, tile-size tuning, and benchmark-assisted tile selection.

For the Real-ESRGAN-powered version, run `resizer2.py` and see `README_resizer2.md`.

## Resizer3 (MLX + Apple Quantization)

`resizer3.py` is currently tracked as sub-version `v3.1`.

`resizer3.py` is the Apple-focused variant that uses MLX for enhancement and
adds optional low-bit quantization controls for faster processing on Apple Silicon.

Key `v3.1` capabilities:

- One Tkinter settings dialog for source, destination, ratio, enhancement toggle, blend strength, and quantization mode.
- MLX-based enhancement pipeline with quantization options: `off`, `8-bit`, `4-bit`.
- Overall progress window with live status/counters and a `Cancel` button to stop jobs mid-run.
- Existing output files with matching names are skipped instead of overwritten.

Run:

```bash
python resizer3.py
```

Dependency note:

- `resizer3.py` requires `mlx` (included in `requirements.txt`).
