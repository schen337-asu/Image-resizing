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

For the Real-ESRGAN-powered version, run `resizer2.py` and see `README_resizer2.md`.
