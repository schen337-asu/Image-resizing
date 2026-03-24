# Image Resizer: Purpose and Usage Guide

## Purpose

This application batch-processes JPEG images with high-quality resizing and optional visual enhancement.

It is designed to:
- Resize `.jpg` and `.jpeg` images using Pillow `LANCZOS` resampling.
- Optionally enhance dynamic range and contrast (CLAHE + tone curve + mild sharpening).
- Process images recursively in all child folders under a selected root folder.
- Save outputs to a sibling folder named in this format:

`<original folder name>+<multiplier>`

- Preserve the relative subfolder structure in the output tree.

## Requirements

- macOS (for native folder picker via `osascript`; fallback manual path input is supported)
- Python 3.13+ recommended
- Dependencies listed in `requirements.txt`

## Setup

From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Application

```bash
python resizer.py
```

The app will prompt for:
1. Input folder (via folder picker UI)
2. Multiplier ratio (for example: `0.5`, `1.25`, `2`)
3. Whether to apply enhancement (`Y/n`)

## Output Behavior

If input folder is:

`/Users/name/Pictures/Holidays`

and multiplier is:

`0.5`

then output root becomes:

`/Users/name/Pictures/Holidays+0.5`

A source image at:

`/Users/name/Pictures/Holidays/Trips/Italy/img001.jpg`

is saved to:

`/Users/name/Pictures/Holidays+0.5/Trips/Italy/img001_x0p5.jpg`

## Notes

- Only `.jpg` and `.jpeg` files are processed.
- Existing files with the same output name will be overwritten.
- JPEG metadata such as EXIF and ICC profile is preserved when available.
- If the folder picker is canceled or unavailable, the app asks for a manual directory path.

## Resizer2 (Real-ESRGAN Version)

For the Real-ESRGAN-powered variant, run `resizer2.py` and follow `README_resizer2.md`.

## Troubleshooting

- If imports fail, ensure the virtual environment is activated and dependencies are installed:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

- If the folder picker does not open, manually enter a full directory path when prompted.
