# Resizer2 (Real-ESRGAN Enhancement)

`resizer2.py` is a batch JPEG resizer with optional Real-ESRGAN image enhancement.

## What It Does

- Resizes JPEGs with Pillow `LANCZOS`
- Optionally enhances images using Real-ESRGAN (`RealESRGAN_x4plus`)
- Processes `.jpg` / `.jpeg` files recursively
- Preserves relative folder structure in output
- Preserves EXIF/ICC metadata when available

Output folder format:

`<original folder name>+<multiplier>`

---

## Requirements

- Python 3.12+ recommended
- Dependencies from `requirements.txt`

Key model dependencies:

- `torch`
- `basicsr`
- `realesrgan`

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run

```bash
python resizer2.py
```

The script will prompt for:

1. Input folder (via macOS folder picker; fallback to typed path)
2. Resize multiplier (examples: `0.5`, `1.25`, `2`)
3. Whether to apply Real-ESRGAN enhancement (`Y/n`)
4. Real-ESRGAN blend strength (`0.0` to `1.0`, default `0.6`) if enhancement is enabled

---

## Example

- Input folder: `/Users/name/Pictures/Holidays`
- Multiplier: `0.5`
- Output folder: `/Users/name/Pictures/Holidays+0.5`

A file:

`/Users/name/Pictures/Holidays/Trips/Italy/img001.jpg`

becomes:

`/Users/name/Pictures/Holidays+0.5/Trips/Italy/img001_x0p5.jpg`

---

## Real-ESRGAN Model Notes

`resizer2.py` uses this default model weight:

- `RealESRGAN_x4plus.pth` (downloaded from official release URL)

Device selection is automatic in this order:

1. CUDA
2. Apple MPS
3. CPU

First run may be slower because model files are downloaded and cached.

---

## Troubleshooting

### Import errors (`torch`, `realesrgan`, `basicsr`, `PIL`, `numpy`)

Reinstall dependencies in your active environment:

```bash
pip install -r requirements.txt
```

If you see `No module named 'torchvision.transforms.functional_tensor'`,
use `resizer2.py` directly (it includes a compatibility shim) and make sure
`requirements.txt` is installed so `torchvision` stays in the tested range.

### Slow processing

- Use enhancement only when needed
- Keep blend strength lower for lighter effect
- Ensure you are running in an accelerated environment (CUDA/MPS if available)

### No files processed

Only `.jpg` and `.jpeg` files are scanned.

---

## File

- Script: `resizer2.py`
- This guide: `README_resizer2.md`
