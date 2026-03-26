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

- Supported Python: 3.10–3.12
- Recommended Python: 3.12.x
- Python 3.13+ is not recommended yet (PyTorch/torchvision compatibility may vary)
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

1. CUDA (NVIDIA GPU)
2. Apple MPS (Apple Silicon / AMD GPU via Metal)
3. CPU

The selected device is printed at startup, e.g. `[Real-ESRGAN] Using Apple GPU (MPS)`.

**Tiling:** A default tile size of 512 px (`DEFAULT_TILE_SIZE`) is used, splitting large
images into overlapping tiles before GPU inference. This prevents out-of-memory errors on
Apple Silicon and keeps throughput stable. Set `tile_size=0` in `RealESRGANEnhancer()` to
disable tiling (only safe for small images or ample unified memory).

**Half precision (FP16):** Enabled automatically for CUDA. Kept at FP32 for MPS because
PyTorch MPS FP16 can produce NaN values in certain Real-ESRGAN operations.

**Parallel processing:** When enhancement is disabled, images are resized in parallel
using a thread pool (up to 8 workers) for faster CPU-bound throughput.

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

- Check startup output — it should say `Using Apple GPU (MPS)` or `Using NVIDIA GPU (CUDA)`
- If it says `Using CPU`, your PyTorch build may not have MPS support (`torch.backends.mps.is_available()` returns False)
- Use enhancement only when needed; lower blend strength reduces the ESRGAN contribution
- For very large images, the default `tile_size=512` keeps GPU memory usage bounded
- Non-enhanced resizes run in parallel automatically (thread pool)

### No files processed

Only `.jpg` and `.jpeg` files are scanned.

---

## File

- Script: `resizer2.py`
- This guide: `README_resizer2.md`
