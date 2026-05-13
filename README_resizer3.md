# Resizer3 (MLX Enhancement + Apple Quantization)

`resizer3.py` is a batch JPEG resizer with optional MLX-based enhancement and
Apple-native low-bit activation quantization, designed for Apple Silicon Macs.

Sub-version: **v3.1**

---

## What It Does

- Resizes JPEGs with Pillow `LANCZOS` resampling
- Optionally enhances images using a lightweight MLX pipeline running on the Apple GPU
- Supports optional activation quantization (`off` / `8-bit` / `4-bit`) to emulate quantized inference output
- Processes `.jpg` / `.jpeg` files recursively, preserving relative folder structure in the output
- Preserves EXIF and ICC profile metadata when available
- Skips output files that already exist (no overwrite)

---

## Interface

Resizer3 opens a **single dark-themed two-column window**:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Resizer3  v3.1                                                      │
├─────────────────────────────┬────────────────────────────────────────┤
│  ┌─ Folders ──────────────┐ │  Processing                            │
│  │  Source folder         │ │                                        │
│  │  [_________________] … │ │  [████████████░░░░░░░  62%          ]  │
│  │  Destination folder    │ │                                        │
│  │  [_________________] … │ │  [47/76]  OK: photo_001.jpg           │
│  └────────────────────────┘ │  ✓ 45  ·  ↷ 2  ·  ✗ 0               │
│  ┌─ Resize ───────────────┐ │                                        │
│  │  Multiplier ratio: 0.5 │ │                      [Cancel] [Close]  │
│  └────────────────────────┘ │                                        │
│  ┌─ MLX Enhancement ──────┐ │                                        │
│  │  ☑ Enable MLX …        │ │                                        │
│  │  Blend strength: 0.6   │ │                                        │
│  │  Quantization: off     │ │                                        │
│  └────────────────────────┘ │                                        │
│                    [Cancel] [Start]                                  │
└─────────────────────────────┴────────────────────────────────────────┘
```

**Left column — Settings** (always visible):
- Source and destination folder pickers
- Multiplier ratio input
- MLX enhancement toggle, blend strength, and quantization selector

**Right column — Progress** (always visible):
- Canvas-based progress bar with a centred `%` text overlay, updated live
- Current file status label
- Running counts: `✓ succeeded · ↷ skipped · ✗ failed`
- Cancel (stops mid-run) and Close (appears when done) buttons

When **Start** is clicked, all settings inputs are disabled so the run configuration remains readable while processing is under way.

---

## Requirements

- **macOS** with Apple Silicon (M-series) for MLX GPU acceleration
- Supported Python: 3.10–3.12
- Recommended Python: 3.12.x
- Python 3.13+ is not recommended (PyTorch/torchvision compatibility may vary)
- Dependencies in `requirements.txt`

Key runtime dependencies for `resizer3.py`:

| Package | Purpose |
|---|---|
| `mlx` | Apple-native ML framework (GPU acceleration) |
| `Pillow` | JPEG decode, resize, metadata handling |
| `numpy` | Array conversion between PIL and MLX |

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
python resizer3.py
```

The window opens immediately. Fill in:

1. **Source folder** — root directory containing JPEG images (subfolders are scanned recursively)
2. **Destination folder** — output root (must be different from and not inside the source folder)
3. **Multiplier ratio** — e.g. `0.5` to halve, `1.25` to scale up, `2` to double
4. **MLX Enhancement** — enable to run the MLX tone/contrast pipeline on the Apple GPU
   - **Blend strength** (`0.0`–`1.0`) — how much of the enhanced result is blended with the baseline resize
   - **Apple quantization** — `off` (default), `8-bit`, or `4-bit` to emulate quantized inference

Click **Start** to begin. The progress bar and counters update in real time. Click **Cancel** to stop mid-run.

---

## Output File Naming

Output files are placed in the destination folder mirroring the source subfolder tree.

Example:

- Source: `/Users/name/Photos/Trip`
- Destination: `/Users/name/Photos/Trip-resized`
- Ratio: `0.5`

A source file at:

`/Users/name/Photos/Trip/Italy/img001.jpg`

is saved to:

`/Users/name/Photos/Trip-resized/Italy/img001_x0p5.jpg`

If the output file already exists it is **skipped** (not overwritten).

---

## MLX Enhancement Details

When enhancement is enabled, each image goes through this pipeline on the Apple GPU via MLX:

1. **Baseline resize** — Pillow `LANCZOS` to the target dimensions
2. **Optional early quantization** — if `8-bit` or `4-bit` is selected, activations are quantized before tone adjustment
3. **Contrast boost** — `contrast = 1.0 + (0.40 × blend_strength)`
4. **Gamma adjustment** — `gamma = 1.0 − (0.20 × blend_strength)`
5. **Optional post-op quantization** — applied again after tone ops to emulate quantized inference output
6. **Blend** — the enhanced result is blended with the baseline resize at the given `blend_strength`

Setting `blend_strength` to `0.0` returns the plain baseline; `1.0` returns the fully enhanced image.

**Quantization modes:**

| Mode | Levels | Effect |
|---|---|---|
| `off` | — | No quantization; full float32 pipeline |
| `8-bit` | 255 | Slight rounding artefacts; faster inference emulation |
| `4-bit` | 15 | More pronounced posterisation; aggressive quantization test |

---

## Parallel Processing

| Enhancement | Processing mode |
|---|---|
| Disabled | Parallel thread pool (up to 8 workers) |
| Enabled | Serial (single thread, GPU handles parallelism internally) |

MLX handles GPU-level parallelism internally; serial CPU dispatch avoids contention.

---

## Troubleshooting

### `MLX is not installed for the active interpreter`

```bash
pip install mlx
```

MLX requires macOS 13.5+ and an Apple Silicon or Apple AMD GPU.

### Import errors (`PIL`, `numpy`)

Ensure the virtual environment is active and dependencies are installed:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Slow processing with MLX enabled

- MLX compiles shaders on first use — first-image latency is expected; subsequent images are faster
- Lower `blend_strength` reduces the GPU work per image
- `4-bit` quantization adds rounding ops; `off` is fastest

### No files processed

Only `.jpg` and `.jpeg` files (case-insensitive) are scanned. Verify the source folder contains images with those extensions.

### Window does not open (Tkinter unavailable)

If Tk is unavailable, the script falls back to interactive terminal prompts for all settings.

---

## File

- Script: `resizer3.py`
- This guide: `README_resizer3.md`
