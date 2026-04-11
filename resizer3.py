"""Batch resize JPEG images with optional MLX enhancement and quantization.

Resizer3 sub-version: v3.1

All processing settings are collected from one Tkinter dialog:
- Source and destination directories
- Resize multiplier ratio
- Optional MLX enhancement blend strength
- Optional Apple-style activation quantization (off / 8-bit / 4-bit)

It resizes every .jpg/.jpeg image in the source directory (including child
folders) and saves output files in the selected destination directory while
preserving relative subfolder structure.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageOps


RESIZER3_SUBVERSION = "v3.1"


@dataclass
class Settings:
    """All processing parameters collected from the dialog."""

    source: Path
    destination: Path
    ratio: float
    apply_enhancement: bool
    blend_strength: float
    quant_bits: int


def parse_ratio_text(value: str) -> float:
    """Parse and validate a positive resize multiplier ratio."""
    try:
        ratio = float(value.strip())
    except ValueError as exc:
        raise ValueError("Multiplier ratio must be a number, e.g. 0.5 or 1.25") from exc

    if ratio <= 0:
        raise ValueError("Multiplier ratio must be greater than 0")
    return ratio


def parse_blend_strength(value: str) -> float:
    """Parse and validate MLX blend strength in the range 0.0-1.0."""
    stripped = value.strip()
    if stripped == "":
        return 0.6
    try:
        strength = float(stripped)
    except ValueError as exc:
        raise ValueError("Blend strength must be a number, e.g. 0.5 or 1.0") from exc
    if strength < 0.0 or strength > 1.0:
        raise ValueError("Blend strength must be between 0.0 and 1.0")
    return strength


def parse_quant_bits_text(value: str) -> int:
    """Parse quantization selection into off / 8-bit / 4-bit."""
    stripped = value.strip().lower()
    if stripped in {"", "off", "none", "0", "0-bit"}:
        return 0
    if stripped in {"8", "8-bit", "int8"}:
        return 8
    if stripped in {"4", "4-bit", "int4"}:
        return 4
    raise ValueError("Quantization must be one of: off, 8-bit, 4-bit")


def ratio_suffix(ratio: float) -> str:
    """Create a filename-safe, human-readable suffix for the ratio."""
    ratio_str = f"{ratio:.4f}".rstrip("0").rstrip(".")
    return ratio_str.replace(".", "p")


def resized_dimensions(width: int, height: int, ratio: float) -> tuple[int, int]:
    """Return clamped target dimensions based on the ratio."""
    new_width = max(1, round(width * ratio))
    new_height = max(1, round(height * ratio))
    return new_width, new_height


def validate_destination_directory(source_directory: Path, destination_directory: Path) -> None:
    """Reject unsafe destination choices that would mix output into the source tree."""
    if destination_directory == source_directory:
        raise SystemExit("Error: Source and destination directories must be different.")
    if destination_directory.is_relative_to(source_directory):
        raise SystemExit(
            "Error: Destination directory cannot be inside the source directory. "
            "Choose a separate output folder."
        )


def iter_jpegs(directory: Path) -> list[Path]:
    """Collect .jpg/.jpeg files recursively from a directory tree."""
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )


class MlxEnhancer:
    """MLX-based enhancement with optional low-bit activation quantization."""

    def __init__(self, blend_strength: float, quant_bits: int) -> None:
        self.blend_strength = blend_strength
        self.quant_bits = quant_bits
        try:
            import mlx.core as mx  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "MLX is not installed for the active interpreter. "
                "Install with: pip install mlx"
            ) from exc
        self.mx = mx

    def _quantize(self, x):
        if self.quant_bits == 0:
            return x
        levels = (1 << self.quant_bits) - 1
        return self.mx.round(self.mx.clip(x, 0.0, 1.0) * levels) / levels

    def enhance(self, image: Image.Image, ratio: float) -> Image.Image:
        """Resize then run lightweight MLX enhancement and blend."""
        rgb = image.convert("RGB")
        target_size = resized_dimensions(rgb.width, rgb.height, ratio)
        baseline = rgb.resize(target_size, resample=Image.Resampling.LANCZOS)

        baseline_np = np.asarray(baseline, dtype=np.float32) / 255.0
        x = self.mx.array(baseline_np)

        # Quantize activations early if requested.
        x = self._quantize(x)

        # Lightweight tone + contrast adjustments on Apple GPU via MLX.
        contrast = 1.0 + (0.40 * self.blend_strength)
        gamma = 1.0 - (0.20 * self.blend_strength)
        x = self.mx.clip((x - 0.5) * contrast + 0.5, 0.0, 1.0)
        x = self.mx.power(x, gamma)

        # Quantize again to emulate post-op quantized inference output.
        x = self._quantize(x)
        x = self.mx.clip(x, 0.0, 1.0)
        self.mx.eval(x)

        enhanced_np = np.array(x, dtype=np.float32)
        enhanced_np = np.nan_to_num(enhanced_np, nan=0.0, posinf=1.0, neginf=0.0)
        enhanced_np = np.clip(enhanced_np, 0.0, 1.0)
        enhanced_np = (enhanced_np * 255.0).round().astype(np.uint8)
        enhanced = Image.fromarray(enhanced_np)

        if self.blend_strength <= 0.0:
            return baseline
        if self.blend_strength >= 1.0:
            return enhanced
        return Image.blend(baseline, enhanced, self.blend_strength)


def show_settings_and_progress_dialog() -> Settings | None:
    """Open a dialog that collects settings in one view, then shows progress in the same dialog."""
    root: tk.Tk | None = None
    result: Settings | None = None

    def fallback_prompt() -> Settings:
        source = input("Tkinter dialog unavailable. Enter source directory path: ").strip()
        destination = input("Enter destination directory path: ").strip()
        if not source or not destination:
            raise SystemExit("No directory selected.")

        ratio_text = input("Enter the multiplier ratio (e.g. 0.5, 1.25, 2): ").strip()
        ratio = parse_ratio_text(ratio_text)

        enhance_text = input("Enable MLX enhancement? [Y/n]: ").strip().lower()
        apply_enhancement = enhance_text in {"", "y", "yes"}

        blend_strength = 0.6
        quant_bits = 0
        if apply_enhancement:
            blend_strength = parse_blend_strength(input("Blend strength [0.0-1.0, default 0.6]: "))
            quant_bits = parse_quant_bits_text(input("Quantization [off/8-bit/4-bit, default off]: "))

        return Settings(
            source=Path(source).expanduser().resolve(),
            destination=Path(destination).expanduser().resolve(),
            ratio=ratio,
            apply_enhancement=apply_enhancement,
            blend_strength=blend_strength,
            quant_bits=quant_bits,
        )

    try:
        root = tk.Tk()
        root.title(f"Resizer3 {RESIZER3_SUBVERSION}")
        root.resizable(False, False)
        root.attributes("-topmost", True)

        source_var = tk.StringVar()
        destination_var = tk.StringVar()
        ratio_var = tk.StringVar(value="1.0")
        enhance_var = tk.BooleanVar(value=True)
        blend_var = tk.StringVar(value="0.6")
        quant_var = tk.StringVar(value="off")

        outer = ttk.Frame(root, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)

        # ===== SETTINGS MODE (initially visible) =====
        settings_frame = ttk.Frame(outer)
        settings_frame.grid(row=0, column=0, sticky="nsew")
        settings_frame.columnconfigure(0, weight=1)

        def browse_for_directory(target: tk.StringVar, title: str) -> None:
            selected = filedialog.askdirectory(
                parent=root,
                title=title,
                mustexist=True,
                initialdir=target.get() or str(Path.home()),
            )
            if selected:
                target.set(selected)

        folders = ttk.LabelFrame(settings_frame, text="Folders", padding=10)
        folders.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        folders.columnconfigure(0, weight=1)

        ttk.Label(folders, text="Source folder").grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Entry(folders, textvariable=source_var, width=54).grid(row=1, column=0, padx=(0, 8), sticky="ew")
        ttk.Button(
            folders,
            text="Browse...",
            command=lambda: browse_for_directory(source_var, "Select source folder containing JPEG images"),
        ).grid(row=1, column=1)

        ttk.Label(folders, text="Destination folder").grid(row=2, column=0, sticky="w", pady=(10, 4))
        ttk.Entry(folders, textvariable=destination_var, width=54).grid(row=3, column=0, padx=(0, 8), sticky="ew")
        ttk.Button(
            folders,
            text="Browse...",
            command=lambda: browse_for_directory(destination_var, "Select destination folder for resized images"),
        ).grid(row=3, column=1)

        resize = ttk.LabelFrame(settings_frame, text="Resize", padding=10)
        resize.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(resize, text="Multiplier ratio (e.g. 0.5, 1.25, 2):").grid(row=0, column=0, sticky="w")
        ttk.Entry(resize, textvariable=ratio_var, width=10).grid(row=0, column=1, padx=(10, 0), sticky="w")

        enhancement = ttk.LabelFrame(settings_frame, text="MLX Enhancement", padding=10)
        enhancement.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        ttk.Checkbutton(
            enhancement,
            text="Enable MLX enhancement",
            variable=enhance_var,
            command=lambda: _toggle_enhancement(),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(enhancement, text="Blend strength (0.0-1.0):").grid(row=1, column=0, sticky="w", pady=(0, 4))
        blend_entry = ttk.Entry(enhancement, textvariable=blend_var, width=10)
        blend_entry.grid(row=1, column=1, sticky="w", pady=(0, 4))

        ttk.Label(enhancement, text="Apple quantization:").grid(row=2, column=0, sticky="w")
        quant_combo = ttk.Combobox(
            enhancement,
            textvariable=quant_var,
            values=("off", "8-bit", "4-bit"),
            state="readonly",
            width=10,
        )
        quant_combo.grid(row=2, column=1, sticky="w")

        ttk.Label(
            settings_frame,
            text=(
                "Note: Processing start may take some time depending on the number of images "
                "to scan and prepare."
            ),
            justify="left",
            foreground="#555555",
            wraplength=520,
        ).grid(row=3, column=0, sticky="w", pady=(0, 10))

        def _toggle_enhancement() -> None:
            enabled = enhance_var.get()
            blend_entry.configure(state="normal" if enabled else "disabled")
            quant_combo.configure(state="readonly" if enabled else "disabled")
            if not enabled:
                quant_var.set("off")

        # ===== PROGRESS MODE (hidden initially) =====
        progress_frame = ttk.Frame(outer)
        progress_frame.columnconfigure(0, weight=1)

        total = 0
        counts: dict[str, int] = {"ok": 0, "skipped": 0, "failed": 0}
        cancel_event = threading.Event()
        progress_state = {"cancelled": False}

        status_var = tk.StringVar(value="Starting...")
        ttk.Label(progress_frame, textvariable=status_var, anchor="w", width=60).grid(
            row=0, column=0, sticky="ew", pady=(0, 8)
        )

        progress_var = tk.IntVar(value=0)
        progress_bar = ttk.Progressbar(
            progress_frame, variable=progress_var, maximum=1, length=500, mode="determinate"
        )
        progress_bar.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        counts_var = tk.StringVar(value="Success: 0   |   Skipped: 0   |   Failed: 0")
        ttk.Label(progress_frame, textvariable=counts_var, anchor="w").grid(
            row=2, column=0, sticky="w", pady=(0, 14)
        )

        button_row_progress = ttk.Frame(progress_frame)
        button_row_progress.grid(row=3, column=0, sticky="e")
        cancel_btn = ttk.Button(button_row_progress, text="Cancel")
        cancel_btn.grid(row=0, column=0, padx=(0, 8))
        close_btn = ttk.Button(button_row_progress, text="Close", command=root.destroy, state="disabled")
        close_btn.grid(row=0, column=1)

        def _request_cancel() -> None:
            if progress_state["cancelled"]:
                return
            progress_state["cancelled"] = True
            cancel_event.set()
            cancel_btn.configure(state="disabled")
            status_var.set("Cancelling... waiting for active work to stop")

        cancel_btn.configure(command=_request_cancel)

        def _update(done: int, filename: str, tag: str) -> None:
            progress_var.set(done)
            status_var.set(f"[{done}/{total}]  {tag}: {filename}")
            counts_var.set(
                f"Success: {counts['ok']}   |   Skipped: {counts['skipped']}   |   Failed: {counts['failed']}"
            )

        def _finish(processed: int) -> None:
            if progress_state["cancelled"]:
                status_var.set(
                    f"Cancelled - processed {processed}/{total}. "
                    f"{counts['ok']} succeeded, {counts['skipped']} skipped, {counts['failed']} failed."
                )
                root.title("Cancelled")
            else:
                status_var.set(
                    f"Done - {counts['ok']} succeeded, {counts['skipped']} skipped, {counts['failed']} failed."
                )
                root.title("Done")
            root.protocol("WM_DELETE_WINDOW", root.destroy)
            cancel_btn.configure(state="disabled")
            close_btn.configure(state="normal")

        def _worker(images: list[Path], process_fn, parallel: bool, max_workers: int) -> None:
            nonlocal total
            total = len(images)
            progress_var.set(0)
            progress_bar.configure(maximum=total if total > 0 else 1)
            processed = 0
            if parallel and max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    future_to_path = {pool.submit(process_fn, p): p for p in images}
                    for future in as_completed(future_to_path):
                        if cancel_event.is_set():
                            for pending in future_to_path:
                                pending.cancel()
                            break
                        image_path = future_to_path[future]
                        try:
                            output_path = future.result()
                            if output_path is None:
                                counts["skipped"] += 1
                                tag = "SKIPPED"
                            else:
                                counts["ok"] += 1
                                tag = "OK"
                        except CancelledError:
                            continue
                        except Exception as err:  # noqa: BLE001
                            counts["failed"] += 1
                            tag = f"FAILED ({err})"
                        processed += 1
                        root.after(0, _update, processed, image_path.name, tag)
            else:
                for image_path in images:
                    if cancel_event.is_set():
                        break
                    try:
                        output_path = process_fn(image_path)
                        if output_path is None:
                            counts["skipped"] += 1
                            tag = "SKIPPED"
                        else:
                            counts["ok"] += 1
                            tag = "OK"
                    except Exception as err:  # noqa: BLE001
                        counts["failed"] += 1
                        tag = f"FAILED ({err})"
                    processed += 1
                    root.after(0, _update, processed, image_path.name, tag)
                root.after(0, _finish, processed)

        def submit() -> None:
            nonlocal result, total
            source_value = source_var.get().strip()
            destination_value = destination_var.get().strip()
            if not source_value or not destination_value:
                messagebox.showerror(
                    "Missing folder",
                    "Select both source and destination folders before continuing.",
                    parent=root,
                )
                return

            try:
                ratio = parse_ratio_text(ratio_var.get())
            except ValueError as err:
                messagebox.showerror("Invalid ratio", str(err), parent=root)
                return

            apply_enhancement = enhance_var.get()
            blend_strength = 0.6
            quant_bits = 0
            if apply_enhancement:
                try:
                    blend_strength = parse_blend_strength(blend_var.get())
                    quant_bits = parse_quant_bits_text(quant_var.get())
                except ValueError as err:
                    messagebox.showerror("Invalid enhancement settings", str(err), parent=root)
                    return

            source_path = Path(source_value).expanduser().resolve()
            destination_path = Path(destination_value).expanduser().resolve()

            if not source_path.exists() or not source_path.is_dir():
                messagebox.showerror("Invalid source", f"'{source_path}' is not a valid directory.", parent=root)
                return
            if not destination_path.exists() or not destination_path.is_dir():
                messagebox.showerror("Invalid destination", f"'{destination_path}' is not a valid directory.", parent=root)
                return

            try:
                validate_destination_directory(source_path, destination_path)
            except SystemExit as err:
                messagebox.showerror("Invalid destination", str(err), parent=root)
                return

            # Scan for images
            images = iter_jpegs(source_path)
            if not images:
                messagebox.showwarning("No images", "No JPEG files found in the source directory.", parent=root)
                return

            result = Settings(
                source=source_path,
                destination=destination_path,
                ratio=ratio,
                apply_enhancement=apply_enhancement,
                blend_strength=blend_strength,
                quant_bits=quant_bits,
            )

            # Switch from settings mode to progress mode
            settings_frame.grid_remove()
            progress_frame.grid(row=0, column=0, sticky="nsew")
            root.title(f"Resizing Images... {RESIZER3_SUBVERSION}")
            root.protocol("WM_DELETE_WINDOW", _request_cancel)

            # Build the process function
            enhancer: MlxEnhancer | None = None
            if result.apply_enhancement:
                try:
                    enhancer = MlxEnhancer(
                        blend_strength=result.blend_strength,
                        quant_bits=result.quant_bits,
                    )
                except RuntimeError as err:
                    status_var.set(f"Error: {err}")
                    close_btn.configure(state="normal")
                    return

            destination_path.mkdir(parents=True, exist_ok=True)

            def _process(image_path: Path) -> Path | None:
                relative_parent = image_path.relative_to(source_path).parent
                target_directory = destination_path / relative_parent
                target_directory.mkdir(parents=True, exist_ok=True)
                return resize_one_jpeg(
                    image_path,
                    result.ratio,
                    result.apply_enhancement,
                    target_directory,
                    enhancer,
                )

            use_parallel = not result.apply_enhancement
            max_workers = min(8, len(images) or 1) if use_parallel else 1

            # Start background processing
            threading.Thread(
                target=_worker,
                args=(images, _process, use_parallel, max_workers),
                daemon=True
            ).start()

        def cancel() -> None:
            root.withdraw()
            root.quit()

        button_row = ttk.Frame(settings_frame)
        button_row.grid(row=4, column=0, sticky="e")
        ttk.Button(button_row, text="Cancel", command=cancel).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_row, text="Start", command=submit).grid(row=0, column=1)

        root.columnconfigure(0, weight=1)
        root.protocol("WM_DELETE_WINDOW", cancel)
        root.bind("<Return>", lambda _event: submit())
        root.bind("<Escape>", lambda _event: cancel())
        _toggle_enhancement()
        root.mainloop()
    except tk.TclError:
        return fallback_prompt()
    finally:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                # Root may already be destroyed by a Close action.
                pass

    return result


def resize_one_jpeg(
    image_path: Path,
    ratio: float,
    apply_enhancement: bool,
    output_directory: Path,
    enhancer: MlxEnhancer | None,
) -> Path | None:
    """Resize one JPEG and optionally run MLX enhancement.

    Returns the output path on success, or None if the file already exists.
    """
    output_name = f"{image_path.stem}_x{ratio_suffix(ratio)}.jpg"
    output_path = output_directory / output_name
    if output_path.exists():
        return None

    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        if apply_enhancement and enhancer is not None:
            resized = enhancer.enhance(img, ratio)
        else:
            new_size = resized_dimensions(img.width, img.height, ratio)
            resized = img.resize(new_size, resample=Image.Resampling.LANCZOS)

        save_kwargs = {
            "format": "JPEG",
            "quality": 95,
            "subsampling": 0,
            "optimize": True,
            "progressive": True,
        }
        if "icc_profile" in img.info:
            save_kwargs["icc_profile"] = img.info["icc_profile"]
        if "exif" in img.info:
            save_kwargs["exif"] = img.info["exif"]

        if resized.mode not in ("RGB", "L"):
            resized = resized.convert("RGB")

        resized.save(output_path, **save_kwargs)
        return output_path





def run() -> None:
    """Show the combined settings and progress dialog."""
    settings = show_settings_and_progress_dialog()
    if settings is None:
        raise SystemExit("Cancelled.")


if __name__ == "__main__":
    run()
