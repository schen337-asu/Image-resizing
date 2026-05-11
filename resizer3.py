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
    """Open a dark-themed two-column dialog: settings (left) | progress (right)."""
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

    # ── Colour palette (Catppuccin Mocha) ─────────────────────────────
    BG      = "#1e1e2e"
    SURFACE = "#313244"
    TEXT    = "#cdd6f4"
    MUTED   = "#6c7086"
    ACCENT  = "#89b4fa"
    SUCCESS = "#a6e3a1"
    DANGER  = "#f38ba8"
    BORDER  = "#45475a"

    try:
        root = tk.Tk()
        root.title(f"Resizer3 {RESIZER3_SUBVERSION}")
        root.resizable(False, False)
        root.attributes("-topmost", True)
        root.configure(bg=BG)
        root.minsize(860, 440)

        # ── TTK style ─────────────────────────────────────────────────
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(
            ".",
            background=BG, foreground=TEXT,
            fieldbackground=SURFACE, troughcolor=SURFACE,
            bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
            font=("Helvetica", 12),
        )
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure(
            "TLabelframe",
            background=BG, bordercolor=BORDER,
        )
        style.configure(
            "TLabelframe.Label",
            background=BG, foreground=ACCENT, font=("Helvetica", 12, "bold"),
        )
        style.configure(
            "TEntry",
            fieldbackground=SURFACE, foreground=TEXT,
            insertcolor=TEXT, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
        )
        style.map(
            "TEntry",
            fieldbackground=[("disabled", BG)],
            foreground=[("disabled", MUTED)],
        )
        style.configure(
            "TButton",
            background=SURFACE, foreground=TEXT,
            bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, padding=(10, 5),
        )
        style.map(
            "TButton",
            background=[("active", BORDER), ("disabled", BG)],
            foreground=[("disabled", MUTED)],
        )
        style.configure(
            "Accent.TButton",
            background=ACCENT, foreground=BG,
            bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
            font=("Helvetica", 12, "bold"), padding=(10, 5),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#74c7ec"), ("disabled", BORDER)],
            foreground=[("active", BG), ("disabled", MUTED)],
        )
        style.configure(
            "Danger.TButton",
            background=SURFACE, foreground=DANGER,
            bordercolor=DANGER, lightcolor=DANGER, darkcolor=DANGER, padding=(10, 5),
        )
        style.map(
            "Danger.TButton",
            background=[("active", BORDER), ("disabled", BG)],
            foreground=[("disabled", MUTED)],
        )
        style.configure("TCheckbutton", background=BG, foreground=TEXT)
        style.map(
            "TCheckbutton",
            background=[("active", BG)],
            indicatorcolor=[("selected", ACCENT), ("!selected", SURFACE)],
        )
        style.configure(
            "TCombobox",
            fieldbackground=SURFACE, foreground=TEXT,
            selectbackground=ACCENT, selectforeground=BG,
            bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("disabled", BG)],
            foreground=[("disabled", MUTED)],
        )
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Helvetica", 10))
        style.configure("Header.TLabel", background=BG, foreground=ACCENT, font=("Helvetica", 15, "bold"))
        style.configure("Version.TLabel", background=BG, foreground=MUTED, font=("Helvetica", 10))
        style.configure("SectionTitle.TLabel", background=BG, foreground=TEXT, font=("Helvetica", 12, "bold"))

        # ── Tk vars ───────────────────────────────────────────────────
        source_var = tk.StringVar()
        destination_var = tk.StringVar()
        ratio_var = tk.StringVar(value="1.0")
        enhance_var = tk.BooleanVar(value=True)
        blend_var = tk.StringVar(value="0.6")
        quant_var = tk.StringVar(value="off")

        # ── Outer container: 3 columns (settings | sep | progress) ────
        outer = ttk.Frame(root, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1, minsize=400)
        outer.columnconfigure(2, weight=1, minsize=360)
        outer.rowconfigure(0, weight=1)

        # Vertical separator
        sep = tk.Frame(outer, width=1, bg=BORDER)
        sep.grid(row=0, column=1, sticky="ns", padx=16)

        # ════════════════════════════════════════════════════════════
        # LEFT COLUMN — Settings
        # ════════════════════════════════════════════════════════════
        settings_col = ttk.Frame(outer)
        settings_col.grid(row=0, column=0, sticky="nsew")
        settings_col.columnconfigure(0, weight=1)

        # Header
        header_row = ttk.Frame(settings_col)
        header_row.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(header_row, text="Resizer3", style="Header.TLabel").pack(side="left")
        ttk.Label(header_row, text=f" {RESIZER3_SUBVERSION}", style="Version.TLabel").pack(
            side="left", padx=(6, 0), pady=(5, 0)
        )

        def browse_for_directory(target: tk.StringVar, title: str) -> None:
            selected = filedialog.askdirectory(
                parent=root,
                title=title,
                mustexist=True,
                initialdir=target.get() or str(Path.home()),
            )
            if selected:
                target.set(selected)

        # Folders section
        folders = ttk.LabelFrame(settings_col, text="Folders", padding=10)
        folders.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        folders.columnconfigure(0, weight=1)

        ttk.Label(folders, text="Source folder", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        src_entry = ttk.Entry(folders, textvariable=source_var, width=38)
        src_entry.grid(row=1, column=0, padx=(0, 8), sticky="ew")
        src_browse_btn = ttk.Button(
            folders,
            text="Browse…",
            command=lambda: browse_for_directory(source_var, "Select source folder containing JPEG images"),
        )
        src_browse_btn.grid(row=1, column=1)

        ttk.Label(folders, text="Destination folder", style="Muted.TLabel").grid(
            row=2, column=0, sticky="w", pady=(10, 4)
        )
        dst_entry = ttk.Entry(folders, textvariable=destination_var, width=38)
        dst_entry.grid(row=3, column=0, padx=(0, 8), sticky="ew")
        dst_browse_btn = ttk.Button(
            folders,
            text="Browse…",
            command=lambda: browse_for_directory(destination_var, "Select destination folder for resized images"),
        )
        dst_browse_btn.grid(row=3, column=1)

        # Resize section
        resize_frame = ttk.LabelFrame(settings_col, text="Resize", padding=10)
        resize_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(resize_frame, text="Multiplier ratio (e.g. 0.5, 1.25, 2):").grid(
            row=0, column=0, sticky="w"
        )
        ratio_entry = ttk.Entry(resize_frame, textvariable=ratio_var, width=10)
        ratio_entry.grid(row=0, column=1, padx=(10, 0), sticky="w")

        # Enhancement section
        enhancement = ttk.LabelFrame(settings_col, text="MLX Enhancement", padding=10)
        enhancement.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        enhance_check = ttk.Checkbutton(
            enhancement,
            text="Enable MLX enhancement",
            variable=enhance_var,
            command=lambda: _toggle_enhancement(),
        )
        enhance_check.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(enhancement, text="Blend strength (0.0–1.0):").grid(
            row=1, column=0, sticky="w", pady=(0, 4)
        )
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
            settings_col,
            text="Processing start may take a moment while images are scanned.",
            style="Muted.TLabel",
            wraplength=380,
        ).grid(row=4, column=0, sticky="w", pady=(0, 14))

        def _toggle_enhancement() -> None:
            enabled = enhance_var.get()
            blend_entry.configure(state="normal" if enabled else "disabled")
            quant_combo.configure(state="readonly" if enabled else "disabled")
            if not enabled:
                quant_var.set("off")

        # Collect all settings inputs for bulk enable/disable when processing starts
        _settings_inputs = [
            src_entry, src_browse_btn,
            dst_entry, dst_browse_btn,
            ratio_entry, enhance_check,
            blend_entry, quant_combo,
        ]

        settings_btn_row = ttk.Frame(settings_col)
        settings_btn_row.grid(row=5, column=0, sticky="e")
        cancel_settings_btn = ttk.Button(settings_btn_row, text="Cancel", style="Danger.TButton")
        cancel_settings_btn.grid(row=0, column=0, padx=(0, 8))
        start_btn = ttk.Button(settings_btn_row, text="Start", style="Accent.TButton")
        start_btn.grid(row=0, column=1)

        # ════════════════════════════════════════════════════════════
        # RIGHT COLUMN — Progress
        # ════════════════════════════════════════════════════════════
        progress_col = ttk.Frame(outer)
        progress_col.grid(row=0, column=2, sticky="nsew")
        progress_col.columnconfigure(0, weight=1)

        ttk.Label(progress_col, text="Processing", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 14)
        )

        # Canvas-based progress bar with centred % overlay
        class CanvasProgressBar:
            """tk.Canvas progress bar that renders a centred percentage label."""

            BAR_H  = 28
            RADIUS = 5

            def __init__(self, parent: tk.Widget) -> None:
                self.canvas = tk.Canvas(
                    parent, height=self.BAR_H, bg=BG,
                    highlightthickness=0, bd=0,
                )
                self._value   = 0
                self._maximum = 1

            def grid(self, **kwargs) -> None:
                self.canvas.grid(**kwargs)
                self.canvas.bind("<Configure>", lambda _e: self._draw())

            def update(self, value: int, maximum: int) -> None:
                self._value   = value
                self._maximum = max(1, maximum)
                self._draw()

            def _draw(self) -> None:
                c = self.canvas
                w = c.winfo_width()
                h = self.BAR_H
                if w <= 1:
                    c.after(50, self._draw)
                    return
                c.delete("all")
                # Background track
                c.create_rectangle(0, 0, w, h, fill=SURFACE, outline=BORDER, width=1)
                # Filled portion
                pct = self._value / self._maximum
                fill_w = max(0, int((w - 2) * pct))
                if fill_w > 0:
                    c.create_rectangle(1, 1, fill_w + 1, h - 1, fill=ACCENT, outline="")
                # Percentage label overlaid in the centre
                c.create_text(
                    w // 2, h // 2,
                    text=f"{pct * 100:.0f}%",
                    fill=TEXT,
                    font=("Helvetica", 10, "bold"),
                    anchor="center",
                )

        canvas_bar = CanvasProgressBar(progress_col)
        canvas_bar.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        status_var = tk.StringVar(value="Waiting to start…")
        status_label = ttk.Label(
            progress_col, textvariable=status_var, anchor="w", wraplength=340,
        )
        status_label.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        counts_var = tk.StringVar(value="—")
        ttk.Label(
            progress_col, textvariable=counts_var, anchor="w", style="Muted.TLabel",
        ).grid(row=3, column=0, sticky="w", pady=(0, 14))

        progress_btn_row = ttk.Frame(progress_col)
        progress_btn_row.grid(row=4, column=0, sticky="e")
        cancel_btn = ttk.Button(
            progress_btn_row, text="Cancel", style="Danger.TButton", state="disabled"
        )
        cancel_btn.grid(row=0, column=0, padx=(0, 8))
        close_btn = ttk.Button(
            progress_btn_row, text="Close", command=root.destroy, state="disabled"
        )
        close_btn.grid(row=0, column=1)

        # ── Shared processing state ────────────────────────────────────
        total = 0
        counts: dict[str, int] = {"ok": 0, "skipped": 0, "failed": 0}
        cancel_event  = threading.Event()
        progress_state = {"cancelled": False}

        def _request_cancel() -> None:
            if progress_state["cancelled"]:
                return
            progress_state["cancelled"] = True
            cancel_event.set()
            cancel_btn.configure(state="disabled")
            status_label.configure(foreground=DANGER)
            status_var.set("Cancelling… waiting for active work to stop")

        cancel_btn.configure(command=_request_cancel)

        def _update(done: int, filename: str, tag: str) -> None:
            canvas_bar.update(done, total)
            status_var.set(f"[{done}/{total}]  {tag}: {filename}")
            counts_var.set(
                f"✓ {counts['ok']}   ·   ↷ {counts['skipped']}   ·   ✗ {counts['failed']}"
            )

        def _finish(processed: int) -> None:
            if progress_state["cancelled"]:
                status_label.configure(foreground=DANGER)
                status_var.set(
                    f"Cancelled — {processed}/{total} processed. "
                    f"{counts['ok']} succeeded, {counts['skipped']} skipped, {counts['failed']} failed."
                )
                root.title("Cancelled — Resizer3")
            else:
                canvas_bar.update(total, total)
                status_label.configure(foreground=SUCCESS)
                status_var.set(
                    f"Done — {counts['ok']} succeeded, "
                    f"{counts['skipped']} skipped, {counts['failed']} failed."
                )
                root.title("Done — Resizer3")
            root.protocol("WM_DELETE_WINDOW", root.destroy)
            cancel_btn.configure(state="disabled")
            close_btn.configure(state="normal")

        def _worker(images: list[Path], process_fn, parallel: bool, max_workers: int) -> None:
            nonlocal total
            total = len(images)
            root.after(0, canvas_bar.update, 0, total)
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
            source_value      = source_var.get().strip()
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
            blend_strength    = 0.6
            quant_bits        = 0
            if apply_enhancement:
                try:
                    blend_strength = parse_blend_strength(blend_var.get())
                    quant_bits     = parse_quant_bits_text(quant_var.get())
                except ValueError as err:
                    messagebox.showerror("Invalid enhancement settings", str(err), parent=root)
                    return

            source_path      = Path(source_value).expanduser().resolve()
            destination_path = Path(destination_value).expanduser().resolve()

            if not source_path.exists() or not source_path.is_dir():
                messagebox.showerror(
                    "Invalid source", f"'{source_path}' is not a valid directory.", parent=root
                )
                return
            if not destination_path.exists() or not destination_path.is_dir():
                messagebox.showerror(
                    "Invalid destination", f"'{destination_path}' is not a valid directory.", parent=root
                )
                return

            try:
                validate_destination_directory(source_path, destination_path)
            except SystemExit as err:
                messagebox.showerror("Invalid destination", str(err), parent=root)
                return

            images = iter_jpegs(source_path)
            if not images:
                messagebox.showwarning(
                    "No images", "No JPEG files found in the source directory.", parent=root
                )
                return

            result = Settings(
                source=source_path,
                destination=destination_path,
                ratio=ratio,
                apply_enhancement=apply_enhancement,
                blend_strength=blend_strength,
                quant_bits=quant_bits,
            )

            # Disable settings inputs so user can see what was used
            for widget in _settings_inputs:
                widget.configure(state="disabled")
            start_btn.configure(state="disabled")
            cancel_settings_btn.configure(state="disabled")

            # Activate progress column
            status_label.configure(foreground=TEXT)
            status_var.set(f"Starting — {len(images)} image(s) found…")
            counts_var.set("✓ 0   ·   ↷ 0   ·   ✗ 0")
            cancel_btn.configure(state="normal")
            root.title(f"Resizing… — Resizer3 {RESIZER3_SUBVERSION}")
            root.protocol("WM_DELETE_WINDOW", _request_cancel)

            # Build enhancer (if requested)
            enhancer: MlxEnhancer | None = None
            if result.apply_enhancement:
                try:
                    enhancer = MlxEnhancer(
                        blend_strength=result.blend_strength,
                        quant_bits=result.quant_bits,
                    )
                except RuntimeError as err:
                    status_label.configure(foreground=DANGER)
                    status_var.set(f"Error: {err}")
                    close_btn.configure(state="normal")
                    return

            destination_path.mkdir(parents=True, exist_ok=True)

            def _process(image_path: Path) -> Path | None:
                relative_parent  = image_path.relative_to(source_path).parent
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
            max_workers  = min(8, len(images) or 1) if use_parallel else 1

            threading.Thread(
                target=_worker,
                args=(images, _process, use_parallel, max_workers),
                daemon=True,
            ).start()

        def cancel() -> None:
            root.withdraw()
            root.quit()

        cancel_settings_btn.configure(command=cancel)
        start_btn.configure(command=submit)

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
