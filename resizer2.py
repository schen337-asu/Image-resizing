"""Batch resize JPEG images and optionally enhance with Real-ESRGAN.

All processing settings are collected from one Tkinter dialog:
- Source and destination directories
- Resize multiplier ratio
- Real-ESRGAN model and blend strength (when enhancement is enabled)

It resizes every .jpg/.jpeg image in the source directory (including child
folders) and saves output files in the selected destination directory while
preserving relative subfolder structure.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import types

import cv2
import numpy as np
from PIL import Image, ImageOps


DEFAULT_REALESRGAN_MODEL_URL = (
	"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
	"RealESRGAN_x4plus.pth"
)

# Default tile size for GPU inference (0 = disabled, 512 recommended for MPS/CUDA)
DEFAULT_TILE_SIZE = 512


@dataclass
class Settings:
	"""All processing parameters collected from the dialog."""

	source: Path
	destination: Path
	ratio: float
	apply_enhancement: bool
	blend_strength: float
	model_url: str
	model_scale: int
	model_num_block: int


REALESRGAN_MODELS: dict[str, tuple[str, int, int]] = {
	"RealESRGAN x4plus — general images": (
		"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
		4,
		23,
	),
	"RealESRGAN x4plus Anime (6B)": (
		"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
		4,
		6,
	),
	"RealESRGAN x2plus — 2\u00d7 upscale": (
		"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
		2,
		23,
	),
}


def parse_ratio_text(value: str) -> float:
	"""Parse and validate a positive resize multiplier ratio."""
	try:
		ratio = float(value.strip())
	except ValueError as exc:
		raise ValueError("Multiplier ratio must be a number, e.g. 0.5 or 1.25") from exc

	if ratio <= 0:
		raise ValueError("Multiplier ratio must be greater than 0")
	return ratio


def ratio_suffix(ratio: float) -> str:
	"""Create a filename-safe, human-readable suffix for the ratio."""
	ratio_str = f"{ratio:.4f}".rstrip("0").rstrip(".")
	return ratio_str.replace(".", "p")


def resized_dimensions(width: int, height: int, ratio: float) -> tuple[int, int]:
	"""Return clamped target dimensions based on the ratio."""
	new_width = max(1, round(width * ratio))
	new_height = max(1, round(height * ratio))
	return new_width, new_height


def parse_yes_no(value: str) -> bool:
	"""Parse yes/no text input into a boolean."""
	answer = value.strip().lower()
	if answer in {"", "y", "yes"}:
		return True
	if answer in {"n", "no"}:
		return False
	raise ValueError("Please answer with y/yes or n/no")


def parse_blend_strength(value: str) -> float:
	"""Parse and validate Real-ESRGAN blend strength in the range 0.0–1.0."""
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


def show_settings_dialog() -> Settings:
	"""Open a single Tkinter dialog to collect all processing settings."""
	root: tk.Tk | None = None
	result: Settings | None = None

	def fallback_prompt() -> Settings:
		source = input("Tkinter dialog unavailable. Enter source directory path: ").strip()
		destination = input("Enter destination directory path: ").strip()
		if not source or not destination:
			raise SystemExit("No directory selected.")
		ratio_text = input("Enter the multiplier ratio (e.g. 0.5, 1.25, 2): ").strip()
		try:
			ratio = parse_ratio_text(ratio_text)
		except ValueError as err:
			raise SystemExit(f"Error: {err}") from err
		enhance_text = input("Apply Real-ESRGAN enhancement? [Y/n]: ").strip()
		try:
			apply_enhancement = parse_yes_no(enhance_text)
		except ValueError as err:
			raise SystemExit(f"Error: {err}") from err
		model_names = list(REALESRGAN_MODELS)
		model_url, model_scale, model_num_block = REALESRGAN_MODELS[model_names[0]]
		blend_strength = 0.6
		if apply_enhancement:
			for i, name in enumerate(model_names):
				print(f"  {i + 1}. {name}")
			choice = input(f"Choose model [1\u2013{len(model_names)}, default 1]: ").strip()
			try:
				idx = (int(choice) - 1) if choice else 0
				if not 0 <= idx < len(model_names):
					raise ValueError
			except ValueError:
				idx = 0
			model_url, model_scale, model_num_block = REALESRGAN_MODELS[model_names[idx]]
			strength_text = input("Blend strength [0.0\u20131.0, default 0.6]: ").strip()
			try:
				blend_strength = parse_blend_strength(strength_text)
			except ValueError as err:
				raise SystemExit(f"Error: {err}") from err
		return Settings(
			source=Path(source).expanduser().resolve(),
			destination=Path(destination).expanduser().resolve(),
			ratio=ratio,
			apply_enhancement=apply_enhancement,
			blend_strength=blend_strength,
			model_url=model_url,
			model_scale=model_scale,
			model_num_block=model_num_block,
		)

	try:
		root = tk.Tk()
		root.title("Image Resizer Settings")
		root.resizable(False, False)
		root.attributes("-topmost", True)

		source_var = tk.StringVar()
		destination_var = tk.StringVar()
		ratio_var = tk.StringVar(value="1.0")
		enhance_var = tk.BooleanVar(value=True)
		model_names = list(REALESRGAN_MODELS)
		model_var = tk.StringVar(value=model_names[0])
		blend_var = tk.DoubleVar(value=0.6)

		outer = ttk.Frame(root, padding=16)
		outer.grid(row=0, column=0, sticky="nsew")
		outer.columnconfigure(0, weight=1)

		def browse_for_directory(target: tk.StringVar, dialog_title: str) -> None:
			selected = filedialog.askdirectory(
				parent=root,
				title=dialog_title,
				mustexist=True,
				initialdir=target.get() or str(Path.home()),
			)
			if selected:
				target.set(selected)

		# ── Folders ──────────────────────────────────────────────────────
		folder_frame = ttk.LabelFrame(outer, text="Folders", padding=10)
		folder_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
		folder_frame.columnconfigure(0, weight=1)

		ttk.Label(folder_frame, text="Source folder").grid(row=0, column=0, sticky="w", pady=(0, 4))
		ttk.Entry(folder_frame, textvariable=source_var, width=54).grid(row=1, column=0, padx=(0, 8), sticky="ew")
		ttk.Button(
			folder_frame,
			text="Browse...",
			command=lambda: browse_for_directory(source_var, "Select source folder containing JPEG images"),
		).grid(row=1, column=1)

		ttk.Label(folder_frame, text="Destination folder").grid(row=2, column=0, sticky="w", pady=(10, 4))
		ttk.Entry(folder_frame, textvariable=destination_var, width=54).grid(row=3, column=0, padx=(0, 8), sticky="ew")
		ttk.Button(
			folder_frame,
			text="Browse...",
			command=lambda: browse_for_directory(destination_var, "Select destination folder for resized images"),
		).grid(row=3, column=1)

		# ── Resize ───────────────────────────────────────────────────────
		resize_frame = ttk.LabelFrame(outer, text="Resize", padding=10)
		resize_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

		ttk.Label(resize_frame, text="Multiplier ratio (e.g. 0.5, 1.25, 2):").grid(
			row=0, column=0, sticky="w", padx=(0, 10)
		)
		ttk.Entry(resize_frame, textvariable=ratio_var, width=10).grid(row=0, column=1, sticky="w")

		# ── Real-ESRGAN Enhancement ───────────────────────────────────────
		enh_frame = ttk.LabelFrame(outer, text="Real-ESRGAN Enhancement", padding=10)
		enh_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
		enh_frame.columnconfigure(1, weight=1)

		# Define _toggle_enhancement before the Checkbutton so the command ref is valid.
		# model_combo and blend_scale are captured by late-binding closure (created below).
		def _toggle_enhancement() -> None:
			if enhance_var.get():
				model_combo.configure(state="readonly")
				blend_scale.configure(state="normal")
				blend_label.configure(foreground="")
			else:
				model_combo.configure(state="disabled")
				blend_scale.configure(state="disabled")
				blend_label.configure(foreground="gray")

		ttk.Checkbutton(
			enh_frame,
			text="Enable Real-ESRGAN enhancement",
			variable=enhance_var,
			command=_toggle_enhancement,
		).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

		ttk.Label(enh_frame, text="Model:").grid(row=1, column=0, sticky="w", pady=(0, 4))
		model_combo = ttk.Combobox(
			enh_frame,
			textvariable=model_var,
			values=model_names,
			state="readonly",
			width=50,
		)
		model_combo.grid(row=1, column=1, sticky="ew", pady=(0, 4))

		ttk.Label(enh_frame, text="Blend strength:").grid(row=2, column=0, sticky="w", pady=(8, 0))
		blend_row = ttk.Frame(enh_frame)
		blend_row.grid(row=2, column=1, sticky="ew", pady=(8, 0))
		blend_scale = tk.Scale(
			blend_row,
			variable=blend_var,
			from_=0.0,
			to=1.0,
			resolution=0.05,
			orient=tk.HORIZONTAL,
			length=280,
			showvalue=False,
		)
		blend_scale.grid(row=0, column=0)
		blend_label = ttk.Label(blend_row, text="0.60", width=5)
		blend_label.grid(row=0, column=1, padx=(8, 0))

		def _update_blend_label(*_: object) -> None:
			blend_label.configure(text=f"{blend_var.get():.2f}")

		blend_var.trace_add("write", _update_blend_label)

		# ── Buttons ───────────────────────────────────────────────────────
		def submit() -> None:
			nonlocal result
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
			model_url, model_scale, model_num_block = REALESRGAN_MODELS[model_var.get()]
			result = Settings(
				source=Path(source_value).expanduser().resolve(),
				destination=Path(destination_value).expanduser().resolve(),
				ratio=ratio,
				apply_enhancement=enhance_var.get(),
				blend_strength=round(blend_var.get(), 2),
				model_url=model_url,
				model_scale=model_scale,
				model_num_block=model_num_block,
			)
			root.withdraw()
			root.quit()

		def cancel() -> None:
			root.withdraw()
			root.quit()

		button_row = ttk.Frame(outer)
		button_row.grid(row=3, column=0, sticky="e")
		ttk.Button(button_row, text="Cancel", command=cancel).grid(row=0, column=0, padx=(0, 8))
		ttk.Button(button_row, text="Continue", command=submit).grid(row=0, column=1)

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
			root.destroy()

	if result is None:
		raise SystemExit("Cancelled.")

	return result


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


class RealESRGANEnhancer:
	"""Real-ESRGAN enhancement wrapper with optional blending."""

	def _ensure_torchvision_compat(self) -> None:
		"""Provide compatibility for basicsr expecting torchvision functional_tensor."""
		try:
			from torchvision.transforms import functional as tv_functional
		except Exception:  # noqa: BLE001
			return

		module_name = "torchvision.transforms.functional_tensor"
		if module_name in sys.modules:
			return

		compat_module = types.ModuleType(module_name)
		if hasattr(tv_functional, "rgb_to_grayscale"):
			compat_module.rgb_to_grayscale = tv_functional.rgb_to_grayscale
		sys.modules[module_name] = compat_module

	@staticmethod
	def _select_device(torch_module):
		"""Return the best available device: CUDA > MPS > CPU."""
		if torch_module.cuda.is_available():
			device = torch_module.device("cuda")
			print("[Real-ESRGAN] Using NVIDIA GPU (CUDA)")
		elif torch_module.backends.mps.is_available():
			device = torch_module.device("mps")
			print("[Real-ESRGAN] Using Apple GPU (MPS)")
		else:
			device = torch_module.device("cpu")
			print("[Real-ESRGAN] Using CPU (no GPU detected)")
		return device

	def __init__(
		self,
		model_url: str = DEFAULT_REALESRGAN_MODEL_URL,
		tile_size: int = DEFAULT_TILE_SIZE,
		model_scale: int = 4,
		num_block: int = 23,
	) -> None:
		self._ensure_torchvision_compat()
		try:
			import torch
			from basicsr.archs.rrdbnet_arch import RRDBNet
			from realesrgan import RealESRGANer
		except ImportError as exc:
			message = (
				"Missing or incompatible dependencies for Real-ESRGAN enhancement. "
				f"Underlying import error: {exc}. "
				f"Active interpreter: {sys.executable}. "
				"Install/repair with: pip install -r requirements.txt. "
				"If the error mentions 'torchvision.transforms.functional_tensor', "
				"run this script directly and keep torchvision pinned to the version "
				"range in requirements.txt."
			)
			raise RuntimeError(message) from exc

		self._torch = torch
		self._model_scale = model_scale
		device = self._select_device(torch)
		# FP16 is stable on CUDA; MPS FP16 can produce NaN values — keep FP32 on MPS/CPU
		use_half = torch.cuda.is_available()
		self._rrdb_model = RRDBNet(
			num_in_ch=3,
			num_out_ch=3,
			num_feat=64,
			num_block=num_block,
			num_grow_ch=32,
			scale=self._model_scale,
		)
		self._upsampler = RealESRGANer(
			scale=self._model_scale,
			model_path=model_url,
			model=self._rrdb_model,
			tile=tile_size,
			tile_pad=10,
			pre_pad=0,
			half=use_half,
			device=device,
		)

	def enhance(self, image: Image.Image, ratio: float, blend_strength: float) -> Image.Image:
		"""Run Real-ESRGAN from original image and blend with baseline resize."""
		rgb = image.convert("RGB")
		target_size = resized_dimensions(rgb.width, rgb.height, ratio)
		baseline = rgb.resize(target_size, resample=Image.Resampling.LANCZOS)

		rgb_np = np.array(rgb)
		bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

		enhanced_bgr, _ = self._upsampler.enhance(bgr_np, outscale=ratio)
		enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
		enhanced = Image.fromarray(enhanced_rgb)

		if enhanced.size != baseline.size:
			enhanced = enhanced.resize(baseline.size, resample=Image.Resampling.LANCZOS)

		if blend_strength <= 0.0:
			return baseline
		if blend_strength >= 1.0:
			return enhanced
		return Image.blend(baseline, enhanced, blend_strength)



def resize_one_jpeg(
	image_path: Path,
	ratio: float,
	apply_enhancement: bool,
	output_directory: Path,
	enhancer: RealESRGANEnhancer | None,
	blend_strength: float,
) -> Path | None:
	"""Resize one JPEG and optionally run Real-ESRGAN enhancement.

	Returns the output path on success, or None if the file already exists.
	"""
	output_name = f"{image_path.stem}_x{ratio_suffix(ratio)}.jpg"
	output_path = output_directory / output_name
	if output_path.exists():
		return None

	with Image.open(image_path) as img:
		img = ImageOps.exif_transpose(img)
		if apply_enhancement and enhancer is not None:
			resized = enhancer.enhance(img, ratio, blend_strength)
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


def show_progress_dialog(
	images: list[Path],
	process_fn,
	*,
	parallel: bool = False,
	max_workers: int = 1,
) -> None:
	"""Show a Tkinter progress window while images are processed in a background thread."""
	total = len(images)
	counts: dict[str, int] = {"ok": 0, "skipped": 0, "failed": 0}

	try:
		root = tk.Tk()
		root.title("Resizing Images\u2026")
		root.resizable(False, False)
		root.attributes("-topmost", True)

		outer = ttk.Frame(root, padding=16)
		outer.grid(row=0, column=0, sticky="nsew")
		outer.columnconfigure(0, weight=1)

		status_var = tk.StringVar(value="Starting\u2026")
		ttk.Label(outer, textvariable=status_var, anchor="w", width=60).grid(
			row=0, column=0, sticky="ew", pady=(0, 8)
		)

		progress_var = tk.IntVar(value=0)
		ttk.Progressbar(
			outer, variable=progress_var, maximum=total, length=500, mode="determinate"
		).grid(row=1, column=0, sticky="ew", pady=(0, 10))

		counts_var = tk.StringVar(value="Success: 0   |   Skipped: 0   |   Failed: 0")
		ttk.Label(outer, textvariable=counts_var, anchor="w").grid(
			row=2, column=0, sticky="w", pady=(0, 14)
		)

		close_btn = ttk.Button(outer, text="Close", command=root.destroy, state="disabled")
		close_btn.grid(row=3, column=0, sticky="e")

		root.protocol("WM_DELETE_WINDOW", lambda: None)

		def _update(done: int, filename: str, tag: str) -> None:
			progress_var.set(done)
			status_var.set(f"[{done}/{total}]  {tag}: {filename}")
			counts_var.set(
				f"Success: {counts['ok']}   |   Skipped: {counts['skipped']}   |   Failed: {counts['failed']}"
			)

		def _finish() -> None:
			status_var.set(
				f"Done \u2014 {counts['ok']} succeeded, {counts['skipped']} skipped, {counts['failed']} failed."
			)
			root.title("Done")
			root.protocol("WM_DELETE_WINDOW", root.destroy)
			close_btn.configure(state="normal")

		def _worker() -> None:
			if parallel and max_workers > 1:
				with ThreadPoolExecutor(max_workers=max_workers) as pool:
					future_to_path = {pool.submit(process_fn, p): p for p in images}
					for i, future in enumerate(as_completed(future_to_path), 1):
						image_path = future_to_path[future]
						try:
							output_path = future.result()
							if output_path is None:
								counts["skipped"] += 1
								tag = "SKIPPED"
							else:
								counts["ok"] += 1
								tag = "OK"
						except Exception as err:  # noqa: BLE001
							counts["failed"] += 1
							tag = f"FAILED ({err})"
						root.after(0, _update, i, image_path.name, tag)
			else:
				for i, image_path in enumerate(images, 1):
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
					root.after(0, _update, i, image_path.name, tag)
			root.after(0, _finish)
		threading.Thread(target=_worker, daemon=True).start()
		root.mainloop()
	except tk.TclError:
		for i, image_path in enumerate(images, 1):
			try:
				output_path = process_fn(image_path)
				if output_path is None:
					counts["skipped"] += 1
					status = "SKIPPED"
				else:
					counts["ok"] += 1
					status = "OK"
			except Exception as err:  # noqa: BLE001
				counts["failed"] += 1
				status = f"FAILED ({err})"
			print(f"[{i}/{total}] {status}: {image_path.name}")
		print(f"\nDone \u2014 {counts['ok']} succeeded, {counts['skipped']} skipped, {counts['failed']} failed.")


def run() -> None:
	"""Collect settings from the dialog and process JPEG files."""
	settings = show_settings_dialog()
	directory = settings.source
	output_directory = settings.destination
	ratio = settings.ratio
	apply_enhancement = settings.apply_enhancement
	blend_strength = settings.blend_strength

	if not directory.exists() or not directory.is_dir():
		raise SystemExit(f"Error: '{directory}' is not a valid directory.")
	if not output_directory.exists() or not output_directory.is_dir():
		raise SystemExit(f"Error: '{output_directory}' is not a valid directory.")
	validate_destination_directory(directory, output_directory)

	images = iter_jpegs(directory)
	if not images:
		raise SystemExit("No JPEG files (.jpg/.jpeg) found in the specified directory.")

	enhancer: RealESRGANEnhancer | None = None
	if apply_enhancement:
		try:
			enhancer = RealESRGANEnhancer(
				model_url=settings.model_url,
				model_scale=settings.model_scale,
				num_block=settings.model_num_block,
			)
		except RuntimeError as err:
			raise SystemExit(f"Error: {err}") from err

	output_directory.mkdir(parents=True, exist_ok=True)

	use_parallel = not apply_enhancement
	max_workers = min(8, len(images) or 1) if use_parallel else 1

	def _process(image_path: Path) -> Path | None:
		relative_parent = image_path.relative_to(directory).parent
		target_directory = output_directory / relative_parent
		target_directory.mkdir(parents=True, exist_ok=True)
		return resize_one_jpeg(
			image_path, ratio, apply_enhancement, target_directory, enhancer, blend_strength
		)

	show_progress_dialog(images, _process, parallel=use_parallel, max_workers=max_workers)


if __name__ == "__main__":
	run()
