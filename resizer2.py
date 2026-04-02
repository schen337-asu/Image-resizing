"""Batch resize JPEG images and optionally enhance with Real-ESRGAN.

The script prompts for:
1) Source and destination directories selected from one Tkinter dialog
2) A positive multiplier ratio (for example: 0.5, 1.25, 2)
3) Whether to apply Real-ESRGAN enhancement and blend strength

It resizes every .jpg/.jpeg image in the source directory (including child
folders) and saves output files in the selected destination directory while
preserving relative subfolder structure.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
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


def pick_directories() -> tuple[Path, Path]:
	"""Open one Tkinter dialog to choose both source and destination directories."""
	root: tk.Tk | None = None
	selection: dict[str, str] = {}

	def fallback_prompt() -> tuple[Path, Path]:
		source = input("Tkinter dialog unavailable. Enter source directory path: ").strip()
		destination = input("Tkinter dialog unavailable. Enter destination directory path: ").strip()
		if not source or not destination:
			raise SystemExit("No directory selected.")
		return Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()

	try:
		root = tk.Tk()
		root.title("Choose source and destination folders")
		root.resizable(False, False)
		root.attributes("-topmost", True)

		source_var = tk.StringVar()
		destination_var = tk.StringVar()

		frame = ttk.Frame(root, padding=16)
		frame.grid(row=0, column=0, sticky="nsew")

		def browse_for_directory(target: tk.StringVar, dialog_title: str) -> None:
			selected = filedialog.askdirectory(
				parent=root,
				title=dialog_title,
				mustexist=True,
				initialdir=target.get() or str(Path.home()),
			)
			if selected:
				target.set(selected)

		def submit() -> None:
			source_value = source_var.get().strip()
			destination_value = destination_var.get().strip()
			if not source_value or not destination_value:
				messagebox.showerror(
					"Missing folder",
					"Select both source and destination folders before continuing.",
					parent=root,
				)
				return
			selection["source"] = source_value
			selection["destination"] = destination_value
			root.quit()

		def cancel() -> None:
			root.quit()

		ttk.Label(frame, text="Source folder").grid(row=0, column=0, sticky="w", pady=(0, 6))
		ttk.Entry(frame, textvariable=source_var, width=54).grid(row=1, column=0, padx=(0, 8), sticky="ew")
		ttk.Button(
			frame,
			text="Browse...",
			command=lambda: browse_for_directory(source_var, "Select source folder containing JPEG images"),
		).grid(row=1, column=1, sticky="ew")

		ttk.Label(frame, text="Destination folder").grid(row=2, column=0, sticky="w", pady=(14, 6))
		ttk.Entry(frame, textvariable=destination_var, width=54).grid(row=3, column=0, padx=(0, 8), sticky="ew")
		ttk.Button(
			frame,
			text="Browse...",
			command=lambda: browse_for_directory(destination_var, "Select destination folder for resized JPEG images"),
		).grid(row=3, column=1, sticky="ew")

		button_row = ttk.Frame(frame)
		button_row.grid(row=4, column=0, columnspan=2, sticky="e", pady=(16, 0))
		ttk.Button(button_row, text="Cancel", command=cancel).grid(row=0, column=0, padx=(0, 8))
		ttk.Button(button_row, text="Continue", command=submit).grid(row=0, column=1)

		root.columnconfigure(0, weight=1)
		frame.columnconfigure(0, weight=1)
		root.protocol("WM_DELETE_WINDOW", cancel)
		root.bind("<Return>", lambda _event: submit())
		root.bind("<Escape>", lambda _event: cancel())
		root.mainloop()
	except tk.TclError:
		return fallback_prompt()
	finally:
		if root is not None:
			root.destroy()

	if not selection:
		raise SystemExit("No directory selected.")

	return (
		Path(selection["source"]).expanduser().resolve(),
		Path(selection["destination"]).expanduser().resolve(),
	)


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

	def __init__(self, model_url: str = DEFAULT_REALESRGAN_MODEL_URL, tile_size: int = DEFAULT_TILE_SIZE) -> None:
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
		self._model_scale = 4
		device = self._select_device(torch)
		# FP16 is stable on CUDA; MPS FP16 can produce NaN values — keep FP32 on MPS/CPU
		use_half = torch.cuda.is_available()
		self._rrdb_model = RRDBNet(
			num_in_ch=3,
			num_out_ch=3,
			num_feat=64,
			num_block=23,
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
) -> Path:
	"""Resize one JPEG and optionally run Real-ESRGAN enhancement."""
	with Image.open(image_path) as img:
		img = ImageOps.exif_transpose(img)
		if apply_enhancement and enhancer is not None:
			resized = enhancer.enhance(img, ratio, blend_strength)
		else:
			new_size = resized_dimensions(img.width, img.height, ratio)
			resized = img.resize(new_size, resample=Image.Resampling.LANCZOS)

		output_name = f"{image_path.stem}_x{ratio_suffix(ratio)}.jpg"
		output_path = output_directory / output_name

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
	"""Prompt user for inputs and process JPEG files in the selected directory."""
	directory, output_directory = pick_directories()
	ratio_input = input("Enter the multiplier ratio (e.g. 0.5, 1.25, 2): ").strip()
	enhance_input = input("Apply Real-ESRGAN enhancement? [Y/n] (recommended): ").strip()

	if not directory.exists() or not directory.is_dir():
		raise SystemExit(f"Error: '{directory}' is not a valid directory.")
	if not output_directory.exists() or not output_directory.is_dir():
		raise SystemExit(f"Error: '{output_directory}' is not a valid directory.")
	validate_destination_directory(directory, output_directory)

	try:
		ratio = parse_ratio_text(ratio_input)
	except ValueError as err:
		raise SystemExit(f"Error: {err}") from err

	try:
		apply_enhancement = parse_yes_no(enhance_input)
	except ValueError as err:
		raise SystemExit(f"Error: {err}") from err

	blend_strength = 0.6
	if apply_enhancement:
		strength_input = input("Real-ESRGAN blend strength [0.0–1.0, default 0.6]: ").strip()
		try:
			blend_strength = parse_blend_strength(strength_input)
		except ValueError as err:
			raise SystemExit(f"Error: {err}") from err

	images = iter_jpegs(directory)
	if not images:
		raise SystemExit("No JPEG files (.jpg/.jpeg) found in the specified directory.")

	enhancer: RealESRGANEnhancer | None = None
	if apply_enhancement:
		try:
			enhancer = RealESRGANEnhancer()
		except RuntimeError as err:
			raise SystemExit(f"Error: {err}") from err

	output_directory.mkdir(parents=True, exist_ok=True)

	if apply_enhancement:
		mode = f"with Real-ESRGAN enhancement (blend strength {blend_strength})"
	else:
		mode = "without enhancement"
	print(f"Found {len(images)} JPEG file(s). Resizing with ratio {ratio} ({mode})...")
	print(f"Source directory: {directory}")
	print(f"Output directory: {output_directory}")
	successes = 0
	failures = 0

	if apply_enhancement:
		# GPU inference must be sequential; parallelise only disk I/O around it
		for image_path in images:
			try:
				relative_parent = image_path.relative_to(directory).parent
				target_directory = output_directory / relative_parent
				target_directory.mkdir(parents=True, exist_ok=True)
				output_path = resize_one_jpeg(
					image_path, ratio, True, target_directory, enhancer, blend_strength
				)
				successes += 1
				print(f"OK: {image_path} -> {output_path}")
			except Exception as err:  # noqa: BLE001
				failures += 1
				print(f"FAILED: {image_path} ({err})")
	else:
		# No GPU involved: parallelise across CPU cores for pure resize workload
		workers = min(8, (len(images) or 1))

		def _process_one(image_path: Path) -> Path:
			relative_parent = image_path.relative_to(directory).parent
			target_directory = output_directory / relative_parent
			target_directory.mkdir(parents=True, exist_ok=True)
			return resize_one_jpeg(
				image_path, ratio, False, target_directory, None, blend_strength
			)

		with ThreadPoolExecutor(max_workers=workers) as pool:
			future_to_path = {pool.submit(_process_one, p): p for p in images}
			for future in as_completed(future_to_path):
				image_path = future_to_path[future]
				try:
					output_path = future.result()
					successes += 1
					print(f"OK: {image_path} -> {output_path}")
				except Exception as err:  # noqa: BLE001
					failures += 1
					print(f"FAILED: {image_path} ({err})")

	print(f"Done. Success: {successes}, Failed: {failures}")


if __name__ == "__main__":
	run()
