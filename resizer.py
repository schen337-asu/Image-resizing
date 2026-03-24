"""Batch resize JPEG images with high quality using Pillow LANCZOS.

The script prompts for:
1) A directory selected from a folder picker UI
2) A positive multiplier ratio (for example: 0.5, 1.25, 2)

It resizes every .jpg/.jpeg image in that directory (including child folders)
and saves output files in the alternate sibling directory named
<original folder name>+<multiplier>, preserving relative subfolder structure.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps


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


def ratio_label(ratio: float) -> str:
	"""Create a readable multiplier label for directory naming."""
	return f"{ratio:.4f}".rstrip("0").rstrip(".")


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


def apply_s_curve(image: Image.Image, strength: float = 0.18) -> Image.Image:
	"""Apply a gentle contrast S-curve using a LUT for natural tone shaping."""
	x = np.linspace(0.0, 1.0, 256)
	slope = 3.0 + (strength * 8.0)
	y = 1.0 / (1.0 + np.exp(-slope * (x - 0.5)))
	y = (y - y.min()) / (y.max() - y.min())
	lut = np.clip(y * 255.0, 0, 255).astype(np.uint8).tolist()

	if image.mode == "RGB":
		return image.point(lut * 3)
	if image.mode == "L":
		return image.point(lut)
	return image.convert("RGB").point(lut * 3)


def enhance_dynamic_range_and_contrast(image: Image.Image) -> Image.Image:
	"""Enhance luminance detail using CLAHE, then add subtle global contrast and clarity."""
	rgb = image.convert("RGB")
	rgb_array = np.array(rgb)
	bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

	lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
	l_channel, a_channel, b_channel = cv2.split(lab)

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	enhanced_l = clahe.apply(l_channel)
	enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
	enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
	enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

	enhanced_image = Image.fromarray(enhanced_rgb)
	enhanced_image = apply_s_curve(enhanced_image, strength=0.16)
	enhanced_image = enhanced_image.filter(
		ImageFilter.UnsharpMask(radius=1.2, percent=90, threshold=3)
	)
	return enhanced_image


def resize_one_jpeg(
	image_path: Path,
	ratio: float,
	apply_enhancement: bool,
	output_directory: Path,
) -> Path:
	"""Resize one JPEG using high-quality LANCZOS and save as JPEG."""
	with Image.open(image_path) as img:
		img = ImageOps.exif_transpose(img)
		new_size = resized_dimensions(img.width, img.height, ratio)
		resized = img.resize(new_size, resample=Image.Resampling.LANCZOS)
		if apply_enhancement:
			resized = enhance_dynamic_range_and_contrast(resized)

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


def iter_jpegs(directory: Path) -> list[Path]:
	"""Collect .jpg/.jpeg files recursively from a directory tree."""
	return sorted(
		path
		for path in directory.rglob("*")
		if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
	)


def pick_directory() -> Path:
	"""Open a folder picker UI and return the selected directory path.

	Uses native macOS chooser via osascript and falls back to manual input.
	"""
	selected = ""
	try:
		result = subprocess.run(
			[
				"osascript",
				"-e",
				'POSIX path of (choose folder with prompt "Select folder containing JPEG images")',
			],
			capture_output=True,
			text=True,
			check=False,
		)
		if result.returncode == 0:
			selected = result.stdout.strip()
	except FileNotFoundError:
		selected = ""

	if not selected:
		selected = input("Folder picker cancelled/unavailable. Enter directory path: ").strip()

	if not selected:
		raise SystemExit("No directory selected.")

	return Path(selected).expanduser().resolve()


def run() -> None:
	"""Prompt user for inputs and resize JPEG files in the selected directory."""
	directory = pick_directory()
	ratio_input = input("Enter the multiplier ratio (e.g. 0.5, 1.25, 2): ").strip()
	enhance_input = input(
		"Enhance dynamic range and contrast? [Y/n] (recommended): "
	).strip()

	if not directory.exists() or not directory.is_dir():
		raise SystemExit(f"Error: '{directory}' is not a valid directory.")

	try:
		ratio = parse_ratio_text(ratio_input)
	except ValueError as err:
		raise SystemExit(f"Error: {err}") from err

	try:
		apply_enhancement = parse_yes_no(enhance_input)
	except ValueError as err:
		raise SystemExit(f"Error: {err}") from err

	images = iter_jpegs(directory)
	if not images:
		raise SystemExit("No JPEG files (.jpg/.jpeg) found in the specified directory.")

	original_folder_name = directory.name or "root"
	output_directory = directory.parent / f"{original_folder_name}+{ratio_label(ratio)}"
	output_directory.mkdir(parents=True, exist_ok=True)

	mode = "with enhancement" if apply_enhancement else "without enhancement"
	print(f"Found {len(images)} JPEG file(s). Resizing with ratio {ratio} ({mode})...")
	print(f"Output directory: {output_directory}")
	successes = 0
	failures = 0

	for image_path in images:
		try:
			relative_parent = image_path.relative_to(directory).parent
			target_directory = output_directory / relative_parent
			target_directory.mkdir(parents=True, exist_ok=True)

			output_path = resize_one_jpeg(
				image_path,
				ratio,
				apply_enhancement,
				target_directory,
			)
			successes += 1
			print(f"OK: {image_path} -> {output_path}")
		except Exception as err:  # noqa: BLE001
			failures += 1
			print(f"FAILED: {image_path} ({err})")

	print(f"Done. Success: {successes}, Failed: {failures}")


if __name__ == "__main__":
	run()
