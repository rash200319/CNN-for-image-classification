"""Scan a dataset and verify the EN3150 maximum image resolution.

Run the portable dataset audit with:

    python tests/test_dataset_resolution.py --dataset-dir <path-to-EuroSAT>

The path can also be supplied through the ``EUROSAT_DIR`` environment
variable. No machine-specific dataset path is required by this test.
"""

import argparse
import os
import sys
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

MAX_IMAGE_SIZE = 64
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def scan_image_dimensions(dataset_dir: Path) -> tuple[Counter, list[tuple[Path, str]]]:
    """Return dimensions and unreadable files for every image below ``dataset_dir``."""
    dimensions = Counter()
    unreadable = []

    for image_path in sorted(dataset_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(image_path) as image:
                dimensions[image.size] += 1
        except (OSError, ValueError) as error:
            unreadable.append((image_path, str(error)))

    return dimensions, unreadable


def print_report(dataset_dir: Path, dimensions: Counter, unreadable: list[tuple[Path, str]]) -> None:
    """Print a reproducible summary suitable for the assignment evidence."""
    image_count = sum(dimensions.values()) + len(unreadable)
    print(f"Dataset: {dataset_dir}")
    print(f"Total image files: {image_count}")
    print("Resolution counts:")
    for (width, height), count in sorted(dimensions.items()):
        print(f"  {width} x {height}: {count} images")

    if dimensions:
        max_width = max(width for width, _ in dimensions)
        max_height = max(height for _, height in dimensions)
        print(f"Maximum resolution: {max_width} x {max_height}")
    else:
        max_width = max_height = 0

    print(f"Unreadable files: {len(unreadable)}")
    if max_width <= MAX_IMAGE_SIZE and max_height <= MAX_IMAGE_SIZE and not unreadable:
        print("PASS: All images satisfy the maximum 64 x 64 requirement.")
    else:
        print("FAIL: The dataset does not satisfy the maximum 64 x 64 requirement.")


def audit_dataset(dataset_dir: Path) -> None:
    """Scan and assert that every readable image satisfies the assignment rule."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    dimensions, unreadable = scan_image_dimensions(dataset_dir)
    if not dimensions and not unreadable:
        raise ValueError(f"No supported image files found in: {dataset_dir}")

    print_report(dataset_dir, dimensions, unreadable)
    oversized = [
        resolution
        for resolution in dimensions
        if resolution[0] > MAX_IMAGE_SIZE or resolution[1] > MAX_IMAGE_SIZE
    ]
    if oversized or unreadable:
        raise AssertionError(
            f"Found {len(oversized)} oversized resolution(s) and "
            f"{len(unreadable)} unreadable file(s)."
        )


class TestDatasetResolution(unittest.TestCase):
    """Unit tests for the resolution audit without requiring a real dataset."""

    def test_scans_all_supported_images(self):
        with TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            for index, size in enumerate(((64, 64), (32, 48))):
                image = Image.new("RGB", size)
                image.save(dataset_dir / f"image_{index}.jpg")
                image.close()

            dimensions, unreadable = scan_image_dimensions(dataset_dir)

        self.assertEqual(dimensions, Counter({(64, 64): 1, (32, 48): 1}))
        self.assertEqual(unreadable, [])

    def test_rejects_an_image_above_64_by_64(self):
        with TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            image = Image.new("RGB", (65, 64))
            image.save(dataset_dir / "oversized.jpg")
            image.close()

            with self.assertRaises(AssertionError):
                audit_dataset(dataset_dir)


def _parse_dataset_directory() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=os.environ.get("EUROSAT_DIR"),
        help="Dataset directory; alternatively set EUROSAT_DIR.",
    )
    args = parser.parse_args()
    if args.dataset_dir is None:
        parser.error("provide --dataset-dir or set EUROSAT_DIR")
    return args.dataset_dir


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audit_dataset(_parse_dataset_directory())
    else:
        unittest.main()
