from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "archive" / "EuroSAT"

IMAGE_SIZE = 64
MAX_IMAGE_SIZE = 64
NUM_CLASSES = 10
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

BATCH_SIZE = 64
SEED = 42

MODEL_B_MAX_PARAMS = 100_000


def validate_image_size(image_size: int) -> int:
    """Validate the assignment's maximum input resolution."""
    if image_size <= 0 or image_size > MAX_IMAGE_SIZE:
        raise ValueError(
            f"image_size must be between 1 and {MAX_IMAGE_SIZE}, got {image_size}"
        )
    return image_size