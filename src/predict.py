"""Single-image and batch inference CLI tool for trained EuroSAT classifiers.

Usage:
    # Single image inference with Model B
    python -m src.predict --image path/to/image.jpg --checkpoint outputs/modelb_adam/best_model.pt --model b

    # Single image inference with Pretrained MobileNetV2
    python -m src.predict --image path/to/image.jpg --checkpoint outputs/final_mobilenet_v2/best_model.pt --model mobilenet_v2 --img-size 64
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.config import CLASS_NAMES, IMAGE_SIZE, validate_image_size
from src.models.model_a import ModelA
from src.models.model_b import ModelB
from src.models.pretrained import IMAGENET_MEAN, IMAGENET_STD, PRETRAINED_MODELS, build_pretrained


def build_model(name: str, num_classes: int) -> torch.nn.Module:
    """Build specified model architecture."""
    if name == "a":
        return ModelA(num_classes=num_classes)
    if name == "b":
        return ModelB(num_classes=num_classes)
    if name in PRETRAINED_MODELS:
        return build_pretrained(name, num_classes=num_classes)
    raise ValueError(f"Unknown model name: {name}")


def get_inference_transform(model_name: str, img_size: int) -> transforms.Compose:
    """Construct data preprocessing pipeline consistent with training."""
    validate_image_size(img_size)
    if model_name in PRETRAINED_MODELS:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    device: torch.device,
    top_k: int = 3,
) -> List[Dict[str, float]]:
    """Load an image, execute forward inference, and return top-k predictions."""
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(CLASS_NAMES)))

    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        results.append(
            {
                "class_index": idx,
                "class_name": CLASS_NAMES[idx],
                "confidence": prob,
                "confidence_percent": round(prob * 100.0, 2),
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Classify a land-use satellite image using a trained model.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model weights (.pt).")
    parser.add_argument("--model", choices=["a", "b", *PRETRAINED_MODELS], required=True, help="Model architecture.")
    parser.add_argument("--img-size", type=int, default=None, help="Input spatial dimension (default: auto).")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to display (default: 3).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.img_size is None:
        img_size = IMAGE_SIZE
    else:
        img_size = args.img_size
    validate_image_size(img_size)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model(args.model, num_classes=len(CLASS_NAMES)).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    transform = get_inference_transform(args.model, img_size)
    predictions = predict_image(model, image_path, transform, device, top_k=args.top_k)

    print("\n" + "=" * 50)
    print(f"  EuroSAT Land-Use Classification Result")
    print(f"  Image: {image_path.name}")
    print(f"  Model: {args.model.upper()} | Device: {device.type.upper()}")
    print("=" * 50)

    top_pred = predictions[0]
    print(f"\n>> PREDICTED CLASS: [{top_pred['class_name']}] ({top_pred['confidence_percent']}%) <<\n")

    print("Top Predictions Breakdown:")
    for rank, p in enumerate(predictions, 1):
        bar_len = int(p["confidence"] * 25)
        bar = "#" * bar_len + "-" * (25 - bar_len)
        print(f"  {rank}. {p['class_name']:<22} {p['confidence_percent']:>6.2f}%  [{bar}]")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
