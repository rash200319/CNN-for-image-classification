"""Evaluate a trained checkpoint on the EuroSAT test set.

Run from the project root, e.g.:

    python -m src.evaluate --model a --checkpoint outputs/modela_adam/best_model.pt

Writes:
    outputs/<run_name>/confusion_matrix.png
    outputs/<run_name>/metrics.json   -- accuracy, precision, recall (macro),
                                          per-class precision/recall, params,
                                          checkpoint size on disk (KB)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.config import IMAGE_SIZE
from src.data.dataset import get_dataloaders
from src.models.model_a import ModelA
from src.models.model_b import ModelB
from src.models.pretrained import IMAGENET_MEAN, IMAGENET_STD, PRETRAINED_MODELS, build_pretrained
from src.utils import count_parameters


def build_model(name: str, num_classes: int) -> torch.nn.Module:
    if name == "a":
        return ModelA(num_classes=num_classes)
    if name == "b":
        return ModelB(num_classes=num_classes)
    if name in PRETRAINED_MODELS:
        return build_pretrained(name, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def normalization_for(name: str):
    if name in PRETRAINED_MODELS:
        return IMAGENET_MEAN, IMAGENET_STD
    return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def plot_confusion_matrix(cm, classes, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["a", "b", *PRETRAINED_MODELS], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent if args.run_name is None else \
        checkpoint_path.parent.parent / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = normalization_for(args.model)
    _, _, test_loader, classes = get_dataloaders(image_size=args.img_size, mean=mean, std=std)

    model = build_model(args.model, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    preds, labels = collect_predictions(model, test_loader, device)

    accuracy = (preds == labels).mean()
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    plot_confusion_matrix(cm, classes, run_dir / "confusion_matrix.png")

    metrics = {
        "test_accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "per_class": {
            classes[i]: {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
                "support": int(support[i]),
            }
            for i in range(len(classes))
        },
        "trainable_params": count_parameters(model),
        "checkpoint_size_kb": checkpoint_path.stat().st_size / 1024,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"test_accuracy={accuracy:.4f} precision_macro={precision_macro:.4f} "
          f"recall_macro={recall_macro:.4f}")
    print(f"params={metrics['trainable_params']:,} "
          f"checkpoint_size_kb={metrics['checkpoint_size_kb']:.1f}")
    print(f"Confusion matrix saved to {run_dir / 'confusion_matrix.png'}")
    print(f"Full metrics saved to {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()