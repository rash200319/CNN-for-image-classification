"""Training script for Model A / Model B on EuroSAT.

Run from the project root, e.g.:

    python -m src.train --model a --optimizer adam --lr 1e-3 --epochs 20
    python -m src.train --model b --optimizer sgd --lr 1e-2 --epochs 20
    python -m src.train --model b --optimizer sgd_momentum --lr 1e-2 --momentum 0.9 --epochs 20

Each run writes:
    outputs/<run_name>/history.csv   -- per-epoch train/val loss + accuracy + time
    outputs/<run_name>/best_model.pt -- checkpoint with the best val accuracy
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import IMAGE_SIZE, SEED, validate_image_size
from src.data.dataset import get_dataloaders
from src.models.model_a import ModelA
from src.models.model_b import ModelB
from src.models.pretrained import IMAGENET_MEAN, IMAGENET_STD, PRETRAINED_MODELS, build_pretrained
from src.utils import count_parameters

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def build_model(name: str, num_classes: int) -> nn.Module:
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


def build_optimizer(name: str, params, lr: float, momentum: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {name}")


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if optimizer is given, else evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["a", "b", *PRETRAINED_MODELS], required=True)
    parser.add_argument("--optimizer", choices=["adam", "sgd", "sgd_momentum"], required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    validate_image_size(args.img_size)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"model{args.model}_{args.optimizer}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "optimizer": args.optimizer,
                "lr": args.lr,
                "momentum": args.momentum,
                "epochs": args.epochs,
                "img_size": args.img_size,
                "seed": SEED,
                "device": str(device),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
            },
            f,
            indent=2,
        )

    mean, std = normalization_for(args.model)
    train_loader, val_loader, _, classes = get_dataloaders(
        image_size=args.img_size, mean=mean, std=std
    )
    model = build_model(args.model, num_classes=len(classes)).to(device)
    print(f"{run_name}: {count_parameters(model):,} trainable params on {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.momentum)

    history = []
    best_val_acc = 0.0
    for epoch in tqdm(range(1, args.epochs + 1), desc=run_name):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        epoch_time = time.time() - start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time,
            }
        )
        print(
            f"[{run_name}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({epoch_time:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"Done. Best val_acc={best_val_acc:.4f}. Logs in {run_dir}")


if __name__ == "__main__":
    main()