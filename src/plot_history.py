"""Plot train/val loss and accuracy curves from one or more training runs.

Single run -- plots that run's own train vs val curves:

    python -m src.plot_history --run-dir outputs/final_a_adam

Multiple runs -- overlays val curves from each run for comparison
(e.g. the Section 3 optimizer sweep) and requires --out:

    python -m src.plot_history --run-dir outputs/sweep_a_adam --run-dir outputs/sweep_a_sgd \
        --run-dir outputs/sweep_a_sgdm --out outputs/sweep_a_comparison.png

Reads:
    <run-dir>/history.csv  -- written by src/train.py

Writes:
    <run-dir>/loss_curve.png            (single-run mode, default --out)
    --out path                          (multi-run mode, --out required)
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(run_dir: Path):
    with open(run_dir / "history.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    return {
        "epoch": epochs,
        "train_loss": [float(r["train_loss"]) for r in rows],
        "val_loss": [float(r["val_loss"]) for r in rows],
        "train_acc": [float(r["train_acc"]) for r in rows],
        "val_acc": [float(r["val_acc"]) for r in rows],
    }


def plot_single(history, run_name: str, out_path: Path):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_loss.plot(history["epoch"], history["train_loss"], label="train")
    ax_loss.plot(history["epoch"], history["val_loss"], label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(f"{run_name}: loss")
    ax_loss.legend()

    ax_acc.plot(history["epoch"], history["train_acc"], label="train")
    ax_acc.plot(history["epoch"], history["val_acc"], label="val")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title(f"{run_name}: accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overlay(histories, run_names, out_path: Path):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))

    for history, name in zip(histories, run_names):
        line, = ax_loss.plot(history["epoch"], history["val_loss"], label=name)
        ax_loss.plot(history["epoch"], history["train_loss"], linestyle="--",
                     color=line.get_color(), alpha=0.5)

    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss (solid=val, dashed=train)")
    ax_loss.set_title("loss comparison")
    ax_loss.legend()

    for history, name in zip(histories, run_names):
        line, = ax_acc.plot(history["epoch"], history["val_acc"], label=name)
        ax_acc.plot(history["epoch"], history["train_acc"], linestyle="--",
                    color=line.get_color(), alpha=0.5)

    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy (solid=val, dashed=train)")
    ax_acc.set_title("accuracy comparison")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True,
                         help="Run directory containing history.csv; repeat for multiple runs")
    parser.add_argument("--out", type=str, default=None,
                         help="Output PNG path (required with multiple --run-dir)")
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.run_dir]

    if len(run_dirs) == 1:
        run_dir = run_dirs[0]
        history = read_history(run_dir)
        out_path = Path(args.out) if args.out else run_dir / "loss_curve.png"
        plot_single(history, run_dir.name, out_path)
    else:
        if not args.out:
            parser.error("--out is required when passing multiple --run-dir")
        histories = [read_history(d) for d in run_dirs]
        run_names = [d.name for d in run_dirs]
        out_path = Path(args.out)
        plot_overlay(histories, run_names, out_path)

    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
