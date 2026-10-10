"""Build a comparison table across training runs from their saved outputs.

    python -m src.compare_runs --run-dir outputs/final_a_adam --run-dir outputs/modelb_adam \
        --out outputs/comparison_table.csv

Reads, per run directory:
    metrics.json  -- trainable_params, test_accuracy, checkpoint_size_kb (written by src/evaluate.py)
    history.csv   -- epoch_time_sec per epoch (written by src/train.py)

Time per epoch is reported as the median across epochs rather than the mean,
since a handful of epochs in these runs were several times slower than the
rest (background system load / sleep interruptions on a shared CPU-only
machine), and the median is robust to that kind of one-off outlier.

Writes:
    --out path (CSV) and prints the same table as Markdown to stdout.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


def summarize_run(run_dir: Path):
    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)
    with open(run_dir / "history.csv", newline="") as f:
        epoch_times = [float(r["epoch_time_sec"]) for r in csv.DictReader(f)]
    return {
        "run": run_dir.name,
        "params": metrics["trainable_params"],
        "checkpoint_kb": round(metrics["checkpoint_size_kb"], 1),
        "epochs": len(epoch_times),
        "median_epoch_time_sec": round(statistics.median(epoch_times), 1),
        "test_accuracy": round(metrics["test_accuracy"], 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    rows = [summarize_run(Path(p)) for p in args.run_dir]

    fieldnames = ["run", "params", "checkpoint_kb", "epochs", "median_epoch_time_sec", "test_accuracy"]
    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    header = "| " + " | ".join(fieldnames) + " |"
    sep = "|" + "|".join(["---"] * len(fieldnames)) + "|"
    print(header)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(row[f]) for f in fieldnames) + " |")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
