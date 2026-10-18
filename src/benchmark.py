"""Synthetic-input architecture benchmarking tool.

Computes theoretical MACs/FLOPs, inference latency, and throughput across
Model A, Model B, MobileNetV2, and EfficientNet-B0.

The latency and throughput values use newly initialized models and synthetic
inputs. They are indicative software measurements, not trained-checkpoint or
physical edge-device performance.

Usage:
    python -m src.benchmark --device cpu
    python -m src.benchmark --device cuda --runs 200 --batch-size 1
    python -m src.benchmark --out outputs/hardware_benchmark.csv
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn

from src.config import IMAGE_SIZE
from src.models.model_a import ModelA
from src.models.model_b import ModelB
from src.models.pretrained import build_pretrained
from src.utils import count_parameters


def compute_macs_and_flops(model: nn.Module, input_size: Tuple[int, int]) -> Tuple[int, int]:
    """Calculate theoretical MACs and FLOPs using PyTorch forward hooks."""
    total_macs = 0

    def conv_hook(module, inp, out):
        nonlocal total_macs
        output_h, output_w = out.shape[2], out.shape[3]
        kernel_h, kernel_w = module.kernel_size
        in_c = module.in_channels
        out_c = module.out_channels
        groups = module.groups
        macs = kernel_h * kernel_w * (in_c // groups) * out_c * output_h * output_w
        total_macs += macs

    def linear_hook(module, inp, out):
        nonlocal total_macs
        macs = module.in_features * module.out_features
        total_macs += macs

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    dummy = torch.randn(1, 3, *input_size)
    model.eval()
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    total_flops = total_macs * 2
    return total_macs, total_flops


def measure_latency(
    model: nn.Module,
    input_size: Tuple[int, int],
    batch_size: int,
    device: torch.device,
    warmup: int = 20,
    runs: int = 100,
) -> Dict[str, float]:
    """Measure inference latency and throughput under repeated passes."""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(batch_size, 3, *input_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)

    mean_batch_time_ms = (sum(timings) / runs) * 1000.0
    mean_per_sample_ms = mean_batch_time_ms / batch_size
    fps = (runs * batch_size) / sum(timings)

    return {
        "batch_latency_ms": round(mean_batch_time_ms, 2),
        "per_sample_latency_ms": round(mean_per_sample_ms, 3),
        "throughput_fps": round(fps, 1),
    }


def benchmark_all(device: torch.device, batch_size: int = 1, runs: int = 100) -> pd.DataFrame:
    """Benchmark Model A, Model B, MobileNetV2, and EfficientNet-B0."""
    models_config = [
        ("Model A (Standard CNN)", ModelA(num_classes=10), (IMAGE_SIZE, IMAGE_SIZE)),
        ("Model B (Depthwise-Sep. CNN)", ModelB(num_classes=10), (IMAGE_SIZE, IMAGE_SIZE)),
        ("MobileNetV2 (Pretrained)", build_pretrained("mobilenet_v2", num_classes=10), (IMAGE_SIZE, IMAGE_SIZE)),
        ("EfficientNet-B0 (Pretrained)", build_pretrained("efficientnet_b0", num_classes=10), (IMAGE_SIZE, IMAGE_SIZE)),
    ]

    records = []
    for name, model, img_size in models_config:
        params = count_parameters(model)
        macs, flops = compute_macs_and_flops(model, img_size)
        metrics = measure_latency(model, img_size, batch_size, device, runs=runs)

        records.append(
            {
                "Model": name,
                "Input Shape": f"3x{img_size[0]}x{img_size[1]}",
                "Parameters": f"{params:,}",
                "MACs (M)": round(macs / 1e6, 2),
                "FLOPs (M)": round(flops / 1e6, 2),
                "Latency (ms/img)": metrics["per_sample_latency_ms"],
                "Throughput (img/s)": metrics["throughput_fps"],
                "Device": device.type.upper(),
                "Measurement": "synthetic input, untrained model",
            }
        )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Profile FLOPs, MACs, and inference latency.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Target device to benchmark latency on.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size (default: 1 for edge latency).")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations (default: 100).")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save results as CSV.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\nBenchmark Profile (Device: {device} | Batch Size: {args.batch_size} | Runs: {args.runs})\n")

    df = benchmark_all(device=device, batch_size=args.batch_size, runs=args.runs)

    try:
        print(df.to_markdown(index=False))
    except (ImportError, Exception):
        print(df.to_string(index=False))
    print()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved benchmark results to {out_path}")


if __name__ == "__main__":
    main()
