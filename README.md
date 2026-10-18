# Resource-Constrained CNN for Edge Image Classification

This repository contains the implementation for the EN3150 Assignment 03
resource-constrained image-classification task.

## Current completion status

All custom CNN architectures, optimizer comparisons, 64x64 SOTA fine-tunes,
hardware benchmarks, edge exports, and the final written report (`report/main.tex`
and `report/main.pdf`) are **100% complete** and synchronized.

| Area | Status |
|---|---|
| EuroSAT dataset and preprocessing | Complete |
| 70/15/15 stratified data split | Complete |
| 64x64 maximum-resolution enforcement | Complete (code-level via `validate_image_size` & dataset audit) |
| Model A standard CNN | Complete (549,290 params) |
| Model B depthwise-separable CNN | Complete (13,288 params, $\le 100,000$ budget) |
| Model A Adam training (64x64) | Complete (**90.03%**, Tesla T4) |
| Model B Adam training (64x64) | Complete (**91.01%**, Tesla T4) |
| SGD and SGD-with-Momentum comparison | Complete for Model B (**79.88%** vs. **89.90%**, Tesla T4) |
| Test evaluation and confusion matrices | Complete for all runs |
| Training/validation loss plots | Complete for all runs |
| MobileNetV2 (64x64, compliant) | Complete (**97.58%**, Tesla T4) |
| EfficientNet-B0 (64x64, compliant) | Complete (**97.36%**, Tesla T4) |
| Author details, dataset sheet confirmation | Complete |
| Final written report (`report/main.tex` & `report/main.pdf`) | Complete |

## Dataset

The project uses the RGB version of the EuroSAT dataset. It is stored locally
under `archive/EuroSAT` (see `src/config.py`).

- Images: 27,000
- Classes: 10
- Image resolution: 64x64 pixels
- Training split: 18,900 images
- Validation split: 4,049 images
- Test split: 4,051 images
- Split method: stratified split with random seed 42

The small rounding difference in the split counts is due to integer sample
allocation. Images are resized to 64x64 by the data pipeline. Training, prediction,
export, and benchmarking reject requested sizes above 64x64.

### Portable dataset-resolution audit

The repository includes a full-dataset audit that does not assume a
machine-specific path. It scans every supported image, reports the resolution
distribution, and fails if an image is larger than the assignment maximum:

```bash
python tests/test_dataset_resolution.py --dataset-dir /path/to/EuroSAT
```

The dataset location can also be supplied through `EUROSAT_DIR`:

```bash
EUROSAT_DIR=/path/to/EuroSAT python tests/test_dataset_resolution.py
```

On Windows PowerShell:

```powershell
$env:EUROSAT_DIR = "D:\datasets\EuroSAT"
python tests/test_dataset_resolution.py
```

For the RGB EuroSAT dataset, the verified result (`outputs/dataset_resolution_audit.txt`)
is 27,000 images at 64x64 with 0 unreadable files (`PASS`).

Classes:

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

## Project structure

```text
src/
  config.py                  Dataset/training config; MAX_IMAGE_SIZE=64 enforcement
  data/dataset.py            Dataset loading, augmentation, and splitting
  data/verify_dataset.py     Dataset verification and statistical profiling tool
  models/model_a.py          Standard convolutional CNN (549k params)
  models/model_b.py          Lightweight depthwise-separable CNN (13k params)
  models/blocks.py           Reusable depthwise-separable conv block
  models/pretrained.py       ImageNet backbones (MobileNetV2, EfficientNet-B0)
  train.py                   Training entry point (--img-size capped at 64)
  evaluate.py                Test metrics and confusion-matrix generation
  benchmark.py               Hardware latency, peak memory, and MAC/FLOP profiler
  predict.py                 Single-image and batch land-use inference CLI
  export.py                  Edge deployment exporter (TorchScript & ONNX) with parity checks
  plot_history.py            Training/validation curve generator
  compare_runs.py            Comparison table aggregator across runs
  utils.py                   Shared parameter counting and helper utilities
tests/
  test_models.py             Architecture budget and tensor shape assertions
  test_benchmark.py          MAC/FLOP calculation assertions
  test_predict.py            Inference pipeline validation
  test_export.py             TorchScript and ONNX export parity assertions
  test_dataset_verification.py  Dataset split distribution assertions
  test_dataset_resolution.py Portable full-dataset resolution audit
docs/
  DATASET.md                 EuroSAT statistical profile and class breakdown
.github/workflows/
  ci.yml                     CI: runs the test suite on Python 3.11 & 3.12
notebooks/
  EN3150_A03_kaggle_colab.ipynb  Executed cloud GPU training run (Model A/B, SOTA, plots)
outputs/                     Checkpoints, config.json, metrics.json, history.csv, plots per run
report/                      LaTeX source (main.tex), figures/, and main.pdf
```

## Custom CNNs

### Model A: Standard CNN

Model A uses regular 2D convolutions:

```text
Input: 3 x 64 x 64
Conv2D(3 -> 16, 3x3) -> ReLU -> MaxPool(2x2)
Conv2D(16 -> 32, 3x3) -> ReLU -> MaxPool(2x2)
Conv2D(32 -> 64, 3x3) -> ReLU -> MaxPool(2x2)
Flatten
Linear(64*8*8 -> 128) -> ReLU
Linear(128 -> 10)
```

Verified trainable parameter count: **549,290**.

### Model B: Lightweight CNN

Model B replaces regular convolutions with depthwise-separable convolutions
and uses global average pooling:

```text
Input: 3 x 64 x 64
DepthwiseSeparableConv(3 -> 32, 3x3) -> MaxPool(2x2)
DepthwiseSeparableConv(32 -> 64, 3x3) -> MaxPool(2x2)
DepthwiseSeparableConv(64 -> 128, 3x3) -> MaxPool(2x2)
Adaptive Global Average Pooling
Linear(128 -> 10)
```

Verified trainable parameter count: **13,288**.

This is below the assignment limit of 100,000 trainable parameters. ReLU is
used because it is inexpensive on edge hardware and avoids the exponential
operations required by sigmoid or tanh.

## Training & Evaluation Results (Tesla T4 GPU, `64x64`, `seed=42`)

**All results below are synchronized across `outputs/`, `notebooks/EN3150_A03_kaggle_colab.ipynb`, and `report/main.tex` (`report/main.pdf`).** Both Model A and Model B were trained for 20 epochs with Adam (`lr=1e-3`) unless noted otherwise.

| Model | Parameters | Checkpoint size | Median time/epoch (T4 GPU) | Test accuracy |
|---|---:|---:|---:|---:|
| Model A (`final_a_adam`) | 549,290 | 2,150.1 KB | 8.53 s | 90.03% |
| Model B (`modelb_adam`) | 13,288 | 63.9 KB | 8.66 s | **91.01%** |

Model B optimizer comparison (20 epochs each, Tesla T4 GPU):

| Run | Test accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Model B + Adam (`modelb_adam`) | **91.01%** | 0.9087 | 0.9064 | 0.9067 |
| Model B + SGD (`modelb_sgd`) | 79.88% | 0.7967 | 0.7879 | 0.7833 |
| Model B + SGD + momentum 0.9 (`modelb_sgd_momentum`) | 89.90% | 0.8963 | 0.8968 | 0.8957 |

Adam and SGD+momentum (same `lr=1e-2` as plain SGD, isolating momentum's
effect) both substantially outperform plain SGD (+10.02 percentage points for
momentum). Model B uses 97.6% fewer trainable parameters than Model A while
outperforming it by 0.98 percentage points on the test set thanks to batch
normalization and global average pooling.

## SOTA fine-tuning (MobileNetV2 and EfficientNet-B0 at `64x64`)

Two ImageNet-pretrained models were fine-tuned end-to-end on the same splits at
the assignment-compliant `64x64` resolution, 5 epochs each, Adam (`lr=1e-4`):

| Model | Input | Params | Size (MB) | MACs (M) | Median time/epoch (T4) | Test accuracy | Macro P / R |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 (`final_mobilenet_v2`) | 64x64 | 2,236,682 | 8.77 | 24.46 | 12.45 s | **97.58%** | 0.975 / 0.974 |
| EfficientNet-B0 (`final_efficientnet_b0`) | 64x64 | 4,020,358 | 15.62 | 31.98 | 16.39 s | **97.36%** | 0.973 / 0.972 |

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Model A with Adam:

```bash
python -m src.train --model a --optimizer adam --lr 1e-3 --epochs 20 --run-name final_a_adam
```

Run Model B with Adam:

```bash
python -m src.train --model b --optimizer adam --lr 1e-3 --epochs 20 --run-name modelb_adam
```

Run plain SGD:

```bash
python -m src.train --model b --optimizer sgd --lr 1e-2 --epochs 20 --run-name modelb_sgd
```

Run SGD with Momentum:

```bash
python -m src.train --model b --optimizer sgd_momentum --lr 1e-2 --momentum 0.9 --epochs 20 --run-name modelb_sgd_momentum
```

Fine-tune SOTA models at 64x64:

```bash
python -m src.train --model mobilenet_v2 --optimizer adam --lr 1e-4 --img-size 64 --epochs 5 --run-name final_mobilenet_v2
python -m src.train --model efficientnet_b0 --optimizer adam --lr 1e-4 --img-size 64 --epochs 5 --run-name final_efficientnet_b0
```

Evaluate a trained checkpoint:

```bash
python -m src.evaluate --model b --checkpoint outputs/modelb_adam/best_model.pt
```

Run the unit-test suite:

```bash
python -m unittest discover -s tests -v
```
