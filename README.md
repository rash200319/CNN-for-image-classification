# Resource-Constrained CNN for Edge Image Classification

Implementation and experimental outputs for EN3150 Assignment 03.

## Dataset

- **Dataset**: EuroSAT (RGB)
- **Path**: `archive/EuroSAT` (configured in `src/config.py`)
- **Total images**: 27,000
- **Classes**: 10 (`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`)
- **Image resolution**: `64 x 64` pixels (`MAX_IMAGE_SIZE = 64`)
- **Split ratios**: 70% training (`18,900`), 15% validation (`4,049`), 15% testing (`4,051`)
- **Split method**: Stratified split (`SEED = 42`)

### Dataset Resolution Audit (`outputs/dataset_resolution_audit.txt`)

```text
Dataset: archive/EuroSAT
Total image files: 27000
Resolution counts:
  64 x 64: 27000 images
Maximum resolution: 64 x 64
Unreadable files: 0
PASS: All images satisfy the maximum 64 x 64 requirement.
```

## Project Structure

```text
src/
  config.py                      Dataset and training configuration (MAX_IMAGE_SIZE = 64)
  data/dataset.py                Dataset loading, transforms, and stratified splitting
  data/verify_dataset.py         Dataset statistical profiling tool
  models/model_a.py              Model A architecture (standard 2D CNN)
  models/model_b.py              Model B architecture (depthwise-separable CNN)
  models/blocks.py               Depthwise-separable convolution block
  models/pretrained.py           Pretrained MobileNetV2 and EfficientNet-B0 builders
  train.py                       Model training CLI
  evaluate.py                    Test evaluation and confusion matrix CLI
  benchmark.py                   MAC/FLOP and latency profiler CLI
  predict.py                     Single-image and batch inference CLI
  export.py                      TorchScript and ONNX exporter CLI
  plot_history.py                Loss and accuracy curve plotting CLI
  compare_runs.py                Run comparison table CLI
  utils.py                       Parameter counting utilities
tests/
  test_models.py                 Architecture parameter budget and shape tests
  test_benchmark.py              MAC/FLOP and resolution validation tests
  test_predict.py                Inference pipeline tests
  test_export.py                 TorchScript and ONNX export tests
  test_dataset_verification.py   Dataset split ratio tests
  test_dataset_resolution.py     Dataset resolution audit and unit tests
docs/
  DATASET.md                     EuroSAT dataset statistical profile
.github/workflows/
  ci.yml                         GitHub Actions test workflow (Python 3.11 & 3.12)
notebooks/
  EN3150_A03_kaggle_colab.ipynb  Executed Kaggle Tesla T4 notebook
outputs/                         Recorded checkpoints, configs, metrics, CSVs, and plots
report/                          LaTeX source (main.tex), figures/, and compiled main.pdf
```

## Model Architectures

### Model A (Standard CNN)

```text
Input: 3 x 64 x 64
Conv2D(3 -> 16, 3x3, pad=1) -> ReLU -> MaxPool2D(2x2)
Conv2D(16 -> 32, 3x3, pad=1) -> ReLU -> MaxPool2D(2x2)
Conv2D(32 -> 64, 3x3, pad=1) -> ReLU -> MaxPool2D(2x2)
Flatten (64 * 8 * 8 = 4096)
Linear(4096 -> 128) -> ReLU
Linear(128 -> 10)
```

- **Trainable parameters**: `549,290`

### Model B (Depthwise-Separable CNN)

```text
Input: 3 x 64 x 64
DepthwiseSeparableConv(3 -> 32, 3x3, pad=1) -> MaxPool2D(2x2)
DepthwiseSeparableConv(32 -> 64, 3x3, pad=1) -> MaxPool2D(2x2)
DepthwiseSeparableConv(64 -> 128, 3x3, pad=1) -> MaxPool2D(2x2)
AdaptiveAvgPool2D(1x1)
Linear(128 -> 10)
```

- **Trainable parameters**: `13,288` (`MODEL_B_MAX_PARAMS = 100,000`)

### Pretrained Models

- **MobileNetV2**: `2,236,682` trainable parameters (`input = 3 x 64 x 64`)
- **EfficientNet-B0**: `4,020,358` trainable parameters (`input = 3 x 64 x 64`)

## Recorded Experimental Outputs

### Training Environment (`outputs/*/config.json`)

- **Device**: `cuda` (Tesla T4)
- **Input size**: `64 x 64`
- **Seed**: `42`
- **PyTorch version**: `2.11.0+cu128`
- **NumPy version**: `2.1.3`

### Overall Run Summary (`outputs/all_results.csv`)

| Run | Model | Optimizer | LR | Momentum | Epochs | Params | Size (KB) | Size (MB) | Median Time/Epoch (s) | Best Val Acc (%) | Test Acc (%) | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `final_a_adam` | `a` | `adam` | `0.001` | `0.9` | 20 | 549,290 | 2150.1 | 2.100 | 8.53 | 89.40 | 90.03 | 0.9020 | 0.8946 | 0.8964 |
| `modelb_adam` | `b` | `adam` | `0.001` | `0.9` | 20 | 13,288 | 63.9 | 0.062 | 8.66 | 91.08 | 91.01 | 0.9087 | 0.9064 | 0.9067 |
| `modelb_sgd` | `b` | `sgd` | `0.01` | `0.9` | 20 | 13,288 | 63.9 | 0.062 | 8.42 | 79.82 | 79.88 | 0.7967 | 0.7879 | 0.7833 |
| `modelb_sgd_momentum` | `b` | `sgd_momentum` | `0.01` | `0.9` | 20 | 13,288 | 63.9 | 0.062 | 8.56 | 89.08 | 89.90 | 0.8963 | 0.8968 | 0.8957 |
| `final_mobilenet_v2` | `mobilenet_v2` | `adam` | `0.0001` | `0.9` | 5 | 2,236,682 | 8976.9 | 8.766 | 12.45 | 97.38 | 97.58 | 0.9752 | 0.9743 | 0.9747 |
| `final_efficientnet_b0` | `efficientnet_b0` | `adam` | `0.0001` | `0.9` | 5 | 4,020,358 | 15997.0 | 15.622 | 16.39 | 96.99 | 97.36 | 0.9732 | 0.9721 | 0.9726 |

### Per-Class Test Metrics (`outputs/*/metrics.json`)

#### Custom Models (`final_a_adam` and `modelb_adam`)

| Class | Support | `final_a_adam` Precision | `final_a_adam` Recall | `final_a_adam` F1 | `modelb_adam` Precision | `modelb_adam` Recall | `modelb_adam` F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AnnualCrop | 450 | 0.8848 | 0.9044 | 0.8945 | 0.8796 | 0.9089 | 0.8940 |
| Forest | 450 | 0.9231 | 0.9867 | 0.9538 | 0.9713 | 0.9778 | 0.9745 |
| HerbaceousVegetation | 451 | 0.8571 | 0.8780 | 0.8675 | 0.8580 | 0.9246 | 0.8901 |
| Highway | 375 | 0.9402 | 0.7547 | 0.8373 | 0.9250 | 0.7893 | 0.8518 |
| Industrial | 375 | 0.9267 | 0.9440 | 0.9353 | 0.9082 | 0.9760 | 0.9409 |
| Pasture | 300 | 0.9502 | 0.8267 | 0.8841 | 0.9033 | 0.9033 | 0.9033 |
| PermanentCrop | 375 | 0.8306 | 0.8240 | 0.8273 | 0.8228 | 0.8293 | 0.8260 |
| Residential | 450 | 0.9194 | 0.9889 | 0.9529 | 0.9752 | 0.9622 | 0.9687 |
| River | 375 | 0.8133 | 0.8827 | 0.8465 | 0.8533 | 0.8373 | 0.8452 |
| SeaLake | 450 | 0.9751 | 0.9556 | 0.9652 | 0.9908 | 0.9556 | 0.9729 |

#### Model B SGD Variants (`modelb_sgd` and `modelb_sgd_momentum`)

| Class | Support | `modelb_sgd` Precision | `modelb_sgd` Recall | `modelb_sgd` F1 | `modelb_sgd_momentum` Precision | `modelb_sgd_momentum` Recall | `modelb_sgd_momentum` F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AnnualCrop | 450 | 0.8676 | 0.8156 | 0.8408 | 0.8568 | 0.9178 | 0.8863 |
| Forest | 450 | 0.8521 | 0.9733 | 0.9087 | 0.9567 | 0.9822 | 0.9693 |
| HerbaceousVegetation | 451 | 0.6362 | 0.9113 | 0.7493 | 0.9152 | 0.8137 | 0.8615 |
| Highway | 375 | 0.6571 | 0.5520 | 0.6000 | 0.8051 | 0.8480 | 0.8260 |
| Industrial | 375 | 0.8476 | 0.9493 | 0.8956 | 0.9030 | 0.9680 | 0.9344 |
| Pasture | 300 | 0.7508 | 0.8133 | 0.7808 | 0.8954 | 0.9133 | 0.9043 |
| PermanentCrop | 375 | 0.7000 | 0.5413 | 0.6105 | 0.8189 | 0.8320 | 0.8254 |
| Residential | 450 | 0.9036 | 0.9578 | 0.9299 | 0.9905 | 0.9289 | 0.9587 |
| River | 375 | 0.7857 | 0.4693 | 0.5876 | 0.8547 | 0.7840 | 0.8178 |
| SeaLake | 450 | 0.9664 | 0.8956 | 0.9296 | 0.9671 | 0.9800 | 0.9735 |

#### Pretrained Models (`final_mobilenet_v2` and `final_efficientnet_b0`)

| Class | Support | `final_mobilenet_v2` Precision | `final_mobilenet_v2` Recall | `final_mobilenet_v2` F1 | `final_efficientnet_b0` Precision | `final_efficientnet_b0` Recall | `final_efficientnet_b0` F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AnnualCrop | 450 | 0.9524 | 0.9778 | 0.9649 | 0.9581 | 0.9667 | 0.9624 |
| Forest | 450 | 0.9912 | 0.9956 | 0.9933 | 0.9890 | 0.9956 | 0.9922 |
| HerbaceousVegetation | 451 | 0.9649 | 0.9756 | 0.9702 | 0.9465 | 0.9800 | 0.9630 |
| Highway | 375 | 0.9753 | 0.9467 | 0.9608 | 0.9653 | 0.9653 | 0.9653 |
| Industrial | 375 | 0.9946 | 0.9813 | 0.9879 | 0.9867 | 0.9867 | 0.9867 |
| Pasture | 300 | 0.9567 | 0.9567 | 0.9567 | 0.9596 | 0.9500 | 0.9548 |
| PermanentCrop | 375 | 0.9592 | 0.9413 | 0.9502 | 0.9617 | 0.9387 | 0.9501 |
| Residential | 450 | 0.9890 | 0.9978 | 0.9934 | 0.9889 | 0.9889 | 0.9889 |
| River | 375 | 0.9735 | 0.9813 | 0.9774 | 0.9783 | 0.9627 | 0.9704 |
| SeaLake | 450 | 0.9955 | 0.9889 | 0.9922 | 0.9978 | 0.9867 | 0.9922 |

### Synthetic-Input Benchmark Outputs (`outputs/benchmark_gpu.csv` & `outputs/benchmark_cpu.csv`)

#### GPU (`CUDA`)

| Model | Input Shape | Parameters | MACs (M) | FLOPs (M) | Latency (ms/img) | Throughput (img/s) | Device | Measurement |
|---|---|---:|---:|---:|---:|---:|---|---|
| Model A (Standard CNN) | `3x64x64` | 549,290 | 11.73 | 23.46 | 0.411 | 2434.3 | CUDA | synthetic input, untrained model |
| Model B (Depthwise-Sep. CNN) | `3x64x64` | 13,288 | 5.14 | 10.28 | 0.616 | 1624.2 | CUDA | synthetic input, untrained model |
| MobileNetV2 (Pretrained) | `3x64x64` | 2,236,682 | 24.46 | 48.92 | 4.924 | 203.1 | CUDA | synthetic input, untrained model |
| EfficientNet-B0 (Pretrained) | `3x64x64` | 4,020,358 | 31.98 | 63.96 | 7.588 | 131.8 | CUDA | synthetic input, untrained model |

#### CPU (`CPU`)

| Model | Input Shape | Parameters | MACs (M) | FLOPs (M) | Latency (ms/img) | Throughput (img/s) | Device | Measurement |
|---|---|---:|---:|---:|---:|---:|---|---|
| Model A (Standard CNN) | `3x64x64` | 549,290 | 11.73 | 23.46 | 1.203 | 831.5 | CPU | synthetic input, untrained model |
| Model B (Depthwise-Sep. CNN) | `3x64x64` | 13,288 | 5.14 | 10.28 | 2.399 | 416.8 | CPU | synthetic input, untrained model |
| MobileNetV2 (Pretrained) | `3x64x64` | 2,236,682 | 24.46 | 48.92 | 7.695 | 130.0 | CPU | synthetic input, untrained model |
| EfficientNet-B0 (Pretrained) | `3x64x64` | 4,020,358 | 31.98 | 63.96 | 10.863 | 92.1 | CPU | synthetic input, untrained model |

### Edge Export Outputs (`outputs/edge_export_report.txt`)

| Model | Checkpoint | Input Shape | Params | PyTorch (KB) | TorchScript (KB) | TorchScript Max Abs Diff | ONNX Graph (KB) | ONNX Max Abs Diff |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `A` | `outputs/final_a_adam/best_model.pt` | `3x64x64` | 549,290 | 2150.1 | 2171.7 | `0.00e+00` | 19.4 | `7.63e-06` |
| `B` | `outputs/modelb_adam/best_model.pt` | `3x64x64` | 13,288 | 63.9 | 92.6 | `0.00e+00` | 26.7 | `9.16e-05` |

## Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models:

```bash
python -m src.train --model a --optimizer adam --lr 1e-3 --epochs 20 --run-name final_a_adam
python -m src.train --model b --optimizer adam --lr 1e-3 --epochs 20 --run-name modelb_adam
python -m src.train --model b --optimizer sgd --lr 1e-2 --epochs 20 --run-name modelb_sgd
python -m src.train --model b --optimizer sgd_momentum --lr 1e-2 --momentum 0.9 --epochs 20 --run-name modelb_sgd_momentum
python -m src.train --model mobilenet_v2 --optimizer adam --lr 1e-4 --img-size 64 --epochs 5 --run-name final_mobilenet_v2
python -m src.train --model efficientnet_b0 --optimizer adam --lr 1e-4 --img-size 64 --epochs 5 --run-name final_efficientnet_b0
```

Evaluate checkpoints and plot curves:

```bash
python -m src.evaluate --model a --checkpoint outputs/final_a_adam/best_model.pt --img-size 64
python -m src.evaluate --model b --checkpoint outputs/modelb_adam/best_model.pt --img-size 64
python -m src.evaluate --model b --checkpoint outputs/modelb_sgd/best_model.pt --img-size 64
python -m src.evaluate --model b --checkpoint outputs/modelb_sgd_momentum/best_model.pt --img-size 64
python -m src.evaluate --model mobilenet_v2 --checkpoint outputs/final_mobilenet_v2/best_model.pt --img-size 64
python -m src.evaluate --model efficientnet_b0 --checkpoint outputs/final_efficientnet_b0/best_model.pt --img-size 64
python -m src.plot_history --run-dir outputs/final_a_adam
python -m src.plot_history --run-dir outputs/modelb_adam
```

Run benchmarks, exports, dataset verification, and unit tests:

```bash
python -m src.benchmark --device cuda --runs 200 --out outputs/benchmark_gpu.csv
python -m src.benchmark --device cpu --runs 100 --out outputs/benchmark_cpu.csv
python -m src.export --model a --checkpoint outputs/final_a_adam/best_model.pt
python -m src.export --model b --checkpoint outputs/modelb_adam/best_model.pt
python -m src.data.verify_dataset --export-md docs/DATASET.md
python tests/test_dataset_resolution.py --dataset-dir archive/EuroSAT
python -m unittest discover -s tests -v
```
