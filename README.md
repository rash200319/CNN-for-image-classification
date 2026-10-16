# Resource-Constrained CNN for Edge Image Classification

This repository contains the implementation for the EN3150 Assignment 03
resource-constrained image-classification task.

## Current completion status

The custom CNN foundation, both SOTA fine-tunes, and the written report are
all implemented. One experiment (compliant 64x64 MobileNetV2) is still
outstanding.

| Area | Status |
|---|---|
| EuroSAT dataset and preprocessing | Complete |
| 70/15/15 stratified data split | Complete |
| 64x64 maximum-resolution enforcement | Complete (code-level, via `validate_image_size`) |
| Model A standard CNN | Complete |
| Model B depthwise-separable CNN | Complete |
| Model B parameter budget | Complete |
| Model A Adam training | Complete (91.46%, Tesla T4) |
| Model B Adam training | Complete (89.61%, Tesla T4) |
| SGD and SGD-with-Momentum comparison | Complete for Model B |
| Test evaluation and confusion matrices | Complete for completed runs |
| Training/validation loss plots | Complete for recorded runs |
| EfficientNet-B0 (64x64, compliant) | Complete (97.51%) |
| MobileNetV2 (64x64, compliant) | **Outstanding** — only a 128x128 "optional study" (98.20%) exists |
| Author details, dataset sheet confirmation | Complete |
| Final written report (`report/main.tex`) | Complete |
| `report/main.pdf` matches `main.tex` | Complete (recompile after any `.tex` edit — it does not happen automatically) |

The report's final Model A result (91.46%) comes from the Kaggle/T4
`final_a_adam` run. The local `outputs/modela_adam` directory is a separate
local run (89.66%) and must not be mixed with the report's Kaggle metrics.

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

For the RGB EuroSAT dataset, the expected result is 27,000 images at 64x64
with no unreadable files, followed by a `PASS` message.

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
  benchmark.py                Hardware latency, peak memory, and MAC/FLOP profiler
  predict.py                  Single-image and batch land-use inference CLI
  export.py                   Edge deployment exporter (TorchScript & ONNX) with parity checks
  plot_history.py            Training/validation curve generator
  compare_runs.py            Comparison table aggregator across runs
  utils.py                   Shared parameter counting and helper utilities
tests/
  test_models.py              Architecture budget and tensor shape assertions
  test_benchmark.py           MAC/FLOP calculation assertions
  test_predict.py             Inference pipeline validation
  test_export.py              TorchScript and ONNX export parity assertions
  test_dataset_verification.py  Dataset split distribution assertions
  test_dataset_resolution.py  Portable full-dataset resolution audit (see below)
docs/
  DATASET.md                  EuroSAT statistical profile and class breakdown
.github/workflows/
  ci.yml                      CI: runs the test suite on Python 3.11 & 3.12
notebooks/
  EN3150_A03_kaggle_colab.ipynb  Cloud GPU training run (Model A/B, SOTA, plots)
outputs/                      Checkpoints, metrics.json, history.csv, plots per run
report/                       LaTeX source (main.tex), figures/, and main.pdf
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

## Existing training evidence

**The numbers below are the report's authoritative results** (`report/main.tex`,
sourced from the Kaggle Tesla T4 notebook run). Both Model A and Model B were
trained for 20 epochs with Adam (lr=1e-3) unless noted otherwise.

| Model | Parameters | Checkpoint size | Median time/epoch (T4 GPU) | Test accuracy |
|---|---:|---:|---:|---:|
| Model A (Adam) | 549,290 | 2,150.1 KB | 10.0 s | **91.46%** |
| Model B (Adam) | 13,288 | 63.9 KB | 10.3 s | **89.61%** |

Model B optimizer comparison (20 epochs each, Tesla T4 GPU):

| Run | Test accuracy | Macro precision | Macro recall |
|---|---:|---:|---:|
| Model B + Adam | **89.61%** | 0.8975 | 0.8918 |
| Model B + SGD | 81.26% | 0.8095 | 0.8042 |
| Model B + SGD + momentum (0.9) | 87.71% | 0.8822 | 0.8723 |

Adam and SGD+momentum (same lr=1e-2 as plain SGD, isolating momentum's
effect) both substantially outperform plain SGD. Momentum's velocity term
dampens oscillation and effectively increases the step size in consistent
gradient directions, closing most of the gap to Adam despite the much
simpler update rule.

Model B uses about 97.6% fewer trainable parameters than Model A for a
1.85-percentage-point accuracy cost — a strong case for the
depthwise-separable design. See `report/main.tex` Section 4 for the full
discussion, loss curves, and confusion matrices.

**Known gap:** `outputs/modelb_adam`, `outputs/modelb_sgd`, and
`outputs/modelb_sgd_momentum` contain the actual Tesla T4 checkpoints/metrics
matching the numbers above. `outputs/final_a_adam` still contains an older
local CPU run (90.27%) rather than the Kaggle-downloaded 91.46% artifacts —
the report's Model A figures were regenerated from the Kaggle run, but the
matching `metrics.json`/`history.csv`/checkpoint were not re-downloaded into
this repo. Also present: `outputs/modela_adam`, a separate earlier local run
(89.66%) — do not confuse either with the report's 91.46% figure.

## SOTA fine-tuning (MobileNetV2 and EfficientNet-B0)

Two ImageNet-pretrained models were fine-tuned end-to-end on the same splits,
5 epochs each, Adam (lr=1e-4):

| Model | Input | Params | Size (MB) | Median time/epoch (T4) | Test accuracy |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 64x64 (compliant) | 4,020,358 | 15.62 | 18.5 s | **97.51%** |
| MobileNetV2 | 128x128 (**optional study only**) | 2,236,682 | 8.77 | 28.6 s | 98.20% |

**MobileNetV2 has not yet been retrained at the compliant 64x64 resolution.**
The 128x128 result above must not be presented as the compliant final
number — see "Resolution and benchmark policy" below. Until a 64x64 rerun
exists, EfficientNet-B0 is the only fully compliant SOTA result.

Same `outputs/` gap as above: `outputs/final_mobilenet_v2` and
`outputs/final_efficientnet_b0` still hold the earlier local CPU-run
artifacts (98.32%/97.70%), not the Tesla T4 numbers cited in the report
(98.20%/97.51%). Re-download the Kaggle run's `outputs/` folder to fix this.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Model A with Adam:

```bash
python -m src.train --model a --optimizer adam --lr 1e-3 --epochs 20
```

Run Model B with Adam:

```bash
python -m src.train --model b --optimizer adam --lr 1e-3 --epochs 20
```

Run plain SGD:

```bash
python -m src.train --model b --optimizer sgd --lr 1e-2 --epochs 20
```

Run SGD with Momentum:

```bash
python -m src.train --model b --optimizer sgd_momentum --lr 1e-2 --momentum 0.9 --epochs 20
```

Evaluate a trained checkpoint:

```bash
python -m src.evaluate --model b --checkpoint outputs/modelb_adam/best_model.pt
```

Evaluation writes `metrics.json` and `confusion_matrix.png` into the
checkpoint's output directory.

## Submission checklist

1. Run the notebook on Kaggle/Colab after pulling the branch containing the
   final code.
2. Download and preserve the generated `final_a_adam` and compliant
   64x64 MobileNetV2 artifacts.
3. Compile `report/main.tex`, inspect all figures and tables, and rename the
   final PDF using the required Moodle filename format.

## Resolution and benchmark policy

The assignment permits a maximum image resolution of 64x64. The shared
`validate_image_size` guard enforces this limit across training,
preprocessing, prediction, export, and benchmark entry points. The notebook
now schedules MobileNetV2 at 64x64 for the compliant run. The historical
128x128 MobileNetV2 result is retained only as an optional resolution study
and must not be presented as the compliant final result.

The benchmark uses synthetic inputs and newly initialized models. Its MAC
values are theoretical convolution/linear estimates, and its latency values
are indicative software measurements rather than physical edge-device
performance.

## Reproducing and recording results

The notebook `notebooks/EN3150_A03_kaggle_colab.ipynb` clones the repository,
runs the custom 20-epoch experiments, evaluates the two pretrained models,
generates plots and confusion matrices, and writes `outputs/all_results.csv`.
Run it on Kaggle or Colab with a T4/P100 GPU and Internet enabled. Save and
download the generated `outputs/` directory, including `final_a_adam`, before
comparing the report with the local checkout. The notebook has been reset to
clear stale outputs before rerunning; its 64x64 MobileNetV2 run must produce
a new result rather than reusing the historical 128x128 result.

The local unit-test suite is run with:

```bash
python -m unittest discover -s tests
```

Latest local validation on `assignment-compliance-sync`:

```text
Ran 15 tests in 1.733s
OK (skipped=1)
```

The skipped test requires optional ONNX runtime support. The full notebook
rerun was not executed in this Windows checkout because the configured
`archive\EuroSAT` path is absent and the notebook requires a Kaggle/Colab GPU
for the complete 20-epoch custom and 5-epoch SOTA experiments. The notebook
has been reset to a clean state and pushed to this branch; run all cells on
Kaggle/Colab, then download and preserve the generated result files.

## Important notes

- The dataset split is generated deterministically using seed 42.
- Model B's parameter budget (<= 100,000 parameters) is enforced with an
  assertion in the model constructor; all entry points additionally reject
  `--img-size` above 64 via `validate_image_size`.
- `outputs/` is gitignored going forward, but several run directories
  (`modela_adam`, `modelb_adam`, `modelb_sgd`, `modelb_sgd_momentum`,
  `final_a_adam`, `final_mobilenet_v2`, `final_efficientnet_b0`) were
  committed earlier and are still tracked — see the "known gap" notes above
  for which of these match the report's current numbers.
- Both pretrained SOTA models (MobileNetV2, EfficientNet-B0) and the final
  LaTeX report (`report/main.tex` / `report/main.pdf`) are implemented and
  present in this repository; only the compliant 64x64 MobileNetV2 rerun is
  still outstanding.
