# Resource-Constrained CNN for Edge Image Classification

This repository contains the implementation for the EN3150 Assignment 03
resource-constrained image-classification task.

## Current completion status

The custom CNN foundation is implemented and tested locally. The complete
assignment report and all required experiments are not yet finished.

| Area | Status |
|---|---|
| EuroSAT dataset and preprocessing | Complete |
| 70/15/15 stratified data split | Complete |
| Model A standard CNN | Complete |
| Model B depthwise-separable CNN | Complete |
| Model B parameter budget | Complete |
| Model A Adam training | Complete |
| Model B Adam training | Complete |
| SGD and SGD-with-Momentum comparison | Complete for Model B |
| Test evaluation and confusion matrices | Complete for completed runs |
| Training/validation loss plots | Complete |
| Two pretrained lightweight SOTA models | Complete |
| Custom-model vs. SOTA comparison table | Outstanding |
| Final written report and trade-off discussion | Outstanding |

## Dataset

The project uses the RGB version of the EuroSAT dataset. It is stored locally
under `archive/EuroSAT`.

- Images: 27,000
- Classes: 10
- Image resolution: 64x64 pixels
- Training split: 18,900 images
- Validation split: 4,049 images
- Test split: 4,051 images
- Split method: stratified split with random seed 42

The small rounding difference in the split counts is due to integer sample
allocation. Images are resized to 64x64 by the data pipeline.

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
  config.py                  Dataset and training configuration
  data/dataset.py            Dataset loading, augmentation, and splitting
  models/model_a.py          Standard convolutional CNN
  models/model_b.py          Lightweight depthwise-separable CNN
  models/blocks.py           Reusable depthwise-separable block
  train.py                   Training entry point
  evaluate.py                Test metrics and confusion-matrix generation
  utils.py                   Shared utilities
outputs/
  modela_adam/              Existing Model A training artifacts
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

Model A has been trained with Adam for 20 epochs:

- Run directory: `outputs/modela_adam`
- History: `outputs/modela_adam/history.csv`
- Checkpoint: `outputs/modela_adam/best_model.pt`
- Final training accuracy: **91.99%**
- Best recorded validation accuracy: **88.84%**
- Approximate checkpoint size: **2,150.1 KB**
- Approximate average epoch time: **48 seconds**

The final test evaluation for this checkpoint is:

- Test accuracy: **89.66%**
- Macro precision: **0.9021**
- Macro recall: **0.8920**

The validation accuracy and test accuracy are reported separately.

Model B has now been trained for 20 epochs with Adam, plain SGD, and SGD with
momentum. The resulting test metrics are:

| Run | Test accuracy | Macro precision | Macro recall | Average epoch time |
|---|---:|---:|---:|---:|
| Model B + Adam | **89.76%** | 0.8948 | 0.8963 | 92.08 s |
| Model B + SGD | 82.35% | 0.8163 | 0.8161 | 87.64 s |
| Model B + SGD + momentum (0.9) | 89.41% | 0.8924 | 0.8913 | 87.63 s |

The Adam and momentum runs substantially outperformed plain SGD under these
learning-rate settings. Momentum improved SGD convergence by carrying a
velocity term from previous updates, reducing oscillation and helping the
optimizer move through shallow or noisy regions of the loss surface. In this
experiment, momentum reached a much stronger result than plain SGD and was
close to Adam.

Generated artifacts are stored in:

- `outputs/modela_adam/`
- `outputs/modelb_adam/`
- `outputs/modelb_sgd/`
- `outputs/modelb_sgd_momentum/`

Approximate FP32 parameter storage calculated from the verified parameter
counts:

| Model | Trainable parameters | Approximate FP32 size |
|---|---:|---:|
| Model A | 549,290 | 2,145.7 KB |
| Model B | 13,288 | 51.9 KB |

The actual checkpoint size can differ because checkpoints may also contain
metadata or serialization overhead.

Custom-model Adam comparison:

| Model | Parameters | Checkpoint size | Average epoch time | Test accuracy |
|---|---:|---:|---:|---:|
| Model A + Adam | 549,290 | 2,150.1 KB | 49.65 s | 89.66% |
| Model B + Adam | 13,288 | 63.4 KB | 92.08 s | 89.76% |

Model B uses approximately 97.6% fewer trainable parameters than Model A and
achieved a very similar test accuracy in this experiment. Its CPU epoch time
was higher in these runs, so parameter count and serialized size should not be
treated as a direct measurement of wall-clock speed. Actual deployment cost
also depends on the hardware kernel implementation and memory bandwidth.

## SOTA fine-tuning (MobileNetV2 and EfficientNet-B0)

Two ImageNet-pretrained lightweight models were fine-tuned end-to-end on the
same EuroSAT splits, for 5 epochs each with Adam (lr=1e-4). Pretrained models
need a larger input than the 64x64 used for Model A/B, so `src/train.py` and
`src/evaluate.py` accept an `--img-size` flag; ImageNet normalization
statistics are used automatically for these two models (see
`src/models/pretrained.py`).

| Model | Input size | Parameters | Checkpoint size | Test accuracy | Macro precision | Macro recall |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 128x128 | 2,236,682 | 8,972.6 KB | **98.32%** | 0.9826 | 0.9823 |
| EfficientNet-B0 | 64x64 | 4,020,358 | 15,992.1 KB | **97.70%** | 0.9764 | 0.9761 |

Both comfortably beat the from-scratch custom models (Model A: 89.66-90.27%,
Model B: 82.35-89.76%, depending on run/optimizer) after only 5 epochs, as
expected for transfer learning from ImageNet features. The same confusion
pairs seen in Model A (Highway/River, HerbaceousVegetation/PermanentCrop,
AnnualCrop/PermanentCrop) still show up in the confusion matrices here, just
far smaller in magnitude, suggesting they're genuinely hard classes in this
dataset rather than a weakness specific to the custom models.

Per-epoch timing on this CPU-only machine was inconsistent (occasional
epochs ran 3-5x longer than the rest, likely due to background system load),
so epoch-time figures for these two runs should be treated as approximate
and not used directly in the time/epoch comparison column.

Generated artifacts:

- `outputs/final_mobilenet_v2/` (checkpoint, history, confusion matrix, loss
  curve, metrics.json)
- `outputs/final_efficientnet_b0/` (same)

To reproduce:

```bash
python -m src.train --model mobilenet_v2 --optimizer adam --lr 1e-4 --img-size 128 --epochs 5 --run-name final_mobilenet_v2
python -m src.evaluate --model mobilenet_v2 --checkpoint outputs/final_mobilenet_v2/best_model.pt --img-size 128
python -m src.plot_history --run-dir outputs/final_mobilenet_v2

python -m src.train --model efficientnet_b0 --optimizer adam --lr 1e-4 --img-size 64 --epochs 5 --run-name final_efficientnet_b0
python -m src.evaluate --model efficientnet_b0 --checkpoint outputs/final_efficientnet_b0/best_model.pt --img-size 64
python -m src.plot_history --run-dir outputs/final_efficientnet_b0
```

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

## Cloud Execution (Kaggle / Colab)

A self-contained notebook is available at `notebooks/EN3150_A03_kaggle_colab.ipynb` to run all experiments (Model A, Model B with 3 optimizers, MobileNetV2, EfficientNet-B0, evaluation, plots, and comparison tables) under a single GPU for consistent benchmark timings.

### Running on Kaggle:
1. Open Kaggle -> **Create -> New Notebook -> File -> Import Notebook** and select `notebooks/EN3150_A03_kaggle_colab.ipynb`.
2. In the right panel (**Session options**):
   - **Accelerator**: Set to **GPU T4 x2** (or P100).
   - **Internet**: Toggle **On** (requires phone verification on Kaggle).
3. Click **Run All** (or **Save Version -> Save & Run All (Commit)** to execute headlessly in the background).
4. After completion, download `results.zip` from the notebook's **Output** tab and extract into `outputs/`.

## Work remaining before submission

The following tasks still need to be completed:

1. **Re-run / Reconcile benchmarks on unified hardware**:
   - Run all models under the same GPU (e.g., via the Kaggle notebook) so that "Time per epoch" in the comparison table is fair and consistent across architectures.
   - Reconcile Model A Adam test accuracy (89.66% vs 90.27%) so the report references canonical outputs.
2. **Include outputs & artifacts**:
   - Ensure final checkpoints and metrics for SOTA models (`final_mobilenet_v2`, `final_efficientnet_b0`) are saved and present in `outputs/`.
3. **Report updates (`report/main.tex`)**:
   - Update tables and numbers with the finalized metrics.
   - SOTA sizes should explicitly include **MB** values in addition to KB as requested in the assignment PDF.
   - Fill in group member names, index numbers, and the GitHub repository link (lines 21–23).
   - Confirm that the dataset information sheet referenced in the assignment brief has been filled in, and remove the `\todofill` note.
   - Review and customize the discussion and trade-off write-up in your own words.
4. **Final Submission**:
   - Compile `report/main.tex` to PDF.
   - Package code and report separately according to course guidelines (`GroupNo_A03_EN3150`).

## Important notes

- The dataset split is generated deterministically using seed 42.
- Model B's parameter budget (<= 100,000 parameters) is strictly enforced with an assertion in the model constructor.
- Regular Git commits are required by the module instructors to demonstrate project progress over time.
