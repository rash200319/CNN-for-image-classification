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
| Training/validation loss plots | Outstanding |
| Two pretrained lightweight SOTA models | Outstanding |
| Final written report and trade-off discussion | Outstanding |

## Dataset

The project uses the RGB version of the EuroSAT dataset. It is stored locally
under `archive (1)/EuroSAT/EuroSAT`.

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

## Work remaining before submission

The following tasks still need to be completed:

1. Generate training/validation loss plots from the four saved history CSV files.
2. Run the optional Model A SGD and Model A SGD-with-Momentum experiments if
   a comparison for both custom architectures is required.
3. Create the custom-model comparison table covering parameters, model size,
   time per epoch, and test accuracy.
4. Implement and fine-tune two lightweight pretrained networks, such as
   MobileNetV2 and SqueezeNet.
5. Record SOTA accuracy, precision, recall, parameter count, model size, and
   training time.
6. Write the final accuracy-versus-memory-versus-computation discussion.
7. Add group-member names, index numbers, and the final submission report.

## Important notes

- The dataset split is generated deterministically using seed 42.
- Model B's parameter budget is enforced with an assertion in the model
  constructor.
- Existing output files are currently generated artifacts and should be
  intentionally included in version control if they are required for
  submission.
- No pretrained SOTA model implementation or final report is currently
  included.
