# CNN for Image Classification — EN3150 Assignment 03

Resource-constrained CNN for edge image classification, using the **EuroSAT** dataset
(`archive (1)/EuroSAT` — RGB, 64×64 already, 10 land-use classes).

## What we have to build

### 1. Data pipeline
- Load EuroSAT RGB images (already ≤64×64, so minimal/no downscaling needed).
- Split **70% train / 15% validation / 15% test**.
- Normalize pixel values; keep class balance in mind across the 10 classes
  (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture,
  PermanentCrop, Residential, River, SeaLake).
- Log dataset details (size, classes, split counts) in the assignment's shared sheet.

### 2. Two custom CNN architectures (20 marks)
Both follow the standard CNN pipeline from the lecture:
`[Conv + ReLU] → Pooling → [Conv + ReLU] → Pooling → Flatten → FC → Softmax`

- **Model A — Standard CNN**
  Interleaved `Conv2D` + `MaxPooling2D` layers (regular 3×3 convolutions,
  `C_out × C_in × K × K` weights per layer — see lecture's parameter-count slides).

- **Model B — Lightweight CNN (Depthwise Separable Convolutions)**
  Replace standard convs with **depthwise separable convolutions**
  (depthwise `F×F` conv per channel, `F²·K·H·W` ops, followed by a `1×1`
  pointwise conv to mix channels, `K²·H·W` ops) — this is the MobileNets trick
  from the lecture, giving roughly an 8–9× reduction in multiply-accumulates
  vs. a regular 3×3 conv.
  **Hard constraint: total trainable parameters ≤ 100,000.**

For both models:
- Explicitly list kernel sizes, filter counts, FC units.
- Compute total trainable parameters by hand/formula (not just `model.summary()`).
- Justify activation function choice on hardware-efficiency grounds (e.g. ReLU
  over sigmoid/tanh — no expensive exponentials, cheap to compute on
  microcontrollers/edge devices).

### 3. Optimizer selection & tuning (15 marks)
- Pick an optimizer (e.g. Adam) + learning rate, justify the choice.
- Compare against **plain SGD** and **SGD + Momentum**.
- Explain how the momentum term affects convergence speed/stability.

### 4. Training & evaluation of Model A / Model B (25 marks)
- Train both ≥20 epochs; plot training vs. validation loss curves.
- Test-set metrics: accuracy, confusion matrix, precision, recall.
- Comparison table: parameter count, disk size (KB), time/epoch, test accuracy.
- Discuss the standard-conv vs. depthwise-separable trade-offs (accuracy vs.
  compute/size).

### 5. Fine-tune two pretrained lightweight SOTA models (20 marks)
Choose two of: MobileNetV1/V2, SqueezeNet, EfficientNet-B0, ShuffleNet.
- Fine-tune on the *same* data splits (transfer learning: freeze backbone or
  fine-tune, replace/train the final classification layer — see lecture's
  "Strategy 1: feature extractor" vs. "Strategy 2: transfer learning" slides).
- Report test accuracy, precision, recall, total params, model size (MB).

### 6. Final comparison & trade-off discussion (20 marks)
Compare custom Model B vs. the two fine-tuned SOTA models — accuracy vs.
memory footprint vs. computational cost trade-offs for edge deployment.

## Key CNN concepts to apply (from lecture slides)
- Convolution arithmetic: output size `W − K + 1 + 2P`; `P = (K−1)/2` keeps
  spatial size unchanged.
- Multiply count for a regular conv layer: `C_out × C_in × K² × H × W`.
- 1×1 convolutions for channel mixing / dimension reduction (bottleneck idea).
- Depthwise separable convolution = depthwise conv (`F²·K·H·W`) + pointwise
  1×1 conv (`K²·H·W`) — far cheaper than a full 3D conv (`F²·K²·H·W`).
- Max pooling for downsampling + translation robustness.
- Receptive field grows with depth/stride; deeper stacks of small kernels
  approximate large receptive fields cheaply.

## Deliverables / submission checklist
- [ ] Report + code uploaded **separately** to Moodle as `Yourgroupno_A03_EN3150`.
- [ ] Code complete, commented, runnable.
- [ ] Group members' names + index numbers in the report.
- [ ] GitHub repo with regular commits (link included in report).
- [ ] Interpretation/discussion sections are as important as the numbers.

Reference: `EN3150_Assignment_03_In23.pdf` (assignment spec) and
`../EN3150 Pattern Recognition_L07.pdf` (CNN lecture slides).
