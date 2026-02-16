# FrFT Loss & Augmentation for ResNet

Experiments applying **Fractional Fourier Transform (FrFT)** to deep learning — explored in two distinct paradigms:

1. **FrFT as a Loss Function** (`frft_resnet18.py`, `frft_resnet2.py`, `frft_resnet3.py`) — FrFT applied to logits at training time to enrich supervision signal, evaluated on the Hymenoptera binary classification task.
2. **FrFT as Data Augmentation** (`train_resnet_FrFTAugment.py`) — FrFT applied to input images during training as a stochastic augmentation, evaluated on CIFAR-10.

---

## Background: Fractional Fourier Transform

The Fractional Fourier Transform generalizes the standard DFT by introducing a rotation order parameter **α**:

- `α = 0` → identity (original signal)
- `α = 1` → standard Fourier Transform
- `α = 0.5` → intermediate time-frequency domain

Applying FrFT at multiple α orders exposes different time-frequency characteristics of a signal, which may carry useful inductive biases for learning.

---

## Experiment 1 — FrFT Loss Function (Hymenoptera)

### Concept

A novel composite loss augments standard Cross-Entropy with additional supervision derived from FrFT-transformed logits:

```
total_loss = CE(logits, targets) + λ · mean_α( CE(|FrFT_α(logits)|, targets) )
```

The FrFT is applied to the logit vector of each sample at multiple fractional orders `α ∈ {0.5, 1.0, 1.5}`, and the magnitude of the complex output is treated as transformed logits.

### Dataset — Hymenoptera

Binary classification: **ants vs. bees** (PyTorch transfer learning tutorial dataset).

| Split | Samples |
|-------|---------|
| Train | ~244    |
| Val   | ~153    |

### Model & Training Configuration

| Parameter        | Value                        |
|-----------------|------------------------------|
| Base model       | ResNet-18 (ImageNet pretrained) |
| Output head      | Linear(512 → 2)              |
| Optimizer        | SGD, lr=0.001, momentum=0.9  |
| LR scheduler     | StepLR (step=7, γ=0.1)       |
| Batch size       | 16                           |
| FrFT α orders    | 0.5, 1.0, 1.5                |
| λ (FrFT weight)  | 0.5                          |

### Scripts

| File              | Description                                      |
|------------------|--------------------------------------------------|
| `frft_resnet18.py` | v1 — FrFT Loss only, auto-downloads dataset     |
| `frft_resnet2.py`  | v2 — FrFT vs. Standard, 10 epochs, loss + acc plots |
| `frft_resnet3.py`  | v3 — FrFT vs. Standard, 30 epochs, accuracy-only plots |

> **Note:** `frft_resnet2.py` and `frft_resnet3.py` expect the dataset at a local path. Update `data_dir` to match your machine.

---

## Experiment 2 — FrFT Data Augmentation (CIFAR-10)

### Concept

FrFT is applied channel-wise to each input image tensor during training with probability 0.5. The fractional orders `α_x` and `α_y` are sampled uniformly from `[0.3, 0.7]` at each call, adding stochastic time-frequency domain perturbations to training images.

The model itself is a standard ResNet-18 trained with ordinary Cross-Entropy loss — the FrFT operates purely at the data level.

### Implementation Details

```
FrFTAugmentation(alpha_range=(0.3, 0.7), prob=0.5)
```

- A 1D FrFT kernel is built via eigendecomposition of the DFT matrix.
- The 2D FrFT is applied separably: first along rows (α_x), then along columns (α_y).
- Only the real part of the output is retained and passed downstream.

### Dataset — CIFAR-10

10-class image classification (32×32 RGB images).

| Split | Samples |
|-------|---------|
| Train | 50,000  |
| Test  | 10,000  |

### Model & Training Configuration

| Parameter       | Value                                      |
|----------------|--------------------------------------------|
| Base model      | ResNet-18 (from scratch)                   |
| Modification    | conv1: 7×7 → 3×3, stride 1; maxpool → Identity |
| Optimizer       | SGD, lr=0.1, momentum=0.9, weight_decay=5e-4 |
| LR scheduler    | CosineAnnealingLR                          |
| Batch size      | 4                                          |
| Epochs          | 20                                         |
| Augmentation    | RandomCrop(32, pad=4) + RandomHorizontalFlip + FrFT |

### Script

| File                        | Description                         |
|----------------------------|-------------------------------------|
| `train_resnet_FrFTAugment.py` | Full training pipeline with FrFT augmentation on CIFAR-10 |

---

## Requirements

```
torch
torchvision
numpy
scipy
matplotlib
```

Install:

```bash
pip install torch torchvision numpy scipy matplotlib
```

---

## Usage

### FrFT Loss experiments

```bash
# v1 — auto-downloads Hymenoptera dataset
python frft_resnet18.py

# v2 — 10-epoch FrFT vs. Standard comparison
python frft_resnet2.py

# v3 — 30-epoch accuracy-only comparison
python frft_resnet3.py
```

### FrFT Augmentation on CIFAR-10

```bash
python train_resnet_FrFTAugment.py
```

CIFAR-10 data will be downloaded automatically to `./data/`. Outputs an `accuracy_plot.png` with training and test accuracy curves.

---

## Project Structure

```
.
├── frft_resnet18.py            # FrFT Loss v1 (Hymenoptera, auto-download)
├── frft_resnet2.py             # FrFT Loss v2 (10-epoch comparison)
├── frft_resnet3.py             # FrFT Loss v3 (30-epoch comparison)
├── train_resnet_FrFTAugment.py # FrFT as data augmentation on CIFAR-10
└── data/
    └── cifar-10-batches-py/    # CIFAR-10 dataset (auto-downloaded)
```

---

## Key Design Choices

| Choice | Rationale |
|--------|-----------|
| Eigendecomposition-based FrFT | Exact fractional power of DFT matrix; numerically stable for small N (logit size or image row/col) |
| Magnitude of complex FrFT output | Produces real-valued scores usable as logits for CE loss |
| Separable 2D FrFT | Applies 1D FrFT independently along each spatial axis; computationally tractable |
| ResNet-18 modifications for CIFAR-10 | Removes aggressive downsampling (large conv, maxpool) to preserve spatial resolution for 32×32 inputs |
