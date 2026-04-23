# Food-101 Fine-Tuning Benchmark

A comparative study of parameter-efficient fine-tuning (PEFT) strategies applied to pretrained CNNs on the [Food-101](https://huggingface.co/datasets/food101) dataset (101 food categories, 75,750 train / 25,250 test images).

## Overview

This project benchmarks four fine-tuning strategies across two backbone architectures:

**Backbones**
- ResNet-18
- EfficientNetV2-S (`tf_efficientnetv2_s`)

**Fine-Tuning Strategies**
- **Linear Probing** — freeze all layers, train the classification head only
- **LoRA** — low-rank adaptation applied to the final linear layer (rank=4, alpha=16)
- **Task-Specific Adapter** — bottleneck adapter (64 units) inserted between the backbone and classifier
- **BatchNorm Tuning** — freeze all parameters except BatchNorm layers and the classifier head

Models are trained for 10 epochs (split across two 5-epoch runs with checkpoint resumption) using Adam (lr=1e-3, batch size=64).

## Evaluation

Beyond the standard clean test set, all models are evaluated on six corrupted test splits to measure robustness:

| Split | Description |
|---|---|
| Clean | Original Food-101 validation set |
| Masked | Partial occlusion applied |
| Noise_Rot | Gaussian noise + rotation |
| Blur_Little | Mild Gaussian blur |
| Blur_Medium | Moderate Gaussian blur |
| Downsampled | Low-resolution downsampling |

Metrics reported: test accuracy, FLOPs (GFLOPs), inference time (ms/image), trainable parameter count.

Grad-CAM visualisations are generated for all ResNet-18 variants to inspect spatial attention across fine-tuning strategies.

## Results Summary (ResNet-18, 10 Epochs)

| Model | Test Acc | Params |
|---|---|---|
| ResNet18 BatchNorm | ~0.674 | 61K |
| ResNet18 Linear Probe | ~0.520 | 52K |
| ResNet18 Adapter | ~0.510 | 118K |
| ResNet18 LoRA | ~0.163 | 2.5K |

## Project Structure

```
Project.ipynb #main notebook
test_splits/ #corrupted test sets
  ├── clean/
  ├── masked/
  ├── noise_rotation/
  ├── blur_little/
  ├── blur_medium/
  └── downsampled/
*.pth #saved model checkpoints (generated during training)
```

## Setup

### Prerequisites

- Python 3.9+
- CUDA GPU recommended (CPU/MPS supported)

### Installation

```bash
pip install -r requirements.txt
```

### Running

Open and run `Project.ipynb` in Jupyter or Google Colab. The notebook is structured into sections:

1. **Setup** — imports, config, seeds
2. **Data** — loads Food-101 from HuggingFace and corrupted test splits from `test_splits/`
3. **Helper Functions** — training loop, evaluation, inference timing, FLOPs calculation
4. **ResNet-18** — four fine-tuning strategies, each with training, checkpoint save/resume, and evaluation
5. **EfficientNetV2-S** — same four strategies applied to a larger backbone
6. **Analysis** — comparison plots (accuracy vs params, accuracy vs inference time, loss curves)
7. **Grad-CAM** — spatial attention visualisations for all ResNet-18 variants
8. **Evaluation on Corrupted Test Sets** — robustness benchmarking across all six splits

### Google Colab Link

https://colab.research.google.com/drive/1ZVFHZFNJq1k4Oru1wIfQlUHQoAjRQKd5?usp=sharing#scrollTo=94Iqb2ZxHarW

## Configuration

Key hyperparameters are defined at the top of the notebook:

```python
NUM_CLASSES = 101
BATCH_SIZE  = 64
NUM_EPOCHS  = 5       #per training run (2 runs = 10 total epochs)
LR          = 1e-3
SEED        = 42
```

## Dependencies

See `requirements.txt` for the full list. Core dependencies: `torch`, `torchvision`, `timm`, `datasets` (HuggingFace), `grad-cam`, `calflops`, `loralib`.
