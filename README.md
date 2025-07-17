# DiffSAC: Diffusion-guided Sampling for Consensus-based Robust Estimation

This repository contains the example code of "DiffSAC: Diffusion-guided Sampling for Consensus-based Robust Estimation" for 2D line fitting task. We will release the full code after the paper is accepted.

## 🔥 Overview

DiffSAC leverages diffusion models to perform robust geometric estimation, specifically focusing on line detection from point cloud data. The method uses a transformer-based diffusion model to iteratively refine consensus labels for identifying which points belong to geometric structures.

## 🛠️ Requirements

```bash
pip install torch torchvision numpy scipy scikit-learn wandb tqdm
```

## 📁 Project Structure

```
DiffSAC/
├── checkpoint.py       # Gradient checkpointing utilities
├── dataset.py         # Dataset loading and preprocessing
├── diffusion.py       # Main diffusion model implementation
├── train_line.py      # Training script for line detection
├── transformer.py     # Transformer backbone with timestep embedding
└── utils.py          # Evaluation metrics and utilities
```

## 🚀 Quick Start

### Data Preparation

Your dataset should be organized as follows:
```
Dataset_line/
├── train/
│   └── 06/
│       ├── data/        # .npz files containing points and lines
│       └── images/      # .jpg images (optional)
├── val/
│   └── 06/
│       ├── data/
│       └── images/
```

Each `.npz` file should contain:
- `points`: Array of 2D points with shape (N, 2)
- `lines`: Ground truth lines with shape (M, 4) representing line endpoints

### Training

```bash
python train_line.py \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 100 \
    --timesteps 1000 \
    --train_data ../Dataset_line/train/06 \
    --val_data ../Dataset_line/val/06 \
    --wandb_mode online
```

### Key Arguments

- `--lr`: Learning rate
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--timesteps`: Number of diffusion timesteps
- `--seed`: Random seed for reproducibility
- `--eval_interval`: Evaluation frequency in epochs
- `--checkpoint`: Path to resume training from checkpoint
- `--wandb_mode`: Weights & Biases logging mode

## 🏗️ Architecture

### Diffusion Model
- **Backbone**: Point Diffusion Transformer with multi-head attention
- **Noise Schedule**: Cosine variance schedule for stable training
- **Embedding**: Sinusoidal timestep embeddings

### Key Components
1. **PointDiffusionTransformer**: Transformer-based denoising network
2. **Gradient Checkpointing**: Memory-efficient training
3. **Cosine Noise Schedule**: Improved diffusion process stability
4. **AUC Evaluation**: Area Under Curve metric for line detection accuracy

## 📊 Evaluation

The model is evaluated using AUC (Area Under Curve) metric with a cutoff threshold of 0.5. The evaluation computes the similarity between predicted and ground truth lines using 2-point line representation.

### Metrics
- **AUC@0.5**: Primary evaluation metric

## 📝 Training Details

- **Optimizer**: AdamW with default parameters
- **Loss Function**: MSE loss between predicted and actual noise
- **Data Augmentation**: Random shuffling of point order
- **Mixed Precision**: Supported via gradient checkpointing
- **Logging**: Weights & Biases integration

## 🎯 Key Features

- **Robust Estimation**: Handles outliers and noise in point data
- **Transformer Architecture**: Self-attention mechanism for global context
- **Diffusion Process**: Iterative refinement of consensus labels
- **Memory Efficient**: Gradient checkpointing for large models
- **Reproducible**: Comprehensive seed setting across all libraries
