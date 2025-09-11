# VPR Training Scripts

This directory contains training scripts for Visual Place Recognition (VPR) models using hybrid triplet loss and contrastive learning approaches.

## Overview

The training approach implements:

1. **Frozen Backbone Training**: Only train a projection head while keeping pretrained features frozen
2. **Hybrid Loss**: Combines triplet loss and contrastive learning  
3. **Hard Negative Mining**: Smart sampling of difficult negative examples
4. **GPS-based Supervision**: Uses GPS coordinates to define positive/negative relationships
5. **Reproducibility**: Fixed seeds and standardized hyperparameters across models
6. **Comprehensive Logging**: Automatic metric logging to CSV files for analysis

## Files

- `train_vpr.py` - Main training script with all loss functions and data mining
- `eval_trained.py` - Evaluation script for trained models  
- `run_experiment.py` - Complete pipeline script (baseline → training → evaluation)
- `analyze_metrics.py` - Script to analyze and plot training metrics
- `trainable_vpr.py` - Extended encoder classes with projection heads

## Directory Structure

After training, the following directories will be created:

```
vpr-baseline/
├── logs/                          # Training metrics (CSV files)
│   ├── dinov2_b_seed42_20240911_143022.csv
│   ├── convnext_b_seed42_20240911_145033.csv
│   └── training_summary.csv       # Automatic summary of all runs
├── trained_models/                # Saved model weights
│   ├── trained_dinov2_b_seed42.pth
│   ├── trained_convnext_b_seed42.pth
│   └── ...
└── plots/                         # Generated analysis plots
    ├── dinov2_b_seed42_20240911_143022_plots.png
    └── ...
```

## Usage

### Basic Training

```bash
# Train a single model
python train_vpr.py --train_csv train_data.csv --val_csv val_data.csv --model dinov2_b --epochs 10

# Train with custom hyperparameters
python train_vpr.py --train_csv train_data.csv --val_csv val_data.csv \
    --model convnext_b --epochs 15 --lr 0.0005 --batch_size 64 \
    --triplet_margin 0.2 --weight_decay 0.001
```

### Complete Experiment Pipeline

```bash
# Run full baseline → training → evaluation
python run_experiment.py --train_csv train.csv --val_csv val.csv --test_csv test.csv --model dinov2_b
```

### Analyze Training Results

```bash
# Analyze metrics from a specific training run
python analyze_metrics.py logs/dinov2_b_seed42_20240911_143022.csv

# Analyze all training runs in logs/ directory
python analyze_metrics.py logs/
```

## Logged Metrics

The training script automatically logs detailed metrics to CSV files:

### Batch-level Metrics
- `batch`: Batch number
- `epoch`: Current epoch
- `train_loss`, `triplet_loss`, `contrastive_loss`: Loss components
- `batch_time`: Time to process the batch

### Epoch Summary Metrics  
- `epoch`: Epoch number
- `avg_train_loss`, `avg_triplet_loss`, `avg_contrastive_loss`: Average losses
- `epoch_time`: Total epoch time

### Validation Metrics
- `val_loss`: Validation loss
- `val_r1`, `val_r5`, `val_r10`: Recall @ 1, 5, 10
- `val_time`: Validation time

All metrics include timestamps and event types for easy analysis and plotting.
```## Key Features

### Reproducibility Strategy
- Fixed random seeds (default: 42)
- Deterministic CUDA operations
- Standardized hyperparameters across all models
- Same training protocol for fair comparison

### Loss Functions
- **Triplet Loss**: `max(0, d(anchor,pos) - d(anchor,neg) + margin)`
- **Multi-Positive Contrastive**: InfoNCE with multiple positives per anchor
- **Hybrid Loss**: Weighted combination of both losses

### Data Mining
- **GPS Clustering**: Creates labels based on geographic proximity (25m threshold)
- **Hard Negative Mining**: Finds embeddings that are similar but geographically distant
- **Online Triplet Mining**: Mines triplets during training with hard negative focus

### Architecture
- **Frozen Backbone**: Pretrained weights frozen, only train projection head
- **Projection Head**: `[backbone_dim → backbone_dim//2 → 256]` with BatchNorm
- **Normalization**: L2 normalize final embeddings for cosine similarity

## Usage

### 1. Single Model Training

```bash
python train_vpr.py \
    --train_csv /path/to/train/poses.csv \
    --val_csv /path/to/val/poses.csv \
    --camera_id front_left_center \
    --model dinov2_b \
    --epochs 15 \
    --batch_size 32 \
    --seed 42
```

### 2. Complete Experiment Pipeline

```bash
python run_experiment.py \
    --train_csv /path/to/train/poses.csv \
    --test_csv /path/to/test/poses.csv \
    --models dinov2_b convnext_b resnet50 \
    --epochs 15 \
    --seed 42
```

This will:
1. Run baseline evaluation for each model
2. Train each model with hybrid loss
3. Evaluate trained models
4. Show comparison summary

### 3. Evaluate Existing Trained Model

```bash
python eval_trained.py \
    --model_path trained_dinov2_b_seed42.pth \
    --train_csv /path/to/train/poses.csv \
    --test_csv /path/to/test/poses.csv \
    --model dinov2_b
```

## Training Configuration

All models use standardized settings for fair comparison:

```python
config = {
    'epochs': 15,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'pos_threshold_m': 25.0,  # GPS threshold for positives
    'neg_threshold_m': 100.0,  # GPS threshold for hard negatives
    'triplet_margin': 0.5,
    'contrastive_temperature': 0.07,
    'projection_dim': 256,
    'dropout': 0.1,
    'weight_decay': 1e-4,
    'triplet_weight': 0.3,
    'contrastive_weight': 1.0,
    'hard_neg_ratio': 0.5
}
```

## Expected Improvements

Based on the training methodology, expected improvements over baseline:

- **DINOv2**: Baseline R@1 ~0.21 → Expected ~0.35-0.45
- **ConvNeXt**: Baseline R@1 ~0.20 → Expected ~0.30-0.40  
- **ResNet50**: Baseline R@1 ~0.16 → Expected ~0.25-0.35

The improvement comes from:
1. **Geographic supervision**: GPS coordinates provide natural training signal
2. **Hard negative mining**: Forces model to distinguish difficult cases
3. **Multiple positives**: Contrastive loss handles multiple similar places
4. **Preserved pretrained features**: Frozen backbone retains general vision capabilities

## Model Outputs

Trained models are saved with:
- Model state dict
- Training configuration  
- Best validation recall
- Training epoch info

## Memory and Speed

Training is memory-efficient due to frozen backbones:
- **Memory**: ~70-80% reduction vs full fine-tuning
- **Speed**: ~5-10x faster training
- **Parameters**: Only ~1-5% of total parameters are trained

## Troubleshooting

Common issues:

1. **CUDA OOM**: Reduce batch size to 16 or 8
2. **No positives found**: Check GPS data quality and pos_threshold_m
3. **Import errors**: Ensure all baseline dependencies are installed
4. **Slow training**: Check if CUDA is available and being used

## Next Steps

After training:

1. Compare baseline vs trained recall metrics
2. Analyze which models benefit most from training
3. Experiment with different GPS thresholds
4. Try progressive unfreezing of backbone layers
5. Test on different datasets/camera views
