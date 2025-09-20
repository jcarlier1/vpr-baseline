# VPR Baseline - Trained Model Support

This document describes how to use the updated `vpr_baseline.py` program with trained models.

## Overview

The `vpr_baseline.py` program has been adapted to support loading trained VPR models in addition to the original pretrained models. You can now load custom trained models from `.pth` checkpoint files.

## New Features

### Command Line Arguments

Two new arguments have been added:

- `--trained_model`: Path to a trained model (.pth file). If provided, this trained model will be loaded instead of using pretrained weights.
- `--unsafe_load`: Allow unsafe torch.load fallback if safe loading fails (use only if you trust the checkpoint file).

### Model Types

The program can now work with two types of models:

1. **Pretrained Models** (original functionality): Standard models with pretrained weights from repositories like Hugging Face, timm, etc.
2. **Trained Models** (new functionality): Custom trained VPR models saved as checkpoint files with additional projection layers.

## Usage Examples

### Using Pretrained Models (Original)

```bash
python vpr_baseline.py \
    --train_csv path/to/train.csv \
    --test_csv path/to/test.csv \
    --model resnet50 \
    --camera_id front_left_center
```

### Using Trained Models (New)

```bash
python vpr_baseline.py \
    --train_csv path/to/train.csv \
    --test_csv path/to/test.csv \
    --model resnet50 \
    --trained_model trained_models/trained_resnet50.pth \
    --camera_id front_left_center
```

### Using Unsafe Loading (if needed)

If you encounter safe loading issues and trust the checkpoint file:

```bash
python vpr_baseline.py \
    --train_csv path/to/train.csv \
    --test_csv path/to/test.csv \
    --model resnet50 \
    --trained_model trained_models/trained_resnet50.pth \
    --unsafe_load \
    --camera_id front_left_center
```

## Available Trained Models

In the `trained_models/` directory:
- `trained_resnet50.pth` - Trained ResNet50 model (epoch 14, R@1: 0.4507)
- `trained_dinov2_b_1full.pth` - Trained DINOv2 base model

## Technical Details

### Safe Loading

The program attempts to load checkpoints safely using PyTorch's `weights_only=True` parameter to prevent arbitrary code execution. If safe loading fails, you can use the `--unsafe_load` flag as a fallback.

### Model Architecture

Trained models use the `TrainableVPREncoder` class which extends the base `Encoder` with:
- Additional projection layers for VPR-specific adaptation
- Configurable projection dimensions and dropout
- Support for both frozen and unfrozen backbone training

### Backward Compatibility

The program maintains full backward compatibility. All existing functionality with pretrained models continues to work exactly as before.

## Testing

The adaptation has been tested with:
- ✓ Argument parsing with new options
- ✓ Pretrained model creation (ResNet50)
- ✓ Trained model loading (ResNet50 checkpoint)
- ✓ Model type verification and projection dimensions

## Example Output

When loading a trained model, you'll see output like:

```
Loading trained model from: trained_models/trained_resnet50.pth
[warn] Safe load failed after allowlisting; falling back to UNSAFE torch.load. Only do this if you trust the file.
Loaded trained model from epoch 14 with R@1: 0.4507
```

This confirms the model was loaded successfully and shows the training statistics from the checkpoint.