#!/usr/bin/env python3
"""
Training and comparison script for VPR models
Runs baseline evaluation, training, and comparison
"""

import subprocess
import argparse
import pandas as pd
from pathlib import Path

def run_baseline_evaluation(model, train_csv, test_csv, camera_id="front_left_center"):
    """Run baseline model evaluation"""
    print(f"\n{'='*60}")
    print(f"RUNNING BASELINE EVALUATION: {model}")
    print(f"{'='*60}")
    
    cmd = [
        "python", "vpr_baseline.py",
        "--train_csv", train_csv,
        "--test_csv", test_csv, 
        "--camera_id", camera_id,
        "--model", model,
        "--input", "224",
        "--batch", "128",
        "--pos_m", "10.0",
        "--k", "10"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0

def run_training(model, train_csv, val_csv, camera_id="front_left_center", epochs=15, seed=42):
    """Run model training"""
    print(f"\n{'='*60}")
    print(f"RUNNING TRAINING: {model}")
    print(f"{'='*60}")
    
    save_path = f"trained_{model}_seed{seed}.pth"
    
    cmd = [
        "python", "train_vpr.py",
        "--train_csv", train_csv,
        "--val_csv", val_csv,
        "--camera_id", camera_id,
        "--model", model,
        "--input_size", "224",
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--batch_size", "32",
        "--lr", "1e-4",
        "--freeze_backbone",
        "--save_path", save_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    success = result.returncode == 0 and Path(save_path).exists()
    return success, save_path

def run_trained_evaluation(model, model_path, train_csv, test_csv, camera_id="front_left_center", seed=42):
    """Run trained model evaluation"""
    print(f"\n{'='*60}")
    print(f"RUNNING TRAINED MODEL EVALUATION: {model}")
    print(f"{'='*60}")
    
    cmd = [
        "python", "eval_trained.py",
        "--model_path", model_path,
        "--train_csv", train_csv,
        "--test_csv", test_csv,
        "--camera_id", camera_id,
        "--model", model,
        "--input_size", "224",
        "--pos_m", "10.0",
        "--batch_size", "128",
        "--seed", str(seed)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    ap = argparse.ArgumentParser(description='Complete VPR training and evaluation pipeline')
    ap.add_argument("--train_csv", required=True, help="Training CSV file")
    ap.add_argument("--test_csv", required=True, help="Test CSV file")
    ap.add_argument("--val_csv", help="Validation CSV file (default: use test_csv)")
    ap.add_argument("--camera_id", default="front_left_center", help="Camera ID")
    ap.add_argument("--models", nargs='+', 
                    default=["dinov2_b", "convnext_b", "resnet50"],
                    choices=["clip_b32", "clip_l14", "siglip_b16", "dinov2_b", "convnext_b", "resnet50"],
                    help="Models to train and evaluate")
    ap.add_argument("--epochs", type=int, default=15, help="Training epochs")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--skip_baseline", action="store_true", help="Skip baseline evaluation")
    ap.add_argument("--skip_training", action="store_true", help="Skip training (evaluate existing models)")
    ap.add_argument("--skip_eval", action="store_true", help="Skip evaluation of trained models")
    args = ap.parse_args()
    
    val_csv = args.val_csv if args.val_csv else args.test_csv
    
    print(f"Pipeline configuration:")
    print(f"  Models: {args.models}")
    print(f"  Training data: {args.train_csv}")
    print(f"  Test data: {args.test_csv}")
    print(f"  Validation data: {val_csv}")
    print(f"  Camera: {args.camera_id}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Seed: {args.seed}")
    
    results = {}
    
    for model in args.models:
        print(f"\n{'#'*80}")
        print(f"PROCESSING MODEL: {model.upper()}")
        print(f"{'#'*80}")
        
        model_results = {
            'model': model,
            'baseline_success': False,
            'training_success': False,
            'evaluation_success': False,
            'model_path': None
        }
        
        # 1. Baseline evaluation
        if not args.skip_baseline:
            print(f"\n--- Step 1/3: Baseline Evaluation ---")
            model_results['baseline_success'] = run_baseline_evaluation(
                model, args.train_csv, args.test_csv, args.camera_id
            )
        else:
            print("Skipping baseline evaluation")
            model_results['baseline_success'] = True
        
        # 2. Training
        if not args.skip_training:
            print(f"\n--- Step 2/3: Training ---")
            training_success, model_path = run_training(
                model, args.train_csv, val_csv, args.camera_id, args.epochs, args.seed
            )
            model_results['training_success'] = training_success
            model_results['model_path'] = model_path
        else:
            print("Skipping training")
            model_path = f"trained_{model}_seed{args.seed}.pth"
            model_results['training_success'] = Path(model_path).exists()
            model_results['model_path'] = model_path
        
        # 3. Trained model evaluation
        if not args.skip_eval and model_results['training_success'] and model_results['model_path']:
            print(f"\n--- Step 3/3: Trained Model Evaluation ---")
            model_results['evaluation_success'] = run_trained_evaluation(
                model, model_results['model_path'], args.train_csv, args.test_csv, 
                args.camera_id, args.seed
            )
        else:
            if args.skip_eval:
                print("Skipping trained model evaluation")
            else:
                print("Cannot evaluate trained model (training failed or model not found)")
        
        results[model] = model_results
    
    # Summary
    print(f"\n{'#'*80}")
    print("PIPELINE SUMMARY")
    print(f"{'#'*80}")
    
    for model, result in results.items():
        print(f"\n{model.upper()}:")
        print(f"  Baseline evaluation: {'✓' if result['baseline_success'] else '✗'}")
        print(f"  Training: {'✓' if result['training_success'] else '✗'}")
        print(f"  Trained evaluation: {'✓' if result['evaluation_success'] else '✗'}")
        if result['model_path']:
            print(f"  Model saved: {result['model_path']}")
    
    # Check if all completed successfully
    all_success = all(
        r['baseline_success'] and r['training_success'] and r['evaluation_success'] 
        for r in results.values()
    )
    
    if all_success:
        print(f"\n🎉 All models completed successfully!")
    else:
        print(f"\n⚠️  Some models had issues. Check the logs above.")
        
    print(f"\nNext steps:")
    print(f"  1. Compare baseline vs trained results")
    print(f"  2. Analyze which models improved most")
    print(f"  3. Experiment with different hyperparameters")

if __name__ == "__main__":
    main()
