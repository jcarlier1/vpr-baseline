import argparse
import torch
import numpy as np
import faiss
from pathlib import Path

from vpr_baseline import load_csv, evaluate, batched_paths_to_embeddings
from train_vpr import TrainableVPREncoder, set_reproducible_training

def load_trained_model(model_path, model_name, input_size=224, device="cuda"):
    """Load a trained VPR model"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Create model with same config as training
    model = TrainableVPREncoder(
        model_name,
        input_size=input_size,
        device=device,
        freeze_backbone=True,  # This was used during training
        projection_dim=config['projection_dim'],
        dropout=config['dropout']
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded trained model from epoch {checkpoint['epoch']} with R@1: {checkpoint['recall']:.4f}")
    
    return model

def evaluate_trained_model(model, train_csv, test_csv, camera_id, pos_thresh_m=10.0, batch_size=128):
    """Evaluate trained model using the same protocol as baseline"""
    
    # Load data
    df_train = load_csv(train_csv, camera_id)
    df_test = load_csv(test_csv, camera_id)
    
    print(f"Train images: {len(df_train)}, Test images: {len(df_test)}")
    
    # Extract embeddings
    print("Embedding train set...")
    train_embeddings, train_time = batched_paths_to_embeddings(
        df_train["abs_path"].tolist(), model, batch_size
    )
    
    print("Embedding test set...")
    test_embeddings, test_time = batched_paths_to_embeddings(
        df_test["abs_path"].tolist(), model, batch_size
    )
    
    # FAISS search
    dim = train_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(train_embeddings.astype("float32"))
    D, I = index.search(test_embeddings.astype("float32"), 10)
    
    # Evaluate
    metrics = evaluate(
        df_train["lat"].to_numpy(), df_train["lon"].to_numpy(),
        df_test["lat"].to_numpy(), df_test["lon"].to_numpy(),
        I, pos_thresh_m
    )
    
    return metrics, train_time, test_time

def main():
    ap = argparse.ArgumentParser(description='Evaluate trained VPR model')
    ap.add_argument("--model_path", required=True, help="Path to trained model")
    ap.add_argument("--train_csv", required=True, help="Training CSV for database")
    ap.add_argument("--test_csv", required=True, help="Test CSV for queries")
    ap.add_argument("--camera_id", default="front_left_center", help="Camera ID")
    ap.add_argument("--model", required=True,
                    choices=["clip_b32", "clip_l14", "siglip_b16", "dinov2_b", "convnext_b", "resnet50"],
                    help="Model name")
    ap.add_argument("--input_size", type=int, default=224, help="Input size")
    ap.add_argument("--pos_m", type=float, default=10.0, help="GPS threshold for positives")
    ap.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()
    
    # Set reproducibility
    set_reproducible_training(args.seed)
    
    # Load trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained_model(args.model_path, args.model, args.input_size, device)
    
    # Evaluate
    metrics, train_time, test_time = evaluate_trained_model(
        model, args.train_csv, args.test_csv, args.camera_id, 
        args.pos_m, args.batch_size
    )
    
    # Print results
    print("\n" + "="*50)
    print("TRAINED MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {args.model} (trained)")
    print(f"Input size: {args.input_size}px")
    print(f"GPS threshold: {args.pos_m}m")
    print(f"Queries with positives: {metrics['valid']}")
    print(f"Recall@1:  {metrics['R@1']:.4f}")
    print(f"Recall@5:  {metrics['R@5']:.4f}")
    print(f"Recall@10: {metrics['R@10']:.4f}")
    print(f"Embed speed: train {train_time:.4f}s/img, test {test_time:.4f}s/img")
    print("="*50)

if __name__ == "__main__":
    main()
