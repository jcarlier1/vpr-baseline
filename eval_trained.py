import argparse
import pickle
import importlib
import torch
import numpy as np
import faiss
from pathlib import Path

from vpr_baseline import load_csv, evaluate, batched_paths_to_embeddings
from train_vpr import TrainableVPREncoder, set_reproducible_training

def _try_allowlist(paths):
    """Try to allowlist globals for safe torch.load.
    Returns True if at least one path was successfully added.
    """
    added_any = False
    add_safe = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe is None:
        return False
    for path in paths:
        try:
            mod_path, attr = path.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            obj = getattr(mod, attr)
            add_safe([obj])
            added_any = True
        except Exception:
            pass
    return added_any

def load_trained_model(model_path, model_name, input_size=224, device="cuda", unsafe_load=False):
    """Load a trained VPR model with safe weights-only loading when supported.
    If safe load fails and `unsafe_load` is True, fall back to unsafe pickle load.
    """
    # Prefer safe weights-only loading
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch: no weights_only kwarg
        print("[warn] Your PyTorch version does not support weights_only; falling back to default torch.load.")
        checkpoint = torch.load(model_path, map_location=device)
    except pickle.UnpicklingError as e:  # type: ignore[name-defined]
        raise  # This except won't trigger because pickle isn't imported; kept for clarity
    except Exception as e:
        msg = str(e)
        # Try allowlisting common NumPy globals used in checkpoints, then retry safe load
        allowlisted = _try_allowlist([
            "numpy.core.multiarray.scalar",
            "numpy._core.multiarray.scalar",
        ])
        if allowlisted:
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            except Exception as e2:
                if unsafe_load:
                    print("[warn] Safe load failed after allowlisting; falling back to UNSAFE torch.load. Only do this if you trust the file.")
                    checkpoint = torch.load(model_path, map_location=device)
                else:
                    raise RuntimeError(
                        "Safe weights-only load failed (even after allowlisting). "
                        "If you trust this checkpoint, rerun with --unsafe_load to allow an unsafe fallback.\n"
                        f"Original error: {e2}"
                    ) from e2
        else:
            if unsafe_load:
                print("[warn] Safe load failed; falling back to UNSAFE torch.load. Only do this if you trust the file.")
                checkpoint = torch.load(model_path, map_location=device)
            else:
                raise RuntimeError(
                    "Safe weights-only load failed and allowlisting isn't available. "
                    "If you trust this checkpoint, rerun with --unsafe_load to allow an unsafe fallback.\n"
                    f"Original error: {e}"
                ) from e
    config = checkpoint['config']
    
    # Create model with same config as training
    model = TrainableVPREncoder(
        model_name,
        input_size=input_size,
        device=device,
        freeze_backbone=False,  # Keep backbone unfrozen as during training for compatibility
        projection_dim=config['projection_dim'],
        dropout=config['dropout']
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded trained model from epoch {checkpoint['epoch']} with R@1: {checkpoint['recall']:.4f}")
    
    return model

def evaluate_trained_model(model, train_csv, test_csv, camera_id, pos_thresh_m=5.0, batch_size=128):
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
    
    # Ensure numpy float32 for FAISS
    def _to_faiss_array(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.ascontiguousarray(x)

    train_np = _to_faiss_array(train_embeddings)
    test_np = _to_faiss_array(test_embeddings)

    # FAISS search (inner product on L2-normalized vectors)
    dim = train_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(train_np)
    D, I = index.search(test_np, 10)
    
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
    ap.add_argument("--pos_m", type=float, default=5.0, help="GPS threshold for positives")
    ap.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--unsafe_load", action="store_true", help="Allow unsafe torch.load fallback if safe load fails")
    args = ap.parse_args()
    
    # Set reproducibility
    set_reproducible_training(args.seed)
    
    # Load trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained_model(args.model_path, args.model, args.input_size, device, unsafe_load=args.unsafe_load)
    
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
