import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch, faiss, timm
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoProcessor, AutoImageProcessor, AutoModel

# ---------- geo helpers ----------
def haversine_np(lat1, lon1, lat2, lon2):
    # all args are vectors or scalars in degrees
    R = 6371000.0  # meters
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon * 0.5) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c  # meters

# ---------- encoders ----------
class Encoder(nn.Module):
    def __init__(self, name, input_size=224, device="cuda"):
        super().__init__()
        self.name = name
        self.device = device
        self.input_size = input_size
        if name in ["clip_b32", "clip_l14", "siglip_b16"]:
            model_id = {
                "clip_b32": "openai/clip-vit-base-patch32",
                "clip_l14": "openai/clip-vit-large-patch14",
                "siglip_b16": "google/siglip-base-patch16-224",
            }[name]
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(device).eval()
            self.kind = "transformers_clip"
        elif name == "dinov2_b":
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
            self.kind = "transformers_dino"
        elif name == "convnext_b":
            self.model = timm.create_model(
                "convnext_base.fb_in22k_ft_in1k", pretrained=True,
                num_classes=0, global_pool="avg"
            ).to(device).eval()
            self.timm_tf = timm.data.create_transform(self.input_size, is_training=False)
            self.kind = "timm"
        elif name == "resnet50":
            self.model = timm.create_model(
                "resnet50.a1_in1k", pretrained=True,
                num_classes=0, global_pool="avg"
            ).to(device).eval()
            self.timm_tf = timm.data.create_transform(self.input_size, is_training=False)
            self.kind = "timm"
        else:
            raise ValueError(f"Unknown model name: {name}")

    @torch.no_grad()
    def encode_batch(self, pil_batch):
        if self.kind == "transformers_clip":
            inputs = self.processor(images=pil_batch, return_tensors="pt").to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = self.model.get_image_features(**inputs)
            return F.normalize(feats.float(), dim=-1)
        elif self.kind == "transformers_dino":
            inputs = self.processor(images=pil_batch, return_tensors="pt")["pixel_values"].to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(inputs).last_hidden_state[:, 0]  # CLS token
            return F.normalize(out.float(), dim=-1)
        else:  # timm
            x = torch.stack([self.timm_tf(img) for img in pil_batch]).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(x)
            return F.normalize(out.float(), dim=-1)

# ---------- io ----------
def load_csv(csv_path, camera_id=None):
    df = pd.read_csv(csv_path)
    if camera_id is not None:
        df = df[df["camera_id"] == camera_id].copy()
    base = Path(csv_path).parent.parent  # .../sequence/poses/poses.csv -> sequence/
    df["abs_path"] = df["img_relpath"].apply(lambda p: str((base / p).resolve()))
    return df.reset_index(drop=True)

def batched_paths_to_embeddings(paths, enc: Encoder, batch_size):
    embs = []
    n = len(paths)
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch_paths = paths[i:i+batch_size]
        pil_batch = [Image.open(p).convert("RGB") for p in batch_paths]
        embs.append(enc.encode_batch(pil_batch))
    embs = torch.cat(embs, dim=0)  # Keep on GPU as tensor
    dt = time.time() - t0
    return embs, dt / n  # seconds per image

# ---------- eval ----------
def evaluate(train_lat, train_lon, test_lat, test_lon, topk_idx, pos_thresh_m):
    # filter queries with at least one positive within threshold
    valid = []
    r1 = r5 = r10 = 0
    for q in range(len(test_lat)):
        d_all = haversine_np(test_lat[q], test_lon[q], train_lat, train_lon)
        min_d = d_all.min()
        if min_d > pos_thresh_m:
            valid.append(False)
            continue
        valid.append(True)
        k_idx = topk_idx[q]
        d_k = d_all[k_idx]
        r1 += (d_k[0] <= pos_thresh_m)
        r5 += (d_k[:5] <= pos_thresh_m).any()
        r10 += (d_k[:10] <= pos_thresh_m).any()
    valid = np.array(valid)
    n_valid = valid.sum()
    if n_valid == 0:
        return {"valid": 0, "R@1": 0.0, "R@5": 0.0, "R@10": 0.0}
    return {
        "valid": int(n_valid),
        "R@1": r1 / n_valid,
        "R@5": r5 / n_valid,
        "R@10": r10 / n_valid,
    }

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--camera_id", default="front_left_center")
    ap.add_argument("--model", required=True,
                    choices=["clip_b32", "clip_l14", "siglip_b16", "dinov2_b", "convnext_b", "resnet50"])
    ap.add_argument("--input", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--pos_m", type=float, default=10.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--save_npz", type=str, default="")
    args = ap.parse_args()

    print(f"Loading CSVs and filtering camera_id={args.camera_id}")
    df_tr = load_csv(args.train_csv, args.camera_id)
    df_te = load_csv(args.test_csv, args.camera_id)
    print(f"Train images: {len(df_tr)}  Test images: {len(df_te)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Encoder(args.model, input_size=args.input, device=device)

    print("Embedding train...")
    xb, sec_per_img_train = batched_paths_to_embeddings(df_tr["abs_path"].tolist(), enc, args.batch)
    print("Embedding test...")
    xq, sec_per_img_test  = batched_paths_to_embeddings(df_te["abs_path"].tolist(), enc, args.batch)

    # FAISS index (cosine via inner product on L2-normalized vectors)
    dim = xb.shape[1]
    res = faiss.StandardGpuResources()  # Add GPU resources
    index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_gpu(res, 0, index)  # Move to GPU
    # Convert GPU tensors to numpy float32 for FAISS (only conversion needed)
    xb_numpy = xb.cpu().numpy().astype("float32")
    xq_numpy = xq.cpu().numpy().astype("float32")
    index.add(xb_numpy)
    D, I = index.search(xq_numpy, args.k)


    # Evaluation by GPS distance threshold
    metrics = evaluate(
        df_tr["lat"].to_numpy(), df_tr["lon"].to_numpy(),
        df_te["lat"].to_numpy(), df_te["lon"].to_numpy(),
        I, args.pos_m
    )
    print("\n=== Results ===")
    print(f"Model: {args.model}  input: {args.input}px")
    print(f"Queries with positives (<= {args.pos_m} m): {metrics['valid']}/{len(df_te)}")
    print(f"Recall@1:  {metrics['R@1']:.3f}")
    print(f"Recall@5:  {metrics['R@5']:.3f}")
    print(f"Recall@10: {metrics['R@10']:.3f}")
    print(f"Embed speed: train {sec_per_img_train:.4f}s/img, test {sec_per_img_test:.4f}s/img")

    if args.save_npz:
        np.savez_compressed(
            args.save_npz,
            xb=xb.cpu().numpy().astype("float16"),
            xq=xq.cpu().numpy().astype("float16"),
            train_lat=df_tr["lat"].to_numpy().astype("float32"),
            train_lon=df_tr["lon"].to_numpy().astype("float32"),
            test_lat=df_te["lat"].to_numpy().astype("float32"),
            test_lon=df_te["lon"].to_numpy().astype("float32"),
            topk=I, sims=D.astype("float32"),
            meta=dict(model=args.model, input=args.input, pos_m=args.pos_m, k=args.k)
        )
        print(f"Saved embeddings and search results to {args.save_npz}.npz")

if __name__ == "__main__":
    main()
