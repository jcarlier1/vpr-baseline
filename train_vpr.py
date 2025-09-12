import argparse, time, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch, faiss, timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
import os
from datetime import datetime

from transformers import AutoProcessor, AutoImageProcessor, AutoModel
from vpr_baseline import Encoder, haversine_np, load_csv, evaluate

# ========== METRIC LOGGING ==========
class MetricLogger:
    """Logs training metrics to CSV file for later plotting"""
    def __init__(self, log_dir="logs", model_name="vpr_model", seed=42):
        self.log_dir = log_dir
        self.model_name = model_name
        self.seed = seed
        os.makedirs(log_dir, exist_ok=True)
        
        # Create unique log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{model_name}_seed{seed}_{timestamp}.csv")
        
        # Initialize CSV file with headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'epoch', 'batch', 'total_loss', 'contrastive_loss', 
                'triplet_loss', 'learning_rate', 'validation_recall', 'event_type'
            ])
        
        print(f"Logging metrics to: {self.log_file}")
    
    def log_batch(self, epoch, batch_idx, total_loss, cont_loss, trip_loss, lr):
        """Log per-batch metrics"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, epoch, batch_idx, 
                f"{total_loss:.6f}", f"{cont_loss:.6f}", f"{trip_loss:.6f}",
                f"{lr:.6f}", "", "batch"
            ])
    
    def log_epoch(self, epoch, total_loss, cont_loss, trip_loss, lr):
        """Log epoch summary metrics"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, epoch, "summary", 
                f"{total_loss:.6f}", f"{cont_loss:.6f}", f"{trip_loss:.6f}",
                f"{lr:.6f}", "", "epoch_summary"
            ])
    
    def log_validation(self, epoch, recall):
        """Log validation metrics"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, epoch, "validation", "", "", "", "", f"{recall:.6f}", "validation"
            ])
    
    def log_final_summary(self, best_recall, total_epochs, training_time):
        """Log final training summary"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, total_epochs-1, "final", "", "", "", "", 
                f"{best_recall:.6f}", f"training_complete_time_{training_time:.1f}s"
            ])

# ========== REPRODUCIBILITY ==========
def set_reproducible_training(seed=42):
    """Set seeds for reproducible training across all models"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Set reproducible training with seed: {seed}")

def get_standardized_training_config():
    """Standardized training settings across all models"""
    return {
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
        'hard_neg_ratio': 0.5  # ratio of hard negatives vs random
    }

# ========== TRAINABLE ENCODER ==========
class TrainableVPREncoder(Encoder):
    def __init__(self, name, input_size=224, device="cuda", freeze_backbone=True, projection_dim=256, dropout=0.1):
        super().__init__(name, input_size, device)
        
        # Get backbone feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Create projection head for VPR adaptation
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim // 2, projection_dim),
            nn.BatchNorm1d(projection_dim)
        ).to(device)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        self.projection_dim = projection_dim
        
    def _get_feature_dim(self):
        """Determine the feature dimension of the backbone"""
        with torch.no_grad():
            if self.kind == "transformers_clip":
                dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
                dummy_pil = [F.to_pil_image(dummy_input[0].cpu())]
                inputs = self.processor(images=dummy_pil, return_tensors="pt").to(self.device)
                feats = self.model.get_image_features(**inputs)
                return feats.shape[1]
            elif self.kind == "transformers_dino":
                return self.model.config.hidden_size
            else:  # timm
                dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
                feats = self.model(dummy_input)
                return feats.shape[1]
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters"""
        frozen_params = 0
        for param in self.model.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            
        trainable_params = sum(p.numel() for p in self.projection.parameters())
        total_params = frozen_params + trainable_params
        
        print(f"Frozen backbone parameters: {frozen_params:,}")
        print(f"Trainable projection parameters: {trainable_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params:.1%}")
    
    def get_backbone_features(self, pil_batch):
        """Extract features from backbone"""
        if self.kind == "transformers_clip":
            inputs = self.processor(images=pil_batch, return_tensors="pt").to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = self.model.get_image_features(**inputs)
        elif self.kind == "transformers_dino":
            inputs = self.processor(images=pil_batch, return_tensors="pt")["pixel_values"].to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = self.model(inputs).last_hidden_state[:, 0]  # CLS token
        else:  # timm
            x = torch.stack([self.timm_tf(img) for img in pil_batch]).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = self.model(x)
        return feats.float()
    
    def forward(self, pil_batch):
        """Forward pass through backbone + projection"""
        backbone_feats = self.get_backbone_features(pil_batch)
        projected_feats = self.projection(backbone_feats)
        return F.normalize(projected_feats, dim=-1)
    
    def encode_batch(self, pil_batch):
        """For inference"""
        if self.training:
            return self.forward(pil_batch)
        else:
            with torch.no_grad():
                projected = self.forward(pil_batch)
                return projected.cpu().numpy()

# ========== LOSS FUNCTIONS ==========
class TripletLoss(nn.Module):
    """Classic triplet loss with margin"""
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

class MultiPositiveContrastiveLoss(nn.Module):
    """Contrastive learning with multiple positives per anchor"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        
        # Ensure labels are on the same device as embeddings
        labels = labels.to(embeddings.device)
        
        # Cosine similarity matrix
        sim_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1), 
            embeddings.unsqueeze(0), 
            dim=2
        ) / self.temperature
        
        # Create positive mask based on labels
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        positive_mask.fill_diagonal_(0)  # Remove self-similarity
        
        # InfoNCE with multiple positives
        exp_sim = torch.exp(sim_matrix)
        pos_sum = (exp_sim * positive_mask).sum(dim=1)
        total_sum = exp_sim.sum(dim=1)
        
        # Avoid log(0)
        eps = 1e-8
        loss = -torch.log(pos_sum / (total_sum + eps) + eps)
        
        # Only compute loss for samples that have positives
        valid_mask = positive_mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            return loss[valid_mask].mean()
        else:
            # Return zero tensor that requires gradients and is connected to computation graph
            return torch.zeros(1, device=embeddings.device, requires_grad=True).squeeze()

class HybridVPRLoss(nn.Module):
    """Combined triplet and contrastive loss"""
    def __init__(self, temperature=0.07, margin=0.5, contrastive_weight=1.0, triplet_weight=0.3):
        super().__init__()
        self.contrastive = MultiPositiveContrastiveLoss(temperature)
        self.triplet = TripletLoss(margin)
        self.cont_weight = contrastive_weight
        self.trip_weight = triplet_weight
        
    def forward(self, embeddings, labels, triplets=None):
        # Always compute contrastive loss
        cont_loss = self.contrastive(embeddings, labels)
        total_loss = self.cont_weight * cont_loss
        
        # Initialize trip_loss with gradients enabled
        trip_loss = torch.zeros(1, device=embeddings.device, requires_grad=True).squeeze()
        
        # Add triplet loss if triplets provided
        if triplets is not None and len(triplets) > 0:
            trip_losses = []
            for anchor_idx, pos_idx, neg_idx in triplets:
                if anchor_idx < len(embeddings) and pos_idx < len(embeddings) and neg_idx < len(embeddings):
                    anchor = embeddings[anchor_idx:anchor_idx+1]
                    positive = embeddings[pos_idx:pos_idx+1]
                    negative = embeddings[neg_idx:neg_idx+1]
                    loss = self.triplet(anchor, positive, negative)
                    trip_losses.append(loss)
            
            if trip_losses:
                trip_loss = torch.stack(trip_losses).mean()
                total_loss += self.trip_weight * trip_loss
        
        return {
            'total': total_loss,
            'contrastive': cont_loss,
            'triplet': trip_loss
        }

# ========== DATA MINING ==========
def create_gps_labels(coordinates, threshold_meters=25):
    """Create cluster labels based on GPS proximity"""
    coords = np.array(coordinates)
    labels = np.zeros(len(coords), dtype=int)
    label_counter = 0
    
    for i in range(len(coords)):
        if labels[i] == 0:  # Not yet assigned
            # Find all points within threshold using haversine
            distances = haversine_np(coords[i, 0], coords[i, 1], coords[:, 0], coords[:, 1])
            close_points = distances < threshold_meters
            labels[close_points] = label_counter
            label_counter += 1
    
    return labels

def mine_hard_negatives_online(embeddings, coordinates, labels, hard_neg_ratio=0.5, neg_threshold_m=100.0):
    """Mine hard negatives: similar embeddings but geographically distant"""
    embeddings_np = embeddings.detach().cpu().numpy()
    coords_np = np.array(coordinates)
    
    hard_negatives = []
    
    for i in range(len(embeddings_np)):
        # Geographic negatives (far away)
        distances = haversine_np(coords_np[i, 0], coords_np[i, 1], coords_np[:, 0], coords_np[:, 1])
        geo_mask = (distances > neg_threshold_m) & (labels != labels[i])
        
        if geo_mask.sum() == 0:
            continue
            
        # Among geographic negatives, find most similar embeddings
        geo_indices = np.where(geo_mask)[0]
        similarities = np.dot(embeddings_np[i], embeddings_np[geo_indices].T)
        
        # Select hard negatives (most similar among geographic negatives)
        if random.random() < hard_neg_ratio:
            # Hard negative: most similar
            hardest_idx = geo_indices[np.argmax(similarities)]
        else:
            # Random negative
            hardest_idx = np.random.choice(geo_indices)
            
        hard_negatives.append(hardest_idx)
    
    return hard_negatives

def mine_triplets_online(embeddings, coordinates, labels, num_triplets=8, hard_neg_ratio=0.5, neg_threshold_m=100.0):
    """Mine triplets with hard negative sampling"""
    # Ensure tensors are on CPU for numpy operations
    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    coords_np = coordinates.detach().cpu().numpy() if isinstance(coordinates, torch.Tensor) else coordinates
    
    triplets = []
    max_attempts = num_triplets * 3  # Prevent infinite loops
    attempts = 0
    
    while len(triplets) < num_triplets and attempts < max_attempts:
        attempts += 1
        
        # Random anchor
        anchor_idx = np.random.randint(0, len(labels_np))
        anchor_label = labels_np[anchor_idx]
        
        # Find positive (same label, different index)  
        pos_candidates = np.where((labels_np == anchor_label) & (np.arange(len(labels_np)) != anchor_idx))[0]
        if len(pos_candidates) == 0:
            continue
        pos_idx = np.random.choice(pos_candidates)
        
        # Find hard negative
        distances = haversine_np(coords_np[anchor_idx, 0], coords_np[anchor_idx, 1], 
                                coords_np[:, 0], coords_np[:, 1])
        geo_mask = (distances > neg_threshold_m) & (labels_np != anchor_label)
        
        if geo_mask.sum() == 0:
            continue
            
        geo_indices = np.where(geo_mask)[0]
        
        if random.random() < hard_neg_ratio and len(geo_indices) > 0:
            # Hard negative mining
            embeddings_np = embeddings.detach().cpu().numpy()
            similarities = np.dot(embeddings_np[anchor_idx], embeddings_np[geo_indices].T)
            neg_idx = geo_indices[np.argmax(similarities)]
        else:
            # Random negative
            neg_idx = np.random.choice(geo_indices) if len(geo_indices) > 0 else None
            
        if neg_idx is not None:
            triplets.append((anchor_idx, pos_idx, neg_idx))
    
    return triplets

# ========== DATASET ==========
class VPRDataset(Dataset):
    """Dataset for VPR training with GPS coordinates"""
    def __init__(self, df, pos_threshold_m=25, precomputed_labels=None):
        self.df = df
        self.pos_threshold_m = pos_threshold_m
        
        # Use precomputed labels if provided, otherwise create new ones
        if precomputed_labels is not None:
            self.labels = precomputed_labels
            print(f"Using precomputed GPS labels with {len(np.unique(self.labels))} clusters")
        else:
            coordinates = df[['lat', 'lon']].values
            self.labels = create_gps_labels(coordinates, pos_threshold_m)
            print(f"Created {len(np.unique(self.labels))} GPS clusters from {len(df)} images")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['abs_path']
        image = Image.open(image_path).convert('RGB')
        
        return {
            'image': image,
            'coordinates': [row['lat'], row['lon']],
            'label': self.labels[idx],
            'idx': idx
        }

def collate_fn(batch):
    """Custom collate function for VPR batch"""
    images = [item['image'] for item in batch]
    coordinates = torch.tensor([item['coordinates'] for item in batch]).float()
    labels = torch.tensor([item['label'] for item in batch]).long()
    indices = torch.tensor([item['idx'] for item in batch]).long()
    
    return {
        'images': images,
        'coordinates': coordinates,
        'labels': labels,
        'indices': indices
    }

# ========== TRAINING LOOP ==========
def train_vpr_model(model, train_loader, val_df, config, save_path, model_name="vpr_model", seed=42):
    """Main training loop with hybrid loss and hard negative mining"""
    
    # Setup loss and optimizer
    hybrid_loss = HybridVPRLoss(
        temperature=config['contrastive_temperature'],
        margin=config['triplet_margin'],
        contrastive_weight=config['contrastive_weight'],
        triplet_weight=config['triplet_weight']
    )
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])
    
    # Initialize metric logger
    logger = MetricLogger(model_name=model_name, seed=seed)
    
    best_recall = 0.0
    model.train()
    training_start_time = time.time()
    
    # Extract train_df and labels for validation (avoid recreating labels)
    train_df = train_loader.dataset.df
    train_labels = train_loader.dataset.labels
    
    print("Starting training...")
    
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        epoch_losses = {'total': 0, 'contrastive': 0, 'triplet': 0}
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch data to device (get device from model)
            device = next(model.parameters()).device
            batch_labels = batch['labels'].to(device)
            batch_coordinates = batch['coordinates'].to(device)
            
            # Forward pass
            embeddings = model.forward(batch['images'])
            
            # Mine triplets every few iterations
            triplets = None
            if batch_idx % 3 == 0:  # Mine every 3rd batch
                triplets = mine_triplets_online(
                    embeddings, batch_coordinates, batch_labels,
                    num_triplets=min(8, len(embeddings)//2),
                    hard_neg_ratio=config['hard_neg_ratio'],
                    neg_threshold_m=config['neg_threshold_m']
                )
            
            # Compute loss
            loss_dict = hybrid_loss(embeddings, batch_labels, triplets)
            
            # Backward pass
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key].item()
            num_batches += 1
            
            # Log and print progress
            current_lr = optimizer.param_groups[0]['lr']
            if batch_idx % 20 == 0:
                logger.log_batch(
                    epoch, batch_idx, 
                    loss_dict['total'].item(), 
                    loss_dict['contrastive'].item(),
                    loss_dict['triplet'].item(),
                    current_lr
                )
                
                print(f"Epoch {epoch:2d}, Batch {batch_idx:3d}: "
                      f"Total={loss_dict['total'].item():.4f}, "
                      f"Cont={loss_dict['contrastive'].item():.4f}, "
                      f"Trip={loss_dict['triplet'].item():.4f}")
        
        scheduler.step()
        
        # Epoch summary
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, epoch_losses['total'], epoch_losses['contrastive'], epoch_losses['triplet'], current_lr)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch:2d} Summary: "
              f"Total={epoch_losses['total']:.4f}, "
              f"Cont={epoch_losses['contrastive']:.4f}, "
              f"Trip={epoch_losses['triplet']:.4f}, "
              f"Time={epoch_time:.1f}s")
        
        # Proper validation every few epochs
        if epoch % 3 == 0 or epoch == config['epochs'] - 1:
            val_recall, train_labels = validate_model(model, train_df, val_df, config, train_labels)
            logger.log_validation(epoch, val_recall)
            print(f"Validation R@1: {val_recall:.4f}")
            
            # Save best model
            if val_recall > best_recall:
                best_recall = val_recall
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'recall': val_recall,
                    'model_name': model_name,
                    'seed': seed,
                    'training_time': time.time() - training_start_time
                }, save_path)
                print(f"Saved best model with R@1: {val_recall:.4f} to {save_path}")
    
    total_training_time = time.time() - training_start_time
    logger.log_final_summary(best_recall, config['epochs'], total_training_time)
    
    print(f"Training completed in {total_training_time:.1f}s. Best validation R@1: {best_recall:.4f}")
    print(f"Metrics logged to: {logger.log_file}")
    return best_recall

def validate_model(model, train_df, val_df, config, train_labels=None):
    """Proper validation: train embeddings vs val embeddings"""
    model.eval()
    
    # Create train dataset (use precomputed labels to avoid recreation)
    if train_labels is None:
        train_dataset = VPRDataset(train_df, config['pos_threshold_m'])
        train_labels = train_dataset.labels
    else:
        train_dataset = VPRDataset(train_df, config['pos_threshold_m'], train_labels)
    
    # Create val dataset (separate clustering)
    val_dataset = VPRDataset(val_df, config['pos_threshold_m'])
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Extract train embeddings (gallery)
    train_embeddings = []
    train_coordinates = []
    
    with torch.no_grad():
        for batch in train_loader:
            emb = model.forward(batch['images'])
            train_embeddings.append(emb.cpu().numpy())
            train_coordinates.append(batch['coordinates'].cpu().numpy())
    
    train_embeddings = np.concatenate(train_embeddings, axis=0)
    train_coordinates = np.concatenate(train_coordinates, axis=0)
    
    # Extract val embeddings (queries)
    val_embeddings = []
    val_coordinates = []
    
    with torch.no_grad():
        for batch in val_loader:
            emb = model.forward(batch['images'])
            val_embeddings.append(emb.cpu().numpy())
            val_coordinates.append(batch['coordinates'].cpu().numpy())
    
    val_embeddings = np.concatenate(val_embeddings, axis=0)
    val_coordinates = np.concatenate(val_coordinates, axis=0)
    
    # Build FAISS index with TRAIN embeddings
    dim = train_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(train_embeddings.astype('float32'))
    
    # Search VAL embeddings against TRAIN index
    D, I = index.search(val_embeddings.astype('float32'), 10)
    
    # Evaluate: val queries vs train gallery
    metrics = evaluate(
        train_coordinates[:, 0], train_coordinates[:, 1],  # gallery (train)
        val_coordinates[:, 0], val_coordinates[:, 1],     # queries (val)
        I, config['pos_threshold_m']
    )
    
    model.train()
    return metrics['R@1'], train_labels  # Return labels for reuse

# ========== MAIN ==========
def main():
    ap = argparse.ArgumentParser(description='Train VPR models with triplet + contrastive learning')
    ap.add_argument("--train_csv", required=True, help="Training CSV file")
    ap.add_argument("--val_csv", required=True, help="Validation CSV file") 
    ap.add_argument("--camera_id", default="front_left_center", help="Camera ID to use")
    ap.add_argument("--model", required=True,
                    choices=["clip_b32", "clip_l14", "siglip_b16", "dinov2_b", "convnext_b", "resnet50"],
                    help="Model to train")
    ap.add_argument("--input_size", type=int, default=224, help="Input image size")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze backbone")
    ap.add_argument("--save_path", type=str, default="", help="Path to save trained model")
    args = ap.parse_args()
    
    # Set reproducibility
    set_reproducible_training(args.seed)
    
    # Get standardized config and override with args
    config = get_standardized_training_config()
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr
    })
    
    print(f"Training config: {config}")
    
    # Load data
    print(f"Loading data with camera_id={args.camera_id}")
    df_train = load_csv(args.train_csv, args.camera_id)
    df_val = load_csv(args.val_csv, args.camera_id)
    print(f"Train: {len(df_train)} images, Val: {len(df_val)} images")
    
    # Create model
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training but was not found.")
    device = "cuda"
    model = TrainableVPREncoder(
        args.model, 
        input_size=args.input_size, 
        device=device,
        freeze_backbone=args.freeze_backbone,
        projection_dim=config['projection_dim'],
        dropout=config['dropout']
    )
    
    # Create dataset and loader
    train_dataset = VPRDataset(df_train, config['pos_threshold_m'])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Create directories for organized saving
    os.makedirs("logs", exist_ok=True)
    os.makedirs("trained_models", exist_ok=True)
    
    # Set save path in trained_models directory
    save_path = args.save_path if args.save_path else f"trained_models/trained_{args.model}_seed{args.seed}.pth"
    
    # Train model
    best_recall = train_vpr_model(model, train_loader, df_val, config, save_path, args.model, args.seed)
    
    print(f"\nTraining completed!")
    print(f"Model: {args.model}")
    print(f"Best validation R@1: {best_recall:.4f}")
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    main()
