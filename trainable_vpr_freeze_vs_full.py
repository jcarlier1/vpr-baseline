import torch
import torch.nn.functional as F
from vpr_baseline import Encoder
import numpy as np

class TrainableVPREncoder(Encoder):
    def __init__(self, name, input_size=224, device="cuda", freeze_backbone=True, projection_dim=256):
        super().__init__(name, input_size, device)
        
        # Get the feature dimension from the backbone
        self.feature_dim = self._get_feature_dim()
        
        # Create projection head for VPR adaptation
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.feature_dim // 2, projection_dim),
            torch.nn.BatchNorm1d(projection_dim)
        ).to(device)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        self.projection_dim = projection_dim
        
    def _get_feature_dim(self):
        """Determine the feature dimension of the backbone"""
        with torch.no_grad():
            if self.kind == "transformers_clip":
                # For CLIP models, we can get this from config or a test forward pass
                dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
                dummy_pil = [torch.nn.functional.to_pil_image(dummy_input[0])]
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
        total_params = 0
        
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            total_params += param.numel()
            
        trainable_params = sum(p.numel() for p in self.projection.parameters())
        total_params += trainable_params
        
        print(f"Frozen backbone parameters: {frozen_params:,}")
        print(f"Trainable projection parameters: {trainable_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params:.1%}")
    
    def get_backbone_features(self, pil_batch):
        """Extract features from the frozen backbone"""
        with torch.no_grad():
            if self.kind == "transformers_clip":
                inputs = self.processor(images=pil_batch, return_tensors="pt").to(self.device)
                feats = self.model.get_image_features(**inputs)
            elif self.kind == "transformers_dino":
                inputs = self.processor(images=pil_batch, return_tensors="pt")["pixel_values"].to(self.device)
                feats = self.model(inputs).last_hidden_state[:, 0]  # CLS token
            else:  # timm
                x = torch.stack([self.timm_tf(img) for img in pil_batch]).to(self.device)
                feats = self.model(x)
        return feats
    
    def forward(self, pil_batch):
        """Forward pass through backbone + projection"""
        backbone_feats = self.get_backbone_features(pil_batch)
        projected_feats = self.projection(backbone_feats)
        return F.normalize(projected_feats, dim=-1)
    
    def encode_batch(self, pil_batch):
        """For inference - use the trained projection"""
        if self.training:
            return self.forward(pil_batch)
        else:
            with torch.no_grad():
                projected = self.forward(pil_batch)
                return projected.cpu().numpy()

# Training utilities
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

def demonstrate_frozen_vs_full_training():
    """Show the difference between frozen and full fine-tuning"""
    
    # Create two identical models
    model_frozen = TrainableVPREncoder("dinov2_b", freeze_backbone=True)
    model_full = TrainableVPREncoder("dinov2_b", freeze_backbone=False)
    
    print("\n=== FROZEN BACKBONE ===")
    frozen_trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    frozen_total = sum(p.numel() for p in model_frozen.parameters())
    
    print("\n=== FULL FINE-TUNING ===")
    full_trainable = sum(p.numel() for p in model_full.parameters() if p.requires_grad)  
    full_total = sum(p.numel() for p in model_full.parameters())
    
    print(f"\nComparison:")
    print(f"Frozen - Trainable: {frozen_trainable:,} / {frozen_total:,} ({frozen_trainable/frozen_total:.1%})")
    print(f"Full - Trainable: {full_trainable:,} / {full_total:,} ({full_trainable/full_total:.1%})")
    print(f"Training speedup: ~{full_trainable/frozen_trainable:.1f}x faster")
    print(f"Memory savings: ~{(full_trainable-frozen_trainable)/full_trainable:.1%}")

if __name__ == "__main__":
    demonstrate_frozen_vs_full_training()
