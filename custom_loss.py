import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridProximityFocalLoss(nn.Module):
    """
    Advanced Loss Engineering:
    1. FOCAL LOSS: Down-weights the overwhelming majority of "Free Space" pixels 
       so the model doesn't just learn to guess "empty" everywhere.
    2. DISTANCE WEIGHTING (DWE): Applies Exponential Decay to hyper-penalize 
       errors in the 0-15m survival zone.
    """
    def __init__(self, beta=0.2, gamma=2.0, grid_size=(200, 200), perception_range=50.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.grid_size = grid_size
        
        # 1. Generate Distance Weight Map (Ego-Centric)
        y_coords = torch.arange(grid_size[0]).float()
        x_coords = torch.arange(grid_size[1]).float()
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        meters_per_pixel = perception_range / grid_size[0]
        
        # Calculate distance from the bottom-center of the image (where the car is)
        dist_from_ego = torch.sqrt((X - grid_size[1]/2)**2 + (Y - grid_size[0])**2) * meters_per_pixel
        self.dist_weight = torch.exp(-self.beta * dist_from_ego)
        self.dist_weight = self.dist_weight / self.dist_weight.sum()

    def forward(self, pred, target):
        weight_map = self.dist_weight.to(pred.device)
        
        # 2. Calculate standard Cross Entropy (no reduction)
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # 3. Calculate Focal Loss modulation
        # p_t is the model's confidence in the true class
        pt = torch.exp(-ce_loss)
        focal_modulation = (1 - pt) ** self.gamma
        
        # 4. Combine: Focal modulator * CE Loss * Distance Weight
        hybrid_loss = focal_modulation * ce_loss * weight_map
        
        return hybrid_loss.mean()
