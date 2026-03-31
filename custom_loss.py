import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridProximityFocalLoss(nn.Module):
    """
    Implements a custom structural target loss for automotive spatial tensors.
    
    1. Focal Modulation: Minimizes weight adjustments toward correctly identified majority classifications (e.g. Free Space background).
    2. Proximity Range Penalty: Enforces exponentially heavier loss factors on local collision radius misclassifications.
    """
    def __init__(self, beta=0.2, gamma=2.0, grid_size=(200, 200), perception_range=50.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.grid_size = grid_size
        
        # Instantiate local proximity spatial mapping
        y_coords = torch.arange(grid_size[0]).float()
        x_coords = torch.arange(grid_size[1]).float()
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        meters_per_pixel = perception_range / grid_size[0]
        
        # Establish radius vector magnitudes oriented toward ego coordinates
        dist_from_ego = torch.sqrt((X - grid_size[1]/2)**2 + (Y - grid_size[0])**2) * meters_per_pixel
        self.dist_weight = torch.exp(-self.beta * dist_from_ego)
        self.dist_weight = self.dist_weight / self.dist_weight.sum()

    def forward(self, pred, target):
        weight_map = self.dist_weight.to(pred.device)
        
        # Standard structural error loss matrix
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Modulate confidence margins against baseline distribution factors
        pt = torch.exp(-ce_loss)
        focal_modulation = (1 - pt) ** self.gamma
        
        # Apply combined localized penalty fields
        hybrid_loss = focal_modulation * ce_loss * weight_map
        
        return hybrid_loss.mean()
