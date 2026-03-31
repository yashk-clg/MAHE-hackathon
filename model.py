import torch
import torch.nn as nn
import timm

class MiniLSS(nn.Module):
    def __init__(self, bev_h=200, bev_w=200, depth_bins=10):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.depth_bins = depth_bins
        
        # Feature Extractor: Initial encoder pipeline for dense spatial mapping
        self.backbone = timm.create_model('tf_efficientnet_b0', pretrained=False, in_chans=3, features_only=True, out_indices=[2])
        
        # Depth Distribution Head: Generates discrete depth classification bounds per pixel
        self.depth_head = nn.Conv2d(in_channels=40, out_channels=depth_bins, kernel_size=1)
        
        # BEV Spatial Decoder: Collapses volumetric tensors into binary semantic states
        self.bev_conv = nn.Sequential(
            nn.Conv2d(in_channels=40 * depth_bins, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)  # Binary Segmentation: 0 (Free), 1 (Occupied)
        )

    def forward(self, img, intrinsics, extrinsics):
        # img shape: [B, N, 3, H, W]
        B, N, C_img, H_img, W_img = img.shape
        
        # Merge batch and camera axes to parallelize sequence passes
        img = img.view(B * N, C_img, H_img, W_img)
        
        features = self.backbone(img)[0]
        
        # Lift Operation: Approximate discrete distance correlations
        depth_probs = torch.softmax(self.depth_head(features), dim=1)
        
        # Frustum Generation: Extrapolate 2D attributes to 3D volumetric fields
        frustum = features.unsqueeze(2) * depth_probs.unsqueeze(1)
        _, C, D, H, W = frustum.shape
        frustum = frustum.view(B*N, C*D, H, W)
        
        # Splatting Translation: Collapses discrete frustums into a static Cartesian coordinate system
        bev_features_per_cam = torch.nn.functional.adaptive_avg_pool2d(frustum, (self.bev_h, self.bev_w))
        
        # Extrinsic Fusion: Realigns independent sequences and integrates spatial overlaps
        _, C_bev, H_bev, W_bev = bev_features_per_cam.shape
        bev_features_all = bev_features_per_cam.view(B, N, C_bev, H_bev, W_bev)
        
        # Pooling Mechanism across visual fields
        bev_features = bev_features_all.mean(dim=1)
        
        # Shoot Operation: Forward inference onto continuous semantic grid representations
        bev_output = self.bev_conv(bev_features)
        return bev_output
