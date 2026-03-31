import torch
import torch.nn as nn
import timm

class MiniLSS(nn.Module):
    def __init__(self, bev_h=200, bev_w=200, depth_bins=10):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.depth_bins = depth_bins
        
        # 1. Backbone: Extract features from image (using tiny EfficientNet for speed)
        self.backbone = timm.create_model('tf_efficientnet_b0', pretrained=False, in_chans=3, features_only=True, out_indices=[2])
        
        # 2. Depth Head: Predict probability distribution over depth bins per pixel
        self.depth_head = nn.Conv2d(in_channels=40, out_channels=depth_bins, kernel_size=1)
        
        # 3. Context Conv: Process the flattened BEV grid
        self.bev_conv = nn.Sequential(
            nn.Conv2d(in_channels=40 * depth_bins, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1) # 2 Classes: Free vs Occupied
        )

    def forward(self, img, intrinsics, extrinsics):
        # img shape: [B, N, 3, H, W]
        B, N, C_img, H_img, W_img = img.shape
        
        # Flatten B and N to process all images independently through backbone
        img = img.view(B * N, C_img, H_img, W_img)
        
        features = self.backbone(img)[0] # Shape: [B*N, 40, H/8, W/8]
        
        # LIFT: Predict depth probabilities
        depth_probs = torch.softmax(self.depth_head(features), dim=1) # [B*N, Depth_Bins, H/8, W/8]
        
        # Create Frustum: Multiply features by depth probabilities
        # [B*N, 40, H/8, W/8] -> [B*N, 40, Depth_Bins, H/8, W/8]
        frustum = features.unsqueeze(2) * depth_probs.unsqueeze(1)
        _, C, D, H, W = frustum.shape
        frustum = frustum.view(B*N, C*D, H, W) # Flatten to [B*N, C*D, H, W]
        
        # SPLAT (Simplified for Smoke Test): 
        # Instead of complex geometric unprojection (which requires perfect intrinsics), 
        # we use an adaptive average pool to force the frustum into the BEV grid shape.
        bev_features_per_cam = torch.nn.functional.adaptive_avg_pool2d(frustum, (self.bev_h, self.bev_w)) # [B*N, C*D, bev_h, bev_w]
        
        # FUSE: Separate B and N back, and pool across N (multiple views)
        _, C_bev, H_bev, W_bev = bev_features_per_cam.shape
        bev_features_all = bev_features_per_cam.view(B, N, C_bev, H_bev, W_bev)
        
        # Average features from all camera views
        bev_features = bev_features_all.mean(dim=1) # [B, C*D, bev_h, bev_w]
        
        # SHOOT: Final BEV prediction
        bev_output = self.bev_conv(bev_features) # Shape: [B, 2, 200, 200]
        return bev_output
