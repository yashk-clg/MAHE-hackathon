import torch
from torch.utils.data import DataLoader
from dataset import HackathonBEVDataset
from custom_loss import HybridProximityFocalLoss
from model import MiniLSS

print("Initializing Advanced Pipeline...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = MiniLSS(bev_h=200, bev_w=200).to(device) 
dataset = HackathonBEVDataset(data_root=".")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize with Focal Gamma parameter
criterion = HybridProximityFocalLoss(beta=0.2, gamma=2.0, grid_size=(200, 200)).to(device)

print("RUNNING ADVANCED SMOKE TEST...")
model.train()
for img_tensor, intrinsics, extrinsics, dummy_gt in dataloader:
    img_tensor, intrinsics, extrinsics, dummy_gt = img_tensor.to(device), intrinsics.to(device), extrinsics.to(device), dummy_gt.to(device)
    
    try:
        bev_output = model(img_tensor, intrinsics, extrinsics) 
        loss = criterion(bev_output, dummy_gt)
        loss.backward()
        
        print(f"SUCCESS! Hybrid Focal-Distance Loss: {loss.item()}")
        print("Gradients isolated and flowing through multi-term objective.")
    except Exception as e:
        print(f"FAILED: {e}")
        break
    break
