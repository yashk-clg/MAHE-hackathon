import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class HackathonBEVDataset(Dataset):
    def __init__(self, data_root, img_size=(256, 704), cameras=("CAM_FRONT", "CAM_FRONT_RIGHT")):
        self.data_root = data_root
        self.img_size = img_size
        self.cameras = cameras
        
        # Build individual lists per camera and compute the shortest valid length
        self.cam_image_lists = []
        for cam in self.cameras:
            cam_folder = os.path.join(data_root, cam)
            if os.path.exists(cam_folder):
                # Sort filenames chronologically
                files = sorted([f for f in os.listdir(cam_folder) if f.endswith('.jpg')])
                self.cam_image_lists.append(files)
            else:
                self.cam_image_lists.append([])
                
        valid_lengths = [len(lst) for lst in self.cam_image_lists if len(lst) > 0]
        self.num_samples = min(valid_lengths) if valid_lengths else 0
        if self.num_samples > 5:
            self.num_samples = 5 # Truncate for the smoke test

    def __getitem__(self, idx):
        imgs = []
        for i, cam in enumerate(self.cameras):
            if self.num_samples > 0 and len(self.cam_image_lists[i]) > idx:
                img_path = os.path.join(self.data_root, cam, self.cam_image_lists[i][idx])
                img = Image.open(img_path).convert('RGB')
                img = img.resize((self.img_size[1], self.img_size[0]))
                img = np.array(img)
            else:
                img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                
            img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std
            imgs.append(img)

        # Stack to [num_cameras, 3, H, W]
        img_tensor = torch.stack(imgs, dim=0)

        # Fake calibrations for each camera
        intrinsics = torch.eye(3).unsqueeze(0).repeat(len(self.cameras), 1, 1)
        extrinsics = torch.eye(4).unsqueeze(0).repeat(len(self.cameras), 1, 1)

        # Dummy Ground Truth (Matches model output: 2 classes, 200x200 grid)
        dummy_gt = torch.zeros(200, 200, dtype=torch.long)

        return img_tensor, intrinsics, extrinsics, dummy_gt

    def __len__(self):
        return self.num_samples if self.num_samples > 0 else 5
