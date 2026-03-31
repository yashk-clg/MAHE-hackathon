# BEV Perception Pipeline

This is a deep learning pipeline for Bird's-Eye View (BEV) perception from multiple autonomous vehicle camera streams. The model utilizes an EfficientNet-based `Lift-Splat-Shoot` (LSS) approach, dynamically aligning asynchronous multi-camera inputs (`CAM_FRONT`, `CAM_FRONT_RIGHT`) and projecting their feature frustums onto a fused top-down occupancy grid.

It uses a custom engineered `HybridProximityFocalLoss`, combining Focal Loss with a predefined distance-decay weight map to dramatically penalize classification errors located directly in front of the ego-vehicle.

## Prerequisites & Installation

1. Clone this repository:
```bash
git clone https://github.com/yashk-clg/MAHE-hackathon.git
cd MAHE-hackathon
```

2. Create a virtual environment and install the required dependencies:
```bash
python -m venv .venv

# Activate the virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset Configuration

Place your camera image sequences inside the root repository directory using their original folder names. Currently, the dataset loader is configured to synchronize:
- `CAM_FRONT`
- `CAM_FRONT_RIGHT`

If these dataset folders are effectively missing or empty, the `dataset.py` loader will intelligently generate un-occupied dummy image tensors to safely allow testing mathematical matrix flows.

## Running the Pipeline

You can verify the entire multi-camera fusion projection and the backward gradient flow of the custom loss using the smoke test script:

```bash
python smoke_test.py
```
