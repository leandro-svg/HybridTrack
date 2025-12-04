#!/usr/bin/env bash
#SBATCH --job-name=debug_ctx_checkpoint
#SBATCH --partition=COOP
#SBATCH --output=debug_ctx_checkpoint.%j.out
#SBATCH --error=debug_ctx_checkpoint.%j.err
#SBATCH --time=0-00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euxo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source ~/hybridtrack_py311/venv/bin/activate
cd ~/hybridtrack_original

python << 'PYTHON_SCRIPT'
import torch
import sys
sys.path.insert(0, 'src')

from yacs.config import CfgNode as CN
from dataset.training_dataset import KITTIDataset
from torch.utils.data import DataLoader

print("="*80)
print("DEBUG: Check what dataset returns")
print("="*80)

# Load config
cfg = CN()
cfg.DATASET = CN()
cfg.DATASET.ROOT = '/FARM/ldibella/hybridtrack_original'
cfg.DATASET.SEQ_LEN = 20
cfg.DATASET.SEQ_STRIDE = 1
cfg.DATASET.RATIO_DATASET = 100.0

# Create dataset
print("\n1. Creating KITTIDataset...")
dataset = KITTIDataset(cfg, mode='train')

print(f"\n2. Dataset length: {len(dataset)}")
print(f"   kitti_sequences_det_context length: {len(dataset.kitti_sequences_det_context)}")

# Check first few items
print("\n3. Checking first 3 items from dataset:")
for i in range(min(3, len(dataset))):
    gt, det, ctx = dataset[i]
    print(f"\n   Item {i}:")
    print(f"     gt shape: {gt.shape}")
    print(f"     det shape: {det.shape}")
    print(f"     ctx type: {type(ctx)}")
    if isinstance(ctx, dict):
        print(f"     ctx keys: {ctx.keys()}")
        det_ctx = ctx.get('detection_context')
        print(f"     detection_context type: {type(det_ctx)}")
        if det_ctx:
            print(f"       detection_context keys: {det_ctx.keys() if isinstance(det_ctx, dict) else 'N/A'}")
            if isinstance(det_ctx, dict) and 'detections_per_frame' in det_ctx:
                dpf = det_ctx['detections_per_frame']
                print(f"       detections_per_frame type: {type(dpf)}, len: {len(dpf)}")
                if len(dpf) > 0:
                    print(f"       First frame type: {type(dpf[0])}")
                    if hasattr(dpf[0], 'shape'):
                        print(f"       First frame shape: {dpf[0].shape}")

# Check raw kitti_sequences_det_context
print("\n4. Raw kitti_sequences_det_context[0]:")
raw_ctx = dataset.kitti_sequences_det_context[0]
print(f"   Type: {type(raw_ctx)}")
if isinstance(raw_ctx, dict):
    print(f"   Keys: {raw_ctx.keys()}")
    if 'detections_per_frame' in raw_ctx:
        dpf = raw_ctx['detections_per_frame']
        print(f"   detections_per_frame len: {len(dpf)}")
        if len(dpf) > 0:
            print(f"   First frame type: {type(dpf[0])}, shape: {dpf[0].shape if hasattr(dpf[0], 'shape') else 'N/A'}")

print("\nDONE")
PYTHON_SCRIPT
