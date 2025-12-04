#!/usr/bin/env bash
#SBATCH --job-name=debug_ctx_compare
#SBATCH --partition=COOP
#SBATCH --output=debug_ctx_compare.%j.out
#SBATCH --error=debug_ctx_compare.%j.err
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
import numpy as np
import sys
sys.path.insert(0, 'src')

from tqdm import tqdm
from yacs.config import CfgNode as CN
from dataset.training_dataset import KITTIDataset
from torch.utils.data import DataLoader

print("="*80)
print("DEBUG: Simulating batch_generation.py logic")
print("="*80)

# Load config
cfg = CN()
cfg.DATASET = CN()
cfg.DATASET.ROOT = '/FARM/ldibella/hybridtrack_original'
cfg.DATASET.SEQ_LEN = 20
cfg.DATASET.SEQ_STRIDE = 1
cfg.DATASET.RATIO_DATASET = 100.0

# Create dataset and dataloader
print("\n1. Creating dataset...")
dataset = KITTIDataset(cfg, mode='train')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

print(f"   Dataset length: {len(dataset)}")

# Simulate batch generation
print("\n2. Simulating batch_generation.GenerateBatch()...")
Context = []
non_empty_contexts = 0

for idx, data_loaded in enumerate(dataloader):
    if idx >= 10:  # Only check first 10
        break
    
    pose_target, pose_input = data_loaded[0], data_loaded[1]
    context_dict = data_loaded[2] if len(data_loaded) >= 3 and data_loaded[2] is not None else {}
    
    # This is what batch_generation.py does
    detection_ctx = context_dict.get('detection_context', {}) if isinstance(context_dict, dict) else {}
    
    print(f"\n   Sample {idx}:")
    print(f"     context_dict type: {type(context_dict)}")
    print(f"     detection_ctx type: {type(detection_ctx)}")
    print(f"     detection_ctx truthy: {bool(detection_ctx)}")
    
    if detection_ctx:
        print(f"     detection_ctx keys: {detection_ctx.keys() if isinstance(detection_ctx, dict) else 'N/A'}")
        has_dpf = 'detections_per_frame' in detection_ctx if isinstance(detection_ctx, dict) else False
        print(f"     has 'detections_per_frame': {has_dpf}")
        
        if has_dpf:
            det_per_frame = detection_ctx['detections_per_frame']
            print(f"     det_per_frame type: {type(det_per_frame)}, len: {len(det_per_frame)}")
            
            # Count non-empty frames
            frame_tensors = []
            for frame_dets in det_per_frame:
                if isinstance(frame_dets, np.ndarray) and frame_dets.size > 0:
                    frame_tensors.append(torch.from_numpy(frame_dets).float())
                elif isinstance(frame_dets, list) and len(frame_dets) > 0:
                    frame_tensors.append(torch.tensor(frame_dets, dtype=torch.float32))
                else:
                    frame_tensors.append(torch.zeros((0, 8), dtype=torch.float32))
            
            non_empty = sum(1 for t in frame_tensors if t.shape[0] > 0)
            print(f"     Non-empty frames: {non_empty}/{len(frame_tensors)}")
            if non_empty > 0:
                non_empty_contexts += 1
            Context.append(frame_tensors)
        else:
            Context.append([torch.zeros((0, 8), dtype=torch.float32) for _ in range(20)])
    else:
        print("     detection_ctx is empty/falsy - appending zeros")
        Context.append([torch.zeros((0, 8), dtype=torch.float32) for _ in range(20)])

print(f"\n\n3. Summary:")
print(f"   Contexts with non-empty frames: {non_empty_contexts}/10")
print(f"   Context[0] first frame shape: {Context[0][0].shape if Context else 'N/A'}")

print("\nDONE")
PYTHON_SCRIPT
