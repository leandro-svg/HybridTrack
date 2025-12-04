#!/usr/bin/env bash
#SBATCH --job-name=debug_state_ctx
#SBATCH --partition=COOP
#SBATCH --output=debug_state_ctx.%j.out
#SBATCH --error=debug_state_ctx.%j.err
#SBATCH --time=0    -00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4090:1
#SBATCH --mem=8G

set -euxo pipefail

export TORCH_CUDA_ARCH_LIST="8.0"
export CUDAARCH=80

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load TensorFlow/2.13.0-foss-2023a

HYBRIDTRACK_VENV=~/hybridtrack_py311/venv
source "$HYBRIDTRACK_VENV/bin/activate"

cd ~/hybridtrack_original
pip install -r requirements.txt

python << 'PYTHON_SCRIPT'
import torch
import numpy as np
import json
import sys
sys.path.insert(0, 'src')

print("="*80)
print("COMPARING STATE FORMAT vs CONTEXT FORMAT")
print("="*80)

# Load dataset.pt which contains pre-processed training sequences
dataset_path = 'src/data/checkpoints/dataset.pt'
print(f"\n1. Loading training dataset from: {dataset_path}")
dataset = torch.load(dataset_path, map_location='cpu')

print(f"   Dataset type: {type(dataset)}")
if isinstance(dataset, dict):
    print(f"   Keys: {list(dataset.keys())}")
elif isinstance(dataset, list):
    print(f"   Length: {len(dataset)}")
    if len(dataset) > 0:
        print(f"   First item type: {type(dataset[0])}")
        if isinstance(dataset[0], dict):
            print(f"   First item keys: {list(dataset[0].keys())}")

# Try to extract trajectory states
print("\n2. Trajectory state format (what model tracks):")
if isinstance(dataset, list) and len(dataset) > 0:
    item = dataset[0]
    if isinstance(item, dict):
        for k, v in item.items():
            if hasattr(v, 'shape'):
                print(f"   {k}: shape={v.shape}, dtype={v.dtype}")
                if 'gt' in k.lower() or 'det' in k.lower() or 'state' in k.lower():
                    if v.ndim >= 2:
                        print(f"     First frame first obj: {v[0, :7] if v.shape[-1] >= 7 else v[0]}")
            elif isinstance(v, (list, tuple)) and len(v) > 0:
                print(f"   {k}: list of len {len(v)}, first type: {type(v[0])}")

# Load detection context for comparison
print("\n3. Detection context format (from JSON):")
ctx_path = 'src/data/ann/validation/detection_context.json'
with open(ctx_path, 'r') as f:
    det_ctx = json.load(f)

first_key = list(det_ctx.keys())[0]
ctx_frame0 = np.array(det_ctx[first_key]['detections_per_frame'][0], dtype=np.float32)
print(f"   Key: {first_key}")
print(f"   Frame 0 shape: {ctx_frame0.shape}")
print(f"   Format: [x_velo, y_velo, z_velo, l, w, h, ry, score]")
if len(ctx_frame0) > 0:
    print(f"   First detection: {ctx_frame0[0]}")
    print(f"   z value: {ctx_frame0[0, 2]:.4f} (should be BOTTOM of box)")

# Load training annotations to compare z values
print("\n4. Ground truth annotations (from trajectories_ann.json):")
ann_path = 'src/data/ann/validation/trajectories_ann.json'
with open(ann_path, 'r') as f:
    ann = json.load(f)

first_ann_key = list(ann.keys())[0]
ann_data = ann[first_ann_key]
print(f"   Key: {first_ann_key}")
print(f"   Keys in annotation: {list(ann_data.keys()) if isinstance(ann_data, dict) else 'list'}")

if isinstance(ann_data, dict) and 'annotations' in ann_data:
    ann_frame0 = ann_data['annotations'][0]
    print(f"   Frame 0 annotation: {ann_frame0}")
elif isinstance(ann_data, list):
    print(f"   First item: {ann_data[0]}")

# Check convert_bbs_type_numpy transforms
print("\n5. Checking convert_bbs_type_numpy (training_dataset.py):")
print("   Training transforms on trajectory states:")
print("     z_new = z_orig + h_orig / 2  (z becomes CENTER)")
print("     ry_new = (pi - ry_orig) + pi/2  (yaw transformed)")
print("")
print("   Detection context (setup_detection_context.py):")
print("     z = z_velo (BOTTOM of box, no transform)")
print("     ry = ry_orig + world_yaw (only registration, no pi transform)")

# Numerical comparison
print("\n6. NUMERICAL COMPARISON:")
if len(ctx_frame0) > 0:
    z_ctx = ctx_frame0[0, 2]
    h_ctx = ctx_frame0[0, 5]
    ry_ctx = ctx_frame0[0, 6]
    
    # What z would be after convert_bbs_type transform
    z_transformed = z_ctx + h_ctx / 2
    # What ry would be after convert_bbs_type transform  
    ry_transformed = (np.pi - ry_ctx) + np.pi / 2
    
    print(f"   Context z (bottom): {z_ctx:.4f}")
    print(f"   Context h: {h_ctx:.4f}")
    print(f"   If transformed z (center): {z_transformed:.4f}")
    print(f"   Difference: {z_transformed - z_ctx:.4f}")
    print("")
    print(f"   Context ry: {ry_ctx:.4f}")
    print(f"   If transformed ry: {ry_transformed:.4f}")
    print(f"   Difference: {ry_transformed - ry_ctx:.4f}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("""
The MODEL sees:
  - Trajectory states: z=CENTER, ry=TRANSFORMED (pi - ry + pi/2)
  - Detection context: z=BOTTOM, ry=ORIGINAL + world_yaw

This is a FUNDAMENTAL MISMATCH in coordinate conventions!

The cross-attention mechanism must learn to bridge this gap, but if the 
context is not being used effectively during training, or if the tracking
coordinate system is different, performance will degrade.

QUESTION: Does the training code ACTUALLY use detection context, or does
it pass empty context and the model learned to ignore it?
""")

# Check if context was actually used during training
print("\n7. Checking training context usage:")
try:
    from tools.batch_generation import GenerateBatch
    print("   GenerateBatch imported successfully")
    # Check if it returns non-empty context
except Exception as e:
    print(f"   Error importing: {e}")

print("\nDONE")
PYTHON_SCRIPT
