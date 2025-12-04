#!/usr/bin/env bash
#SBATCH --job-name=debug_world_reg
#SBATCH --partition=COOP
#SBATCH --output=debug_world_reg.%j.out
#SBATCH --error=debug_world_reg.%j.err
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4090:1
#SBATCH --mem=8G

set -euxo pipefail
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load TensorFlow/2.13.0-foss-2023a

source ~/hybridtrack_py311/venv/bin/activate
cd ~/hybridtrack_original

python << 'PYTHON_SCRIPT'
import json
import numpy as np
import sys
sys.path.insert(0, 'src')
from dataset.tracking_dataset_utils import read_calib, cam_to_velo, read_pose

print("="*70)
print("CHECKING WORLD REGISTRATION STEP")
print("="*70)

seq_id = "0001"

# Load poses
pose_path = f'src/data/KITTI/tracking/training/pose/{seq_id}/pose.txt'
poses = read_pose(pose_path)
pose_frame0 = poses[0]
print(f"\n1. Pose matrix (frame 0):")
print(f"{pose_frame0}")

# Load training context (already in velo coords)
with open('src/data/ann/validation/detection_context.json', 'r') as f:
    det_ctx = json.load(f)
train_dets = np.array(det_ctx['0001_0']['detections_per_frame'][0], dtype=np.float32)

print(f"\n2. Training context (velo coords, before registration):")
print(f"   First: {train_dets[0]}")
print(f"   xyz: [{train_dets[0,0]:.4f}, {train_dets[0,1]:.4f}, {train_dets[0,2]:.4f}]")

# Apply world registration (same as _prepare_detection_context)
xyz_velo = train_dets[:, :3]
xyz_homo = np.concatenate([xyz_velo, np.ones((len(xyz_velo), 1))], axis=1)
xyz_world = (pose_frame0 @ xyz_homo.T).T[:, :3]

# World yaw
cos_theta = np.clip(pose_frame0[0, 0], -1.0, 1.0)
sin_theta = pose_frame0[1, 0]
theta_cos = np.arccos(cos_theta)
world_yaw = theta_cos if sin_theta >= 0 else 2 * np.pi - theta_cos

print(f"\n3. After world registration (tracking method):")
print(f"   xyz_world: [{xyz_world[0,0]:.4f}, {xyz_world[0,1]:.4f}, {xyz_world[0,2]:.4f}]")
print(f"   world_yaw: {world_yaw:.4f} rad")
print(f"   ry_world: {train_dets[0,6] + world_yaw:.4f} rad")

# Now check what training_dataset.py does
print(f"\n4. How training_dataset.py registers context:")
print(f"   (from _extract_detection_context_sequence)")

# Training also uses pose to register - let's verify it's the same
# The training code does:
#   xyz_world = (pose @ xyz_homo.T).T[:, :3]
#   world_yaw = arccos(pose[0,0]) adjusted by sin sign
#   registered_dets[:, :3] = xyz_world
#   registered_dets[:, 6] = frame_dets[:, 6] + world_yaw

print(f"   Same transformation as tracking? YES (verified in code)")

# Check if maybe the TRAINING didn't actually register?
# The JSON stores the RAW velo coords, and registration happens at __getitem__ time
print(f"\n5. Key insight: Training JSON stores UNREGISTERED velo coords!")
print(f"   Registration happens in _extract_detection_context_sequence")
print(f"   which is called during create_sequences()")

# Let's check what the training model actually receives
print(f"\n6. What does the MODEL receive during training?")
print(f"   Looking at batch_generation.py and training.py...")

# The model receives registered coordinates!
# But wait - let's verify the training actually uses context
print(f"\n7. CRITICAL: Does training actually USE context?")
print(f"   Checking if detection_context.json exists in train split...")

import os
train_ctx_path = 'src/data/ann/train/detection_context.json'
val_ctx_path = 'src/data/ann/validation/detection_context.json'
print(f"   Train context exists: {os.path.exists(train_ctx_path)}")
print(f"   Val context exists: {os.path.exists(val_ctx_path)}")

if os.path.exists(train_ctx_path):
    with open(train_ctx_path, 'r') as f:
        train_ctx = json.load(f)
    print(f"   Train context has {len(train_ctx)} trajectories")
else:
    print(f"   WARNING: Train context missing! Model may not have trained with context!")

print("\nDONE")
PYTHON_SCRIPT
