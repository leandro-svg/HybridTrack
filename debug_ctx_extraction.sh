#!/usr/bin/env bash
#SBATCH --job-name=debug_extract
#SBATCH --partition=COOP
#SBATCH --output=debug_extract.%j.out
#SBATCH --error=debug_extract.%j.err
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euxo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source ~/hybridtrack_py311/venv/bin/activate
cd ~/hybridtrack_original

python << 'PYTHON_SCRIPT'
import json
import numpy as np
import sys
sys.path.insert(0, 'src')

print("="*80)
print("DEBUG: _extract_detection_context_sequence LOGIC")
print("="*80)

# Load detection context
det_path = 'src/data/ann/train/detection_context.json'
with open(det_path, 'r') as f:
    det_ctx = json.load(f)

# Simulate what happens in create_sequences
seq_len = 20  # default
stride = 1

# Pick a test trajectory
test_key = '0000_5'
det_data = det_ctx[test_key]
detections_per_frame = det_data.get("detections_per_frame", [])

print(f"\nTest key: {test_key}")
print(f"detections_per_frame length: {len(detections_per_frame)}")
print(f"seq_len: {seq_len}")

# Simulate extraction at different start indices
for start_idx in [0, 1, 10, 15]:
    print(f"\n--- start_idx={start_idx} ---")
    print(f"  start_idx + seq_len = {start_idx + seq_len}")
    print(f"  len(detections_per_frame) = {len(detections_per_frame)}")
    
    if start_idx + seq_len > len(detections_per_frame):
        print(f"  SKIP: {start_idx + seq_len} > {len(detections_per_frame)}")
        continue
    
    # This is what _extract_detection_context_sequence does
    sliced_detections = []
    for frame_idx in range(start_idx, start_idx + seq_len):
        frame_dets = np.array(detections_per_frame[frame_idx], dtype=np.float32)
        print(f"  Frame {frame_idx}: shape {frame_dets.shape}, num_dets={frame_dets.shape[0] if frame_dets.ndim > 0 else 0}")
        if frame_dets.size > 0 and frame_dets.shape[0] > 0:
            sliced_detections.append(frame_dets)
        else:
            sliced_detections.append(np.zeros((0, 8), dtype=np.float32))
    
    non_empty = sum(1 for d in sliced_detections if d.shape[0] > 0)
    print(f"  Non-empty frames: {non_empty}/{len(sliced_detections)}")

# Also check how trajectories_ann length compares
traj_path = 'src/data/ann/train/trajectories_ann.json'
with open(traj_path, 'r') as f:
    traj_ann = json.load(f)

traj_data = traj_ann[test_key]
traj_frames = len(traj_data.get('frame_id', []))
print(f"\n\nTrajectory {test_key} has {traj_frames} frames")
print(f"Detection context has {len(detections_per_frame)} frames")

if traj_frames != len(detections_per_frame):
    print("WARNING: Frame count mismatch!")

print("\nDONE")
PYTHON_SCRIPT
