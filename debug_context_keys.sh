#!/usr/bin/env bash
#SBATCH --job-name=debug_keys
#SBATCH --partition=COOP
#SBATCH --output=debug_keys.%j.out
#SBATCH --error=debug_keys.%j.err
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
import sys
sys.path.insert(0, 'src')

print("="*80)
print("COMPARING KEYS BETWEEN TRAJECTORY AND DETECTION CONTEXT")
print("="*80)

# Load trajectory annotations
traj_path = 'src/data/ann/train/trajectories_ann.json'
with open(traj_path, 'r') as f:
    traj_ann = json.load(f)

# Load detection context
det_path = 'src/data/ann/train/detection_context.json'
with open(det_path, 'r') as f:
    det_ctx = json.load(f)

print(f"\nTrajectory annotations keys (first 10): {list(traj_ann.keys())[:10]}")
print(f"Detection context keys (first 10): {list(det_ctx.keys())[:10]}")

# Check overlap
traj_keys = set(traj_ann.keys())
det_keys = set(det_ctx.keys())

overlap = traj_keys & det_keys
missing_in_det = traj_keys - det_keys
extra_in_det = det_keys - traj_keys

print(f"\nTrajectory keys: {len(traj_keys)}")
print(f"Detection context keys: {len(det_keys)}")
print(f"Overlap: {len(overlap)}")
print(f"Missing in detection context: {len(missing_in_det)}")
print(f"Extra in detection context: {len(extra_in_det)}")

if missing_in_det:
    print(f"\nFirst 5 missing in det_ctx: {list(missing_in_det)[:5]}")
if extra_in_det:
    print(f"First 5 extra in det_ctx: {list(extra_in_det)[:5]}")

# Check if a specific key exists and has data
if overlap:
    test_key = list(overlap)[0]
    print(f"\nTest key '{test_key}':")
    print(f"  In traj_ann: {test_key in traj_ann}")
    print(f"  In det_ctx: {test_key in det_ctx}")
    if test_key in det_ctx:
        dets = det_ctx[test_key].get('detections_per_frame', [])
        print(f"  detections_per_frame length: {len(dets)}")
        if dets:
            print(f"  First frame detections: {len(dets[0])}")
            if dets[0]:
                print(f"  First detection: {dets[0][0][:4]}...")

print("\nDONE")
PYTHON_SCRIPT
