import torch
import os
import sys
sys.path.insert(0, 'src')

print("="*80)
print("VERIFYING DETECTION CONTEXT LOADING")
print("="*80)

# Check the raw checkpoint data
dataset_path = 'src/data/checkpoints/dataset.pt'
print(f"\n1. Loading: {dataset_path}")
data = torch.load(dataset_path, map_location='cpu')

# Check index 6 (train_context)
if len(data) > 6:
    train_ctx = data[6]
    print(f"\n2. Train context (data[6]):")
    print(f"   Type: {type(train_ctx)}")
    if isinstance(train_ctx, list):
        print(f"   Length: {len(train_ctx)}")
        # Check if frames have data
        non_empty_frames = 0
        total_frames = 0
        for seq_idx, seq_ctx in enumerate(train_ctx[:10]):  # First 10 sequences
            if isinstance(seq_ctx, list):
                for frame_idx, frame_ctx in enumerate(seq_ctx):
                    total_frames += 1
                    if hasattr(frame_ctx, 'shape'):
                        if frame_ctx.shape[0] > 0:
                            non_empty_frames += 1
                            if seq_idx == 0 and frame_idx < 3:
                                print(f"   Seq {seq_idx} Frame {frame_idx}: {frame_ctx.shape}")
                                print(f"     First det: {frame_ctx[0] if frame_ctx.shape[0] > 0 else 'N/A'}")
        print(f"   Non-empty frames in first 10 seqs: {non_empty_frames}/{total_frames}")

# Check when data was generated
print("\n3. Dataset.pt modification time:")
import time
mtime = os.path.getmtime(dataset_path)
print(f"   {time.ctime(mtime)}")

# Check detection_context.json
det_ctx_path = 'src/data/ann/train/detection_context.json'
print(f"\n4. Detection context JSON: {det_ctx_path}")
print(f"   Exists: {os.path.exists(det_ctx_path)}")
if os.path.exists(det_ctx_path):
    import json
    with open(det_ctx_path, 'r') as f:
        det_ctx = json.load(f)
    print(f"   Number of trajectories: {len(det_ctx)}")
    first_key = list(det_ctx.keys())[0]
    first_traj = det_ctx[first_key]
    print(f"   First key: {first_key}")
    print(f"   First traj detections_per_frame len: {len(first_traj.get('detections_per_frame', []))}")
    if first_traj.get('detections_per_frame'):
        first_frame = first_traj['detections_per_frame'][0]
        print(f"   First frame detections: {len(first_frame)}")
        if len(first_frame) > 0:
            print(f"   First detection: {first_frame[0][:4]}...")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
If detection_context.json has data but dataset.pt has empty contexts,
the dataset.pt needs to be regenerated with proper context loading.

To regenerate, run training_script.py with the dataset generation mode.
""")
