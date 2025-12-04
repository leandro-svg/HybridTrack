#!/usr/bin/env bash
#SBATCH --job-name=check_ctx
#SBATCH --partition=COOP
#SBATCH --output=check_ctx.%j.out
#SBATCH --error=check_ctx.%j.err
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
import torch
import sys
sys.path.insert(0, 'src')

print("="*80)
print("CHECKING IF TRAINING DATA HAD DETECTION CONTEXT")
print("="*80)

dataset_path = 'src/data/checkpoints/dataset.pt'
print(f"\nLoading: {dataset_path}")
data = torch.load(dataset_path, map_location='cpu')

print(f"\nData structure:")
print(f"  Type: {type(data)}")
print(f"  Length: {len(data)}")

# Expected format: [train_input, train_target, cv_input, cv_target, train_history, cv_history, 
#                   train_context, train_history_context, cv_context, cv_history_context]

if len(data) >= 7:
    train_context = data[6]
    print(f"\n  data[6] (train_context):")
    if train_context is not None:
        print(f"    Type: {type(train_context)}")
        if isinstance(train_context, list):
            print(f"    Length: {len(train_context)}")
            non_empty = 0
            for i, ctx in enumerate(train_context):
                if ctx is not None and len(ctx) > 0:
                    non_empty += 1
                    if non_empty <= 3:
                        print(f"    Sample {i}: {len(ctx)} frames")
                        if len(ctx) > 0 and ctx[0] is not None:
                            if hasattr(ctx[0], 'shape'):
                                print(f"      Frame 0 shape: {ctx[0].shape}")
                                if ctx[0].numel() > 0:
                                    print(f"      First detection: {ctx[0][0]}")
            print(f"    Total non-empty: {non_empty}/{len(train_context)}")
    else:
        print("    None")
else:
    print(f"  Data has only {len(data)} elements (expected >= 7 for context)")

if len(data) >= 9:
    cv_context = data[8]
    print(f"\n  data[8] (cv_context):")
    if cv_context is not None:
        print(f"    Type: {type(cv_context)}")
        if isinstance(cv_context, list):
            print(f"    Length: {len(cv_context)}")
            non_empty = sum(1 for ctx in cv_context if ctx is not None and len(ctx) > 0)
            print(f"    Total non-empty: {non_empty}/{len(cv_context)}")
    else:
        print("    None")

print("\n" + "="*80)
print("Inspecting first 10 elements of data tuple:")
print("="*80)
for i, item in enumerate(data[:min(len(data), 10)]):
    if item is None:
        print(f"  data[{i}]: None")
    elif isinstance(item, torch.Tensor):
        print(f"  data[{i}]: Tensor shape {item.shape}")
    elif isinstance(item, list):
        print(f"  data[{i}]: List len {len(item)}")
        if len(item) > 0:
            first = item[0]
            if isinstance(first, torch.Tensor):
                print(f"    First element: Tensor shape {first.shape}")
            elif isinstance(first, list):
                print(f"    First element: List len {len(first)}")
            elif isinstance(first, dict):
                print(f"    First element: Dict keys {list(first.keys())[:5]}")
            else:
                print(f"    First element: {type(first)}")
    else:
        print(f"  data[{i}]: {type(item)}")

print("\nDONE")
PYTHON_SCRIPT
