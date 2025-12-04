#!/bin/bash
#SBATCH --job-name=hybridtrack_debug
#SBATCH --output=hybridtrack_debug.%j.out
#SBATCH --error=hybridtrack_debug.%j.err
#SBATCH --partition=COOP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

echo "========================================="
echo "EXTENSIVE DEBUG TRACKING RUN"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

# Load environment
source ~/.bashrc
conda activate hybridtrack

# Print versions
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# Tracking with context enabled - extensive debug
echo "========================================="
echo "TRACKING WITH CONTEXT (DEBUG MODE)"
echo "========================================="
cd /FARM/ldibella/hybridtrack_original

# Run only sequence 0000 for quick debug
python src/tracker/track.py \
    --config src/configs/tracking.yaml \
    --dataset kitti \
    --split test \
    --sequences 0000 \
    --output_dir results/debug_context_enabled \
    --verbose

echo ""
echo "========================================="
echo "Job finished at: $(date)"
echo "========================================="
