#!/usr/bin/env bash
#SBATCH --job-name=hybridtrack_train
#SBATCH --partition=COOP
#SBATCH --output=hybridtrack_train.%j.out
#SBATCH --error=hybridtrack_train.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090:1
#SBATCH --mem=80G

set -euxo pipefail

export TORCH_CUDA_ARCH_LIST="8.0"
export CUDAARCH=80

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load TensorFlow/2.13.0-foss-2023a

HYBRIDTRACK_VENV=~/hybridtrack_py311/venv
if [ ! -d "$HYBRIDTRACK_VENV" ]; then
    python3 -m venv "$HYBRIDTRACK_VENV"
fi
source "$HYBRIDTRACK_VENV/bin/activate"
python --version

cd ~/HybridTrack

# Ensure requirements are installed (no-op if already done)
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install easyconfig

# Comprehensive CUDA check
echo "=== CUDA Environment Check ==="
nvidia-smi || echo "nvidia-smi not available"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "SLURM_LOCALID: $SLURM_LOCALID"
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Current CUDA device:', torch.cuda.current_device())
    for i in range(torch.cuda.device_count()):
        print(f'CUDA device {i}: {torch.cuda.get_device_name(i)}')
else:
    print('No CUDA devices detected')
"
echo "=== End CUDA Check ==="

# Launch training (with fallback to CPU if GPU fails)
python src/training_script.py --cfg src/configs/training.yaml || {
    echo "GPU training failed, attempting CPU fallback..."
    FORCE_CPU=true python src/training_script.py --cfg src/configs/training.yaml
}