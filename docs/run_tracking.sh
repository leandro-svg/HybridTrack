#!/usr/bin/env bash
#SBATCH --job-name=hybridtrack_tracking
#SBATCH --partition=COOP
#SBATCH --output=hybridtrack_tracking.%j.out
#SBATCH --error=hybridtrack_tracking.%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4090:1
#SBATCH --mem=48G

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

# Ensure requirements are installed
pip install --upgrade pip
pip install -r requirements.txt

# Optional: verify GPU access quickly
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"

# Launch tracking
python src/run_tracking.py --cfg_file src/configs/tracking.yaml