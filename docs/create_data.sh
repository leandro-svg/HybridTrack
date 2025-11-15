#!/usr/bin/env bash
#SBATCH --job-name=hybridtrack_data
#SBATCH --partition=COOP
#SBATCH --output=hybridtrack_data.%j.out
#SBATCH --error=hybridtrack_data.%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4090:0
#SBATCH --mem=32G

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
pip install --upgrade pip
pip install -r requirements.txt
pip install gdown

# === Paths ===
PROJECT_ROOT=~/HybridTrack
DATA_ROOT="$PROJECT_ROOT/src/data"
KITTI_LINK_TARGET="$DATA_ROOT/KITTI"   # symlink or directory expected by docs

mkdir -p "$DATA_ROOT"

echo "NOTE: Make sure KITTI raw tracking data is available and linked to $KITTI_LINK_TARGET"
echo "If not yet done, you can create a symlink like:"
echo "  ln -s /path/to/KITTI $KITTI_LINK_TARGET"

# === 1. Download pose data (single file) ===
# From docs/create_data.md:
# https://drive.google.com/file/d/1o-ay2FhlOEnKFmqXMWnbH7e6gId8L7P9/view?usp=sharing
POSE_ID="1o-ay2FhlOEnKFmqXMWnbH7e6gId8L7P9"
POSE_OUT_DIR="$KITTI_LINK_TARGET/tracking"
mkdir -p "$POSE_OUT_DIR"
cd "$POSE_OUT_DIR"

# This will create something like pose.zip or similar; you may need to adjust name.
gdown --id "$POSE_ID" -O pose_data.zip
unzip -o pose_data.zip
rm -f pose_data.zip

DET_DRIVE_LINK="https://drive.google.com/file/d/1w1ZadwgWgWRr6Le5wZNCkAh3bdIuoqvB/view?usp=drive_link"
DET_ROOT="/FARM/ldibella/HybridTrack/src/data/KITTI/tracking/detections"
mkdir -p "$DET_ROOT"
cd "$DET_ROOT"

gdown --folder --remaining-ok "$DET_DRIVE_LINK" -O virconv

if [ -d "virconv" ]; then
    echo "virconv folder already named correctly"
else
    echo "If folder name from Drive is different, rename it to 'virconv' as expected by docs."
fi

# === 3. Generate annotations ===
cd "$PROJECT_ROOT"
python docs/data_utils/setup_trajectory.py