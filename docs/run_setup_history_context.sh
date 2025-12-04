#!/usr/bin/env bash
#SBATCH --job-name=setup_history_context
#SBATCH --partition=COOP
#SBATCH --output=setup_history_context.%j.out
#SBATCH --error=setup_history_context.%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euxo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0

HYBRIDTRACK_VENV=/FARM/ldibella/hybridtrack_py311/venv
if [ ! -d "$HYBRIDTRACK_VENV" ]; then
    python3 -m venv "$HYBRIDTRACK_VENV"
fi
source "$HYBRIDTRACK_VENV/bin/activate"
python --version

cd /FARM/ldibella/hybridtrack_original

# Ensure required packages are installed
pip install --upgrade pip
pip install numpy scipy tqdm

# Run the history context extraction
echo "=== Starting history context extraction ==="
cd docs/data_utils
python setup_history_context.py
echo "=== History context extraction complete ==="

# Show results
echo ""
echo "=== Generated files ==="
ls -lh /FARM/ldibella/hybridtrack_original/src/data/ann/train/history_context.json 2>/dev/null || echo "train/history_context.json not found"
ls -lh /FARM/ldibella/hybridtrack_original/src/data/ann/validation/history_context.json 2>/dev/null || echo "validation/history_context.json not found"
