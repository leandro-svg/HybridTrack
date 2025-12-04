#!/usr/bin/env bash
#SBATCH --job-name=debug_collate
#SBATCH --partition=COOP
#SBATCH --output=debug_collate.%j.out
#SBATCH --error=debug_collate.%j.err
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

python inspect_gate_value.py