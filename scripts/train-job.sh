#!/bin/bash
#SBATCH --job-name=ip2p-continuous
#SBATCH --account=mds_team
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -e

cd "$SLURM_SUBMIT_DIR"

# Load Mamba
module load mamba

# Activate environment
source activate image-editor

# Verify GPU
nvidia-smi
python -c "import torch; assert torch.cuda.is_available()"

# Train
python -u train.py --config configs/continuous.yaml