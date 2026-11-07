#!/bin/bash
#SBATCH --job-name=ml-train
#SBATCH --account=mds_team
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm-%j.out

set -e

cd "$SLURM_SUBMIT_DIR"

module load python
source pytorch_tutorial/bin/activate

# Verify GPU
nvidia-smi
python -c "import torch; assert torch.cuda.is_available()"

# Train
python -u train.py --config configs/baseline.yaml