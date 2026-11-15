#!/bin/bash
#SBATCH --job-name=ip2p-sd-poc
#SBATCH --account=mds_team
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:30:00
#SBATCH --output=outputs/slurm-%j.out

set -e

cd "$SLURM_SUBMIT_DIR"

if [ -n "${HF_HOME:-}" ]; then
    mkdir -p "$HF_HOME"
fi
mkdir -p outputs

module load python
source pytorch_tutorial/bin/activate

# Verify GPU
nvidia-smi
python -c "import torch; assert torch.cuda.is_available()"

# Train
python -u train.py --config configs/baseline.yaml "$@"
