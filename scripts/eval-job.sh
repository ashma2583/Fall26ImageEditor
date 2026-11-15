#!/bin/bash
#SBATCH --job-name=ip2p-eval
#SBATCH --account=mds_team
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm-%j.out

set -e

cd "$SLURM_SUBMIT_DIR"

module load python
source pytorch_tutorial/bin/activate

# Verify GPU
nvidia-smi
python -c "import torch; assert torch.cuda.is_available()"

# Evaluate, e.g. sbatch scripts/eval-job.sh --config configs/continuous.yaml
python -u evaluate.py "$@"
