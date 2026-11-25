#!/bin/bash
#SBATCH --job-name=ip2p-epoch8-eval
#SBATCH --account=mds_team
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm-%j.out

set -e
cd "$SLURM_SUBMIT_DIR"

project_scratch="/scratch/mds_team_root/mds_team/$USER/Fall26ImageEditor"
export MDST_OUTPUT_ROOT="$project_scratch"
export HF_HOME="$project_scratch/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

module load python
source pytorch_tutorial/bin/activate
python -c "import torch; assert torch.cuda.is_available()"
python -u eval_resolution_guidance.py \
    --config configs/baseline-general256-mild-8ep.yaml \
    --eval-config configs/general256-epoch8-eval.yaml
