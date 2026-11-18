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

# Sample utilization and memory use throughout the job for later analysis.
gpu_log="outputs/gpu-${SLURM_JOB_ID}.csv"
nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv \
    --loop=10 \
    --filename="$gpu_log" &
gpu_monitor_pid=$!
cleanup_gpu_monitor() {
    kill "$gpu_monitor_pid" 2>/dev/null || true
    wait "$gpu_monitor_pid" 2>/dev/null || true
}
trap cleanup_gpu_monitor EXIT
echo "GPU telemetry: $gpu_log"

# Train
train_config="configs/baseline.yaml"
if [ "${1:-}" = "--config" ]; then
    if [ "$#" -lt 2 ]; then
        echo "--config requires a YAML path" >&2
        exit 2
    fi
    train_config="$2"
    shift 2
fi
python -u train.py --config "$train_config" "$@"
