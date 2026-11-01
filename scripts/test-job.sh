
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=mds_team
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00
#SBATCH --output=test-%j.out

echo "Job started!"
echo "Running on: $(hostname)"
echo "Slurm account: $SLURM_JOB_ACCOUNT"
echo "Job ID: $SLURM_JOB_ID"

sleep 5

echo "Job finished successfully!"
