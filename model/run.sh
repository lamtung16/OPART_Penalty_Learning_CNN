#!/bin/bash
#SBATCH --array=0-2267
#SBATCH --time=96:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/err-%A_%a.out
#SBATCH --job-name=cnn

python run_one_model.py $SLURM_ARRAY_TASK_ID