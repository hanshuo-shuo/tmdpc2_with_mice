#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=result.txt
#SBATCH --partition=job
#SBATCH --time=24:00:00  # 1 hour
#SBATCH --cpus-per-task=1


# Load the Conda environment
source ~/.bashrc
conda activate shuo_1

# Run a Python script using Conda environment
python train.py