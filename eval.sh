#!/bin/bash
#SBATCH --job-name=tdmpc2_uncertainty_chase
#SBATCH --output=result.txt
#SBATCH --partition=job
#SBATCH --time=48:00:00  
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1  # Request GPU

# Load the Conda environment
source ~/.bashrc
conda activate shuo_1

# Run a Python script using Conda environment
python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/uncertainty_chase_only_value/1111/default/models/final.pt