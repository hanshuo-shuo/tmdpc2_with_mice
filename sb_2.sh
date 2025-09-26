#!/bin/bash
#SBATCH --job-name=rerere
#SBATCH --output=result_new_baseline.txt
#SBATCH --partition=job
#SBATCH --time=48:00:00  
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1  # Request GPU

# Load the Conda environment
source ~/.bashrc
conda activate shuo_1

# Run a Python script using Conda environment
python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/cellworld_d2ac_baseline_tdmpc/1234/normaltrain/models/final.pt save_name=1234 seed=None

python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/cellworld_d2ac_baseline_tdmpc/2345/normaltrain/models/final.pt save_name=2345 seed=None

python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/cellworld_d2ac_baseline_tdmpc/5432/normaltrain/models/final.pt save_name=5432 seed=None

python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/cellworld_d2ac_baseline_tdmpc/6543/normaltrain/models/final.pt save_name=6543 seed=None

python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/cellworld_d2ac_baseline_tdmpc/7654/normaltrain/models/final.pt save_name=7654 seed=None

# python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/surprise/2755/re/models/final.pt