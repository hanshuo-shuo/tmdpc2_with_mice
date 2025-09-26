#!/bin/bash
#SBATCH --job-name=rerere
#SBATCH --output=result_new_ptsb.txt
#SBATCH --partition=job
#SBATCH --time=48:00:00  
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1  # Request GPU

# Load the Conda environment
source ~/.bashrc
conda activate shuo_1

# Run a Python script using Conda environment
# python train.py seed=1234 ptsb=true
# python train.py seed=7654 ptsb=true
# python train.py seed=2345 ptsb=true
# python train.py seed=6543 ptsb=true
# python train.py seed=1234
# python train.py seed=2345
# python train.py seed=3456
# python train.py seed=4567
# python train.py seed=5678

# python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/cellworld_d2ac_baseline_tdmpc/5432/normaltrain/models/final.pt save_name=5432_21_05 seed=None world_name="21_05"

# python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/cellworld_d2ac_baseline_tdmpc/5432/normaltrain/models/final.pt save_name=5432_19_09 seed=None world_name="19_09"

# python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/surprise/2755/re/models/final.pt

python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/alex_exp/1234/spread_exp/models/final.pt seed=1234 save_name=1234_spread_exp world_name="21_05"
python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/alex_exp/2345/spread_exp/models/final.pt seed=2345 save_name=2345_spread_exp world_name="21_05"
python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/alex_exp/3456/spread_exp/models/final.pt seed=3456 save_name=3456_spread_exp world_name="21_05"
python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/alex_exp/4567/spread_exp/models/final.pt seed=4567 save_name=4567_spread_exp world_name="21_05"
python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/alex_exp/5678/spread_exp/models/final.pt seed=5678 save_name=5678_spread_exp world_name="21_05"

python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/alex_exp/1234/cluster_exp/models/final.pt seed=1234 save_name=1234_cluster_exp world_name='"clump01_05"'
python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/alex_exp/2345/cluster_exp/models/final.pt seed=2345 save_name=2345_cluster_exp world_name='"clump01_05"'
python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/alex_exp/3456/cluster_exp/models/final.pt seed=3456 save_name=3456_cluster_exp world_name='"clump01_05"'
python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/alex_exp/4567/cluster_exp/models/final.pt seed=4567 save_name=4567_cluster_exp world_name='"clump01_05"'
python evaluate.py checkpoint=/home/shv7753/tlppo_cellworld_evade/logs/alex_exp/5678/cluster_exp/models/final.pt seed=5678 save_name=5678_cluster_exp world_name='"clump01_05"'


