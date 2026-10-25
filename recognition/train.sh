#!/bin/bash
#SBATCH --job-name=improved3d
#SBATCH --partition=largecpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/improved_%j.out
#SBATCH --error=logs/cpu_smoke_%j.err


source $HOME/miniconda3/bin/activate
conda activate torch
cd ~/PatternAnalysis-2026/recognition
# activate your Python environment here

python train.py --data-root /home/groups/comp3710/HipMRI_Study_open \
    --model improved --out-dir runs/improved