#!/bin/bash
#SBATCH --job-name=improved3d
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/improved_%j.out

cd ~/PatternAnalysis-2026/recognition/<your-folder>
# activate your Python environment here

python train.py --data-root /home/groups/comp3710/HipMRI_Study_open \
    --model improved --out-dir runs/improved