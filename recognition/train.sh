#!/bin/bash
#SBATCH --job-name=improved3d
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --output=logs/cpu_smoke_%j.out
#SBATCH --error=logs/cpu_smoke_%j.err


source $HOME/miniconda3/bin/activate
conda activate torch
cd ~/PatternAnalysis-2026/recognition

python -u train.py --data-root /home/groups/comp3710/HipMRI_Study_open \
    --model baseline --out-dir runs/baseline \
    --split-file runs/improved/splits.json