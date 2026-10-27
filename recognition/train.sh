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

python compare.py --improved runs/improved/test_results.json \
    --baseline runs/baseline/test_results.json --out-dir comparison