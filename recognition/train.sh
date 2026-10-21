   #!/bin/bash
   #SBATCH --job-name=smoke3d
   #SBATCH --partition=a100-test
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=4
   #SBATCH --mem=32G
   #SBATCH --time=00:30:00
   #SBATCH --output=logs/smoke_%j.out

   cd ~/PatternAnalysis-2026/recognition/logs
   # activate your Python environment here
   python train.py --data-root /home/groups/comp3710/HipMRI_Study_open \
       --epochs 2 --max-train 4 --out-dir runs/smoke