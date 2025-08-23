#!/bin/bash
#SBATCH --job-name=default
#SBATCH --output=logs/job_default.out
#SBATCH --error=logs/job_default.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load python/3.12
module load cuda/13.0
module load cudnn/9.12.0

python train.py --config configs/default.json