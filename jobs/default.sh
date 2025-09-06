#!/bin/bash
#SBATCH --job-name=large_default
#SBATCH --output=logs/job_large_default.out
#SBATCH --error=logs/job_large_default.err
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=kylecm11@byu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load python/3.12
module load cuda
module load cudnn

source .venv/bin/activate

python train.py --config configs/default.json --save states/large_hungarian.pt