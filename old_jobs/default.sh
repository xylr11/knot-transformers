#!/bin/bash
#SBATCH --job-name=test_hungarian
#SBATCH --output=logs/job_test_hungarian.out
#SBATCH --error=logs/job_test_hungarian.err
#SBATCH --time=04:00:00
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

python train.py --config configs/default.json --save states/test_hungarian.pt