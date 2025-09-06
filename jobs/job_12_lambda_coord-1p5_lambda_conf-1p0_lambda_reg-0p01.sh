#!/bin/bash
#SBATCH --job-name=gs_12
#SBATCH --output=logs/job_12_lambda_coord-1p5_lambda_conf-1p0_lambda_reg-0p01.out
#SBATCH --error=logs/job_12_lambda_coord-1p5_lambda_conf-1p0_lambda_reg-0p01.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=kylecm11@byu.edu
#SBATCH --mail-type=FAIL

module load python/3.12
module load cuda
module load cudnn

source .venv/bin/activate

python train.py --config configs/config_12_lambda_coord-1p5_lambda_conf-1p0_lambda_reg-0p01.json --save states/best_model_12_lambda_coord-1p5_lambda_conf-1p0_lambda_reg-0p01.pt
