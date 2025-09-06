#!/bin/bash
#SBATCH --job-name=gs_5
#SBATCH --output=logs/job_5_lambda_coord-1p0_lambda_conf-1p0_lambda_reg-1p0.out
#SBATCH --error=logs/job_5_lambda_coord-1p0_lambda_conf-1p0_lambda_reg-1p0.err
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

python train.py --config configs/config_5_lambda_coord-1p0_lambda_conf-1p0_lambda_reg-1p0.json --save states/best_model_5_lambda_coord-1p0_lambda_conf-1p0_lambda_reg-1p0.pt
