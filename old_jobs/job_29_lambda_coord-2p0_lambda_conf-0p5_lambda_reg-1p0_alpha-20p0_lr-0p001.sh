#!/bin/bash
#SBATCH --job-name=gs_29
#SBATCH --output=logs/job_29_lambda_coord-2p0_lambda_conf-0p5_lambda_reg-1p0_alpha-20p0_lr-0p001.out
#SBATCH --error=logs/job_29_lambda_coord-2p0_lambda_conf-0p5_lambda_reg-1p0_alpha-20p0_lr-0p001.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=kylecm11@byu.edu
#SBATCH --mail-type=FAIL

module load python/3.12
module load cuda
module load cudnn

source .venv/bin/activate

python train.py --config configs/config_29_lambda_coord-2p0_lambda_conf-0p5_lambda_reg-1p0_alpha-20p0_lr-0p001.json --save states/best_model_29_lambda_coord-2p0_lambda_conf-0p5_lambda_reg-1p0_alpha-20p0_lr-0p001.pt
