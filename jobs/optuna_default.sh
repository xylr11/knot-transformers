#!/bin/bash
#SBATCH --job-name=test_optuna
#SBATCH --output=logs/job_test_optuna.out
#SBATCH --error=logs/job_test_optuna.err
#SBATCH --time=12:00:00
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

python optuna_train.py --config configs/optuna_default.json --save_dir states/ --trials 10