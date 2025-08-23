#!/bin/bash
#SBATCH --job-name=default
#SBATCH --output=logs/job_default.out
#SBATCH --error=logs/job_default.err
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=kylecmarkham@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load python/3.12
module load cuda/13.0
module load cudnn/9.12.0

source .venv/bin/activate

python train.py --config configs/default.json --save states/best_model_default.pt