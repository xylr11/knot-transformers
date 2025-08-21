import itertools
import os
import json

# Grid values
param_grid = {
    "lambda_coord": [1.0],
    "lambda_conf": [0.1],
    "lambda_reg": [0.1],
    "alpha": [10.0],
}

# Where to save jobs/configs
job_dir = "jobs"
os.makedirs(job_dir, exist_ok=True)

# Base SLURM header
slurm_header = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.out
#SBATCH --error=logs/{job_name}.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load cuda/11.8  # UPDATE TO CORRECT MODULES
source ~/envs/myenv/bin/activate  # UPDATE TO CORRECT VENV PATH
"""

# Iterate through all hyperparameter combinations
keys = list(param_grid.keys())
for values in itertools.product(*param_grid.values()):
    params = dict(zip(keys, values))

    # Unique name for job
    job_name = "gs_" + "_".join(f"{k}{v}" for k, v in params.items())

    # Config file
    config_path = os.path.join(job_dir, f"{job_name}.json")
    with open(config_path, "w") as f:
        json.dump(params, f, indent=2)

    # Slurm script
    script_path = os.path.join(job_dir, f"{job_name}.sh")
    with open(script_path, "w") as f:
        f.write(slurm_header.format(job_name=job_name))
        f.write(f"\npython train.py --config {config_path}\n")

print(f"Generated {len(list(itertools.product(*param_grid.values())))} jobs in {job_dir}/")
