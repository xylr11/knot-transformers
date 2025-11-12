import os
import itertools
import json

def generate_configs_jobs(): 
    # Maybe make reference default.json instead
    base_config = {
        "model": "DeepSets",
        "loss": "HungarianLoss",
        "model_params": {
            "in_dim": 2,
            "hidden_dim": 512,
            "num_outputs": 50
        },
        "loss_params": {
            "lambda_coord": 10.0,
            "lambda_conf": 0.01,
            "lambda_unmatched": 0.01
            },
        "train_params": {
            "batch_size": 32,
            "num_epochs": 35,
            "lr": 5e-5
        }
    }

    grid = {
        "model_params.hidden_dim": [256, 512],
        "loss_params.lambda_coord": [10.0, 5.0, 1.0],
        "loss_params.lambda_conf": [0.5, 1.0, 1.5],
        "loss_params.lambda_unmatched": [0.01, 0.1]
    }

    os.makedirs("configs", exist_ok=True)
    os.makedirs("jobs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    def set_nested(d, key_path, value):
        keys = key_path.split(".")
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

    param_names = list(grid.keys())
    param_values = list(grid.values())
    all_combinations = list(itertools.product(*param_values))

    for i, combo in enumerate(all_combinations):
        config = json.loads(json.dumps(base_config))

        # Apply hyperparams
        for k, v in zip(param_names, combo):
            set_nested(config, k, v)

        # File-friendly name
        tag = "_".join(f"{k.split('.')[-1]}-{v}".replace(".", "p") for k, v in zip(param_names, combo))

        config_path = f"configs/config_{i}_{tag}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Write job script from default
        job_path = f"jobs/job_{i}_{tag}.sh"
        with open(job_path, "w") as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name=gs_{i}
#SBATCH --output=logs/job_{i}_{tag}.out
#SBATCH --error=logs/job_{i}_{tag}.err
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

python train.py --config {config_path} --save states/best_model_{i}_{tag}.pt --plot plots/plot_{i}_{tag}.png
""")

    print(f"Generated {len(all_combinations)} configs + jobs.")

if __name__ == "__main__":
    generate_configs_jobs()
