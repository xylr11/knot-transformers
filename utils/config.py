import json
from pathlib import Path

def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def get_loss(name, params):
    module = __import__(f"losses.{name}", fromlist=[""])
    return module.Loss(**params)

def get_model(name, params):
    module = __import__(f"models.{name}", fromlist=[""])
    return module.Model(**params)