import importlib
import json
from pathlib import Path

def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def get_class(module_path, class_name):
    """Dynamically import and return a class given its module path and class name."""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)