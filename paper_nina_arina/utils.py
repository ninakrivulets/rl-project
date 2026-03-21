import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def make_run_dir(project_root, config):
    run_name = config["experiment"]["name"]
    seed = config["seed"]
    output_root = Path(project_root) / config["experiment"].get("output_root", "runs")
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"{run_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)

