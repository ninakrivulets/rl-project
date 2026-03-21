import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


def save_curve_plot(evaluations, path):
    if not evaluations:
        return
    df = pd.DataFrame(evaluations)
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["env_step"], df["mean_return"], marker="o", linewidth=2)
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title("SR-SAC learning curve")
    plt.grid(alpha=0.3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
