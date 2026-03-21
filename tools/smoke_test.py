import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from paper_nina_arina.train import train_from_config


def main():
    project_root = Path(__file__).resolve().parents[1]
    summary = train_from_config(project_root / "configs" / "smoke_test.json", project_root=project_root)
    if not Path(summary["video_path"]).exists():
        raise RuntimeError("Smoke test did not produce video")
    if not Path(summary["checkpoint_path"]).exists():
        raise RuntimeError("Smoke test did not produce checkpoint")
    print("Smoke test OK")
    print(summary)


if __name__ == "__main__":
    main()
