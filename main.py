import argparse
from pathlib import Path

from paper_nina_arina.train import train_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    summary = train_from_config(args.config, args.overrides, project_root=project_root)
    print(f"Run saved to {summary['run_dir']}")
    print(f"Final eval return: {summary['final_eval_return']:.2f}")
    print(f"Video: {summary['video_path']}")


if __name__ == "__main__":
    main()
