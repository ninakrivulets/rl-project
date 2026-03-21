import argparse
from pathlib import Path

import pandas as pd

from paper_nina_arina.train import train_from_config
from paper_nina_arina.utils import load_json


def build_tables(project_root, plan, summaries):
    refs = load_json(project_root / "article_refs.json")
    summary_by_name = {item["run_name"]: item for item in summaries}

    run_rows = []
    for summary in summaries:
        run_rows.append(summary)
    runs_df = pd.DataFrame(run_rows).sort_values("run_name")

    experiment_rows = []
    for experiment in plan["experiments"]:
        for row in experiment["rows"]:
            summary = summary_by_name[row["run"]]
            article_return = None
            article_url = ""
            if row.get("article_ref"):
                reference = refs[row["article_ref"]]
                article_return = reference["values"][str(row["article_step"])]
                article_url = reference["source_url"]
            experiment_rows.append(
                {
                    "experiment": experiment["name"],
                    "why": experiment["why"],
                    "label": row["label"],
                    "run_name": summary["run_name"],
                    "env_id": summary["env_id"],
                    "replay_ratio": summary["replay_ratio"],
                    "resets_enabled": summary["resets_enabled"],
                    "total_env_steps": summary["total_env_steps"],
                    "final_eval_return": summary["final_eval_return"],
                    "best_eval_return": summary["best_eval_return"],
                    "article_return_at_same_budget": article_return,
                    "article_source_url": article_url,
                    "video_path": summary["video_path"],
                }
            )
    experiments_df = pd.DataFrame(experiment_rows)
    runs_df.to_csv(project_root / "runs" / "selected_runs_summary.csv", index=False)
    experiments_df.to_csv(project_root / "runs" / "selected_experiment_table.csv", index=False)
    try:
        runs_df.to_markdown(project_root / "runs" / "selected_runs_summary.md", index=False)
        experiments_df.to_markdown(project_root / "runs" / "selected_experiment_table.md", index=False)
    except Exception:
        (project_root / "runs" / "selected_runs_summary.md").write_text(
            runs_df.to_string(index=False)
        )
        (project_root / "runs" / "selected_experiment_table.md").write_text(
            experiments_df.to_string(index=False)
        )
    return runs_df, experiments_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="configs/selected_runs.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    plan = load_json(project_root / args.plan)

    summaries = []
    for run in plan["runs"]:
        config_path = project_root / run["config"]
        summary_path = project_root / "runs" / f"{run['name']}_seed0" / "summary.json"
        if summary_path.exists() and not args.force:
            summaries.append(load_json(summary_path))
            continue
        summaries.append(train_from_config(config_path, project_root=project_root))

    runs_df, experiments_df = build_tables(project_root, plan, summaries)
    print(runs_df.to_string(index=False))
    print()
    print(experiments_df.to_string(index=False))


if __name__ == "__main__":
    main()
