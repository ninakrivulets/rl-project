from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import trange

from .agent import SRSACAgent
from .config import load_config
from .envs import get_env_specs, make_env
from .evaluation import evaluate_policy
from .replay_buffer import ReplayBuffer
from .reporting import save_curve_plot
from .utils import make_run_dir, save_json, seed_everything, select_device


def train_from_config(config_path, overrides=None, project_root=None):
    config = load_config(config_path, overrides)
    project_root = Path(project_root or Path(config_path).resolve().parents[1])
    run_dir = make_run_dir(project_root, config)
    device = select_device(config.get("device", "auto"))
    seed_everything(config["seed"])
    print('name', config["experiment"]["name"])
    print('env id', config["env"]["id"])
    print('total_env_steps', config["train"]["total_env_steps"])
    print("replay_ratio", config["algorithm"]["updates_per_step"])
    print("resets_enabled", config["algorithm"]["resets"])
    

    env = make_env(config["env"], seed=config["seed"])
    specs = get_env_specs(env)
    agent = SRSACAgent(specs, config["algorithm"], device)
    replay_buffer = ReplayBuffer(
        specs["obs_dim"], specs["action_dim"], config["algorithm"]["replay_size"]
    )

    save_json(run_dir / "resolved_config.json", config)

    observation, _ = env.reset(seed=config["seed"])
    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    best_eval = float("-inf")
    updates_since_reset = 0
    reset_events = []
    evaluations = []
    episodes = []
    last_update = {}

    loop = trange(1, config["train"]["total_env_steps"] + 1, desc=config["experiment"]["name"])
    for env_step in loop:
        if env_step <= config["train"]["start_steps"]:
            action = env.action_space.sample()
        else:
            action = agent.sample_action(observation, explore=True)

        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add(observation, action, reward, next_observation, float(done))

        observation = next_observation
        episode_return += reward
        episode_length += 1

        if done:
            episodes.append(
                {
                    "episode": episode_index,
                    "env_step": env_step,
                    "return": float(episode_return),
                    "length": int(episode_length),
                }
            )
            episode_index += 1
            observation, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        if env_step >= config["train"]["start_training_after"] and len(replay_buffer) >= config["train"]["batch_size"]:
            losses = []
            for _ in range(config["algorithm"]["updates_per_step"]):
                batch = replay_buffer.sample(config["train"]["batch_size"])
                losses.append(agent.update(batch))
                updates_since_reset += 1
            last_update = {
                key: sum(item[key] for item in losses) / len(losses)
                for key in losses[0]
            }

        if env_step % config["train"]["eval_every"] == 0 or env_step == config["train"]["total_env_steps"]:
            evaluation = evaluate_policy(
                agent,
                config["env"],
                config["seed"],
                config["train"]["eval_episodes"],
            )
            best_eval = max(best_eval, evaluation["mean_return"])
            eval_row = {
                "env_step": env_step,
                "mean_return": evaluation["mean_return"],
                "std_return": evaluation["std_return"],
                "best_return": best_eval,
                "updates_since_reset": updates_since_reset,
                "resets_done": len(reset_events),
            }
            eval_row.update(last_update)
            evaluations.append(eval_row)
            loop.set_postfix(
                eval=f"{evaluation['mean_return']:.1f}",
                best=f"{best_eval:.1f}",
                alpha=f"{last_update.get('alpha', agent.alpha.item()):.3f}",
            )
            if config["algorithm"]["resets"] and updates_since_reset >= config["algorithm"]["reset_interval_updates"]:
                agent.reset()
                updates_since_reset = 0
                reset_events.append({"env_step": env_step, "kind": "full_agent_reset"})

    env.close()

    video = evaluate_policy(
        agent,
        config["env"],
        config["seed"],
        config["train"]["video_episodes"],
        video_path=run_dir / "final_episode.mp4",
    )

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(agent.state_dict(), checkpoint_path)

    pd.DataFrame(evaluations).to_csv(run_dir / "evaluations.csv", index=False)
    pd.DataFrame(episodes).to_csv(run_dir / "training_episodes.csv", index=False)
    pd.DataFrame(reset_events).to_csv(run_dir / "resets.csv", index=False)
    save_curve_plot(evaluations, run_dir / "learning_curve.png")

    summary = {
        "run_name": config["experiment"]["name"],
        "env_id": config["env"]["id"],
        "seed": config["seed"],
        "device": device,
        "total_env_steps": config["train"]["total_env_steps"],
        "replay_ratio": config["algorithm"]["updates_per_step"],
        "resets_enabled": config["algorithm"]["resets"],
        "reset_interval_updates": config["algorithm"]["reset_interval_updates"],
        "final_eval_return": evaluations[-1]["mean_return"],
        "best_eval_return": best_eval,
        "final_eval_std": evaluations[-1]["std_return"],
        "video_return": video["mean_return"],
        "num_resets": len(reset_events),
        "run_dir": str(run_dir),
        "video_path": str(run_dir / "final_episode.mp4"),
        "curve_path": str(run_dir / "learning_curve.png"),
        "checkpoint_path": str(checkpoint_path),
    }
    save_json(run_dir / "summary.json", summary)
    return summary
