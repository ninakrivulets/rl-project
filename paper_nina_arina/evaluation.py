from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .envs import make_env


def evaluate_policy(agent, env_config, seed, episodes, video_path=None):
    render_mode = "rgb_array" if video_path else None
    env = make_env(env_config, seed=seed, render_mode=render_mode)
    returns = []
    frames = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + episode)
        done = False
        total_reward = 0.0
        if video_path and episode == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        while not done:
            action = agent.sample_action(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if video_path and episode == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
        returns.append(total_reward)

    if video_path and frames:
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fps = env.metadata.get("render_fps", 30)
        imageio.mimsave(video_path, frames, fps=fps, macro_block_size=1)

    env.close()
    returns = np.asarray(returns, dtype=np.float32)
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "returns": returns.tolist(),
        "episodes": int(episodes),
        "video_path": str(video_path) if video_path else "",
    }
