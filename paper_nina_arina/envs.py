import os
from pathlib import Path

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation


def _register_dm_control():
    try:
        import shimmy
    except ImportError as e:
        raise RuntimeError(
            "For dm_control environments install shimmy[dm-control]"
        ) from e
    try:
        gym.register_envs(shimmy)
    except Exception:
        pass


def make_env(env_config, seed=None, render_mode=None):
    env_id = env_config["id"]
    kwargs = dict(env_config.get("kwargs", {}))
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    if env_id.startswith("dm_control/"):
        _register_dm_control()
    env = gym.make(env_id, **kwargs)
    print("env:", env)
    print("env.observation_space:", getattr(env, "observation_space", None))
    
    obs_space = getattr(env, "observation_space", None)
    shape = getattr(obs_space, "shape", None)
    print("shape:", shape)
    if not hasattr(env.observation_space, "shape") or len(env.observation_space.shape) != 1:
        env = FlattenObservation(env)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    return env


def get_env_specs(env):
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("This project supports only continuous Box action spaces")
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("This project expects flat Box observations")
    return {
        "obs_dim": int(np.prod(env.observation_space.shape)),
        "action_dim": int(np.prod(env.action_space.shape)),
        "action_low": env.action_space.low.astype(np.float32),
        "action_high": env.action_space.high.astype(np.float32),
    }
