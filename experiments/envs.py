from typing import Tuple

import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecMonitor


def get_pendulum_env() -> Tuple[GymEnv, GymEnv]:
    train_env = PixelObservationWrapper(
        gym.make("Pendulum-v1", render_mode="rgb_array")
    )
    eval_env = Monitor(
        PixelObservationWrapper(gym.make("Pendulum-v1", render_mode="rgb_array"))
    )
    return train_env, eval_env


def get_cheetah_env() -> Tuple[GymEnv, GymEnv]:
    train_env = PixelObservationWrapper(
        gym.make("HalfCheetah-v4", render_mode="rgb_array")
    )
    eval_env = Monitor(
        PixelObservationWrapper(gym.make("HalfCheetah-v4", render_mode="rgb_array"))
    )
    return train_env, eval_env


def get_carracer_env() -> Tuple[GymEnv, GymEnv]:
    train_env = FrameStack(GrayScaleObservation(gym.make("CarRacing-v2")), 4)
    eval_env = Monitor(
        FrameStack(
            GrayScaleObservation(gym.make("CarRacing-v2", render_mode="rgb_array")), 4
        )
    )
    return train_env, eval_env


def get_lunarlander_env(n_parallel: int) -> Tuple[GymEnv, GymEnv]:
    train_env = make_vec_env(
        "LunarLander-v2", n_envs=n_parallel, env_kwargs={"continuous": True}
    )
    eval_env = make_vec_env(
        "LunarLander-v2",
        n_envs=8,
        env_kwargs={"continuous": True, "render_mode": "rgb_array"},
    )

    return train_env, eval_env


def get_env(name: str, n_parallel: int) -> Tuple[GymEnv, GymEnv]:
    if name == "pendulum":
        return get_pendulum_env()
    elif name == "cheetah":
        return get_cheetah_env()
    elif name == "carracer":
        return get_carracer_env()
    elif name == "lunarlander":
        return get_lunarlander_env(n_parallel)
    else:
        raise ValueError(f"Unknown environment: {name}")
