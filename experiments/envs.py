from typing import Tuple

import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from stable_baselines3.common.monitor import Monitor


def get_pendulum_env() -> Tuple[gym.Env, gym.Env]:
    train_env = PixelObservationWrapper(
        gym.make("Pendulum-v1", render_mode="rgb_array")
    )
    eval_env = Monitor(
        PixelObservationWrapper(gym.make("Pendulum-v1", render_mode="rgb_array"))
    )
    return train_env, eval_env


def get_cheetah_env() -> Tuple[gym.Env, gym.Env]:
    train_env = PixelObservationWrapper(
        gym.make("HalfCheetah-v4", render_mode="rgb_array")
    )
    eval_env = Monitor(
        PixelObservationWrapper(gym.make("HalfCheetah-v4", render_mode="rgb_array"))
    )
    return train_env, eval_env


def get_carracer_env() -> Tuple[gym.Env, gym.Env]:
    train_env = FrameStack(GrayScaleObservation(gym.make("CarRacing-v2")), 4)
    eval_env = Monitor(
        FrameStack(
            GrayScaleObservation(gym.make("CarRacing-v2", render_mode="rgb_array")), 4
        )
    )
    return train_env, eval_env


def get_lunarlander_env() -> Tuple[gym.Env, gym.Env]:
    train_env = gym.make("LunarLander-v2", continuous=True)
    eval_env = Monitor(
        gym.make("LunarLander-v2", continuous=True, render_mode="rgb_array")
    )
    return train_env, eval_env


def get_env(name: str) -> Tuple[gym.Env, gym.Env]:
    if name == "pendulum":
        return get_pendulum_env()
    elif name == "cheetah":
        return get_cheetah_env()
    elif name == "carracer":
        return get_carracer_env()
    elif name == "lunarlander":
        return get_lunarlander_env()
    else:
        raise ValueError(f"Unknown environment: {name}")