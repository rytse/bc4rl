from pathlib import Path
from typing import Type

import gymnasium as gym
from rl_zoo3 import linear_schedule
from stable_baselines3 import SAC

from bsac import BSAC, BisimConfig
from encoder import CustomCNN, CustomMLP

# REPLAY_BUFFER_SIZE = 200_000
REPLAY_BUFFER_SIZE = 1_000_000

C = 0.5
K = 1.0
GRAD_PENALTY = 10.0

BISIM_BATCH_SIZE = 2048
N_CRITIC_TRAINING_STEPS = 5
N_ENCODER_TRAINING_STEPS = 1


def get_sac(policy_type: str, env: gym.Env, device: str, tb_logdir: Path) -> SAC:
    return SAC(
        policy=policy_type,
        env=env,
        batch_size=256,
        learning_rate=linear_schedule(7.3e-4),
        ent_coef="auto",
        gamma=0.99,
        tau=0.01,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
        buffer_size=REPLAY_BUFFER_SIZE,
        policy_kwargs={
            "features_extractor_class": CustomMLP,
            "net_arch": [400, 300],
            "share_features_extractor": True,
        },
        device=device,
        tensorboard_log=str(tb_logdir),
    )


def get_bsac(policy_type: str, env: gym.Env, device: str, tb_logdir: Path) -> BSAC:
    return BSAC(
        policy=policy_type,
        env=env,
        batch_size=256,
        learning_rate=linear_schedule(7.3e-4),
        gamma=0.99,
        tau=0.01,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
        bisim_config=BisimConfig(
            C=C,
            K=K,
            grad_penalty=GRAD_PENALTY,
            batch_size=BISIM_BATCH_SIZE,
            critic_training_steps=N_CRITIC_TRAINING_STEPS,
            encoder_training_steps=N_ENCODER_TRAINING_STEPS,
        ),
        buffer_size=REPLAY_BUFFER_SIZE,
        policy_kwargs={
            "features_extractor_class": CustomMLP,
            "net_arch": [400, 300],
            "share_features_extractor": True,
        },
        device=device,
        tensorboard_log=str(tb_logdir),
    )


def get_algo(algo: str, policy_type: str, env: gym.Env, device: str, tb_logdir: Path):
    if algo == "sac":
        return get_sac(policy_type, env, device, tb_logdir)
    elif algo == "bsac":
        return get_bsac(policy_type, env, device, tb_logdir)
    else:
        raise ValueError(f"Invalid algo: {algo}")
