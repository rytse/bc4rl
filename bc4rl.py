import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.type_aliases import ReplayBufferSamples


def bisim_loss(
    replay_samples: ReplayBufferSamples,
    observation_space: gym.Space,
    encoder: nn.Module,
    critic: nn.Module,
    c: float,
    K: float,
) -> torch.Tensor:
    """
    Estimates the difference between the L2 norm of the encoded space Z and the bisimulation
    distance as approximated by the critic.

    :param replay_buffer: (ReplayBuffer) the replay buffer
    :param encoder: (nn.Module) the encoder
    :param critic: (nn.Module) the critic
    :param c: (float) the trade-off parameter between reward difference and transition difference
    :param K: (float) the Lipschitz constant of the critic
    :n_samples: (int) the number of samples to use for the estimation

    :return: (torch.Tensor) the bisimulation loss
    """
    obs = replay_samples.observations
    next_obs = replay_samples.next_observations

    pre_obs = preprocess_obs(obs, observation_space)
    pre_next = preprocess_obs(next_obs, observation_space)

    zs = encoder(pre_obs)  # (n_samples, z_dim)
    next_zs = encoder(pre_next)  # (n_samples, z_dim)
    critique = critic(next_zs)  # (n_samples, 1)

    zs_i = zs[:, None, :]
    zs_j = zs[None, :, :]

    critique_i = critique[:, None, :]
    critique_j = critique[None, :, :]

    rewards_i = replay_samples.rewards[:, None]
    rewards_j = replay_samples.rewards[None, :]

    encoded_distance = torch.norm(zs_i - zs_j, dim=2).unsqueeze(-1)  # TODO try L1

    reward_distance = torch.abs(rewards_i - rewards_j)
    critique_distance = torch.abs(critique_i - critique_j)  # TODO check abs?
    bisim_distance = (1 - c) * reward_distance + c / K * critique_distance

    return F.mse_loss(encoded_distance, bisim_distance)


def gradient_penalty(
    encoder: nn.Module, critic: nn.Module, data: torch.Tensor, K: float
) -> torch.Tensor:
    """
    Computes the gradient penalty for the critic, where the target norm is K.

    :param encoder: (nn.Module) the encoder
    :param critic: (nn.Module) the critic
    :param data: (torch.Tensor) input data
    :param K: (float) target gradient norm

    :return: (torch.Tensor) gradient penalty
    """
    s = data.clone().detach().requires_grad_(True)
    z = encoder(s).clone().detach().requires_grad_(True)

    critique = critic(z)
    grad = torch.autograd.grad(
        critique, z, grad_outputs=torch.ones_like(critique), create_graph=True
    )[0]

    return (grad.norm(2, dim=1) - K).pow(2).mean()
