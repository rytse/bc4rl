import torch
import torch.nn as nn
import torch.nn.functional as F


def bisim_loss(
    preprocessed_obs: torch.Tensor,
    preprocessed_next_obs: torch.Tensor,
    rewards: torch.Tensor,
    encoder: nn.Module,
    critic: nn.Module,
    c: float,
    K: float,
    n_samp: int = 128,
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
    zs = encoder(preprocessed_obs)  # (n_samples, z_dim)
    next_zs = encoder(preprocessed_next_obs)  # (n_samples, z_dim)
    critique = critic(next_zs)  # (n_samples, 1)

    # Randomly sample n_samp pairs of zs and critique
    assert n_samp <= zs.shape[0]
    idx_i = torch.randperm(zs.shape[0])[:n_samp]
    idx_j = torch.randperm(zs.shape[0])[:n_samp]

    zs_i = zs[idx_i]
    zs_j = zs[idx_j]
    critique_i = critique[idx_i]
    critique_j = critique[idx_j]
    rewards_i = rewards[idx_i]
    rewards_j = rewards[idx_j]

    encoded_distance = torch.norm(zs_i - zs_j, dim=1).unsqueeze(-1)  # TODO try L1
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
