from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BisimLoss(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        z_dim: int,
        c: float,
        K: float,
        width: int = 128,
        depth: int = 2,
    ):
        super().__init__()

        self.encoder = encoder
        self.z_dim = z_dim
        self.c = c
        self.K = K

        self.layers = nn.ModuleList([nn.Linear(z_dim, width), nn.ReLU()])
        for _ in range(depth):
            self.layers.extend([nn.Linear(width, width), nn.ReLU()])
        self.layers.append(nn.Linear(width, 1))
        self.critic = nn.Sequential(*self.layers)

    def forward(
        self,
        preprocessed_obs: torch.Tensor,
        preprocessed_next_obs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        zs = self.encoder(preprocessed_obs)  # (n_samples, z_dim)
        next_zs = self.encoder(preprocessed_next_obs)  # (n_samples, z_dim)
        critique = self.critic(next_zs)  # (n_samples, 1)

        zs_i = zs[:, None, :]
        zs_j = zs[None, :, :]

        critique_i = critique[:, None, :]
        critique_j = critique[None, :, :]

        rewards_i = rewards[:, None]
        rewards_j = rewards[None, :]

        encoded_distance = torch.norm(zs_i - zs_j, dim=2).unsqueeze(-1)  # TODO try L1

        reward_distance = torch.abs(rewards_i - rewards_j)
        critique_distance = torch.abs(critique_i - critique_j)  # TODO check abs?
        bisim_distance = (
            1 - self.c
        ) * reward_distance + self.c / self.K * critique_distance

        return F.mse_loss(encoded_distance, bisim_distance)


class GradientPenalty(nn.Module):
    def __init__(self, encoder: nn.Module, critic: nn.Module, K: float):
        super().__init__()

        self.encoder = encoder
        self.critic = critic
        self.K = K

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        s = data.clone().detach().requires_grad_(True)
        z = self.encoder(s).clone().detach().requires_grad_(True)

        critique = self.critic(z)
        grad = torch.autograd.grad(
            [critique],
            [z],
            grad_outputs=torch.jit.annotate(
                Optional[List[Optional[torch.Tensor]]], [torch.ones_like(critique)]
            ),
            create_graph=True,
        )[0]

        assert isinstance(grad, torch.Tensor)

        return (grad.norm(2, dim=1) - self.K).pow(2).mean()
