from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from bc4rl.nn import SpectrallyNormalizedMLP


class BisimCritic(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, net_arch: List[int]):
        super(BisimCritic, self).__init__()
        self.net = SpectrallyNormalizedMLP(
            in_dim=2 * feature_dim + action_dim,
            out_dim=1,
            net_arch=net_arch,
            act=nn.ReLU,
            ortho_init=True,
        )
        self.feature_dim = feature_dim
        self.action_dim = action_dim

    def forward(
        self,
        current_state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-allocate tensor for the output
        batch_size = current_state.shape[0]
        input_tensor = torch.zeros(
            batch_size, 2 * self.feature_dim + self.action_dim
        ).to(current_state.device)
        input_tensor[:, : self.feature_dim] = current_state
        input_tensor[:, self.feature_dim : 2 * self.feature_dim] = next_state
        input_tensor[:, 2 * self.feature_dim :] = action

        return self.net(input_tensor)


class AntisymmetricNN(nn.Module):
    def __init__(self, k: int, n_s: int, n_a: int):
        super(AntisymmetricNN, self).__init__()
        self.k = k

        self.encode_z = nn.Sequential(
            spectral_norm(nn.Linear(k, k)),
            nn.ReLU(),
            spectral_norm(nn.Linear(k, k)),
        )
        self.encode_sa = nn.Sequential(
            nn.Linear(n_s + n_a, k),
            nn.ReLU(),
            nn.Linear(k, k),
        )
        self.final_mlp = nn.Sequential(
            spectral_norm(nn.Linear(k, k)), nn.ReLU(), spectral_norm(nn.Linear(k, k))
        )

    def forward(
        self,
        z: torch.Tensor,
        s_i: torch.Tensor,
        a_i: torch.Tensor,
        s_j: torch.Tensor,
        a_j: torch.Tensor,
    ) -> torch.Tensor:
        z_encoded = self.encode_z(z)

        u_encoded = self.encode_sa(torch.cat([s_i, a_i], dim=-1))
        v_encoded = self.encode_sa(torch.cat([s_j, a_j], dim=-1))

        score_u = torch.sum(z_encoded * u_encoded, dim=-1) / (self.k**0.5)
        score_v = torch.sum(z_encoded * v_encoded, dim=-1) / (self.k**0.5)

        # Apply softmax to get attention weights
        alpha_u, alpha_v = F.softmax(
            torch.stack([score_u, score_v], dim=-1), dim=-1
        ).unbind(-1)

        # Compute weighted difference
        diff = alpha_u.unsqueeze(-1) * u_encoded - alpha_v.unsqueeze(-1) * v_encoded

        return self.final_mlp(diff)
