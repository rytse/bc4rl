from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, k, n_s, n_a, d_model):
        super(AntisymmetricNN, self).__init__()
        self.d_model = d_model

        # Input encoding
        self.encode_z = nn.Sequential(
            nn.Linear(k, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

        self.encode_s = nn.Sequential(
            nn.Linear(n_s, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2),
        )

        self.encode_a = nn.Sequential(
            nn.Linear(n_a, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2),
        )

        # Final processing
        self.final_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def forward(self, z, s_i, a_i, s_j, a_j):
        # Input encoding
        z_encoded = self.encode_z(z)

        # Encode and concatenate s and a for both i and j
        u_encoded = torch.cat([self.encode_s(s_i), self.encode_a(a_i)], dim=-1)
        v_encoded = torch.cat([self.encode_s(s_j), self.encode_a(a_j)], dim=-1)

        # Attention-like mechanism
        score_u = torch.sum(z_encoded * u_encoded, dim=-1) / (self.d_model**0.5)
        score_v = torch.sum(z_encoded * v_encoded, dim=-1) / (self.d_model**0.5)

        # Apply softmax to get attention weights
        alpha_u, alpha_v = F.softmax(
            torch.stack([score_u, score_v], dim=-1), dim=-1
        ).unbind(-1)

        # Compute weighted difference
        diff = alpha_u.unsqueeze(-1) * u_encoded - alpha_v.unsqueeze(-1) * v_encoded

        # Final processing
        output = self.final_mlp(diff)

        return output.squeeze(-1)
