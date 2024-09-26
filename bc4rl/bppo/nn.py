from typing import List

import torch
import torch.nn as nn

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
