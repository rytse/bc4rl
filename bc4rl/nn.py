from typing import Type

import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int,
        depth: int,
        act: Type[nn.Module] = nn.ReLU,
        orth_init: bool = False,
    ):
        super(Mlp, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.depth = depth
        self.activation = act
        self.orth_init = orth_init

        layers = [self._get_linear(in_dim, width), act()]
        for _ in range(depth):
            layers.append(self._get_linear(width, width))
            layers.append(act())
        layers.append(self._get_linear(width, out_dim))

        self.mlp = nn.Sequential(*layers)

    def _get_linear(self, in_dim: int, out_dim: int) -> nn.Linear:
        linear = nn.Linear(in_dim, out_dim)
        if self.orth_init:
            nn.init.orthogonal_(
                linear.weight,
                int(nn.init.calculate_gain(self.activation.__name__.lower())),
            )
        return linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
