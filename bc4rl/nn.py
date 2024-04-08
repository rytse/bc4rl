from typing import List, Type

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    :param width: (int) Number of units in hidden layer
    :param depth: (int) Number of hidden layers
    :param activation: (Type[nn.Module]) Activation function
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        net_arch: List[int] = [16],
        act: Type[nn.Module] = nn.ReLU,
        orth_init: bool = False,
    ):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = act
        self.orth_init = orth_init

        layers = [self._get_linear(in_dim, net_arch[0]), act()]
        for i in range(1, len(net_arch)):
            layers.append(self._get_linear(net_arch[i - 1], net_arch[i]))
            layers.append(act())
        layers.append(self._get_linear(net_arch[-1], out_dim))

        self.mlp = nn.Sequential(*layers)

    def _get_linear(self, in_dim: int, out_dim: int) -> nn.Linear:
        linear = nn.Linear(in_dim, out_dim)
        if self.orth_init:
            nn.init.orthogonal_(
                linear.weight,
                int(nn.init.calculate_gain(self.act.__name__.lower())),
            )
        return linear

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)
