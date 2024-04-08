from typing import List, Type

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from bc4rl.nn import MLP


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to process stacked frames.
    """

    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.cnn(observations)


class CustomMLP(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    :param width: (int) Number of units in hidden layer
    :param depth: (int) Number of hidden layers
    :param activation: (Type[nn.Module]) Activation function
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 16,
        net_arch: List[int] = [16],
        act: Type[nn.Module] = nn.ReLU,
        orth_init: bool = False,
    ):
        super().__init__(observation_space, features_dim)
        self.act = act
        self.orth_init = orth_init

        self.mlp = MLP(
            observation_space.shape[0], features_dim, net_arch, act, orth_init
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)
