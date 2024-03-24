from typing import Type

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


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
        features_dim: int = 64,
        width: int = 64,
        depth: int = 2,
        activation: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__(observation_space, features_dim)

        in_dim = observation_space.shape[0]
        layers = [nn.Linear(in_dim, width), activation()]
        for _ in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(activation())
        layers.append(nn.Linear(width, features_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)
