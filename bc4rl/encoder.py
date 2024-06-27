from typing import Dict, List, Type

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict

from bc4rl.nn import MLP


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


class CustomCNN(BaseFeaturesExtractor):
    """Convolutional encoder of pixels observations."""

    def __init__(
        self,
        observation_space: spaces.Box,
        feature_dim: int,
        depth: int = 2,
        num_filters: int = 32,
        stride: int = 2,
        orth_init: bool = False,
    ):
        super().__init__(observation_space=observation_space, features_dim=feature_dim)

        self.feature_dim = feature_dim

        convs = [
            nn.Conv2d(
                observation_space.shape[0],
                num_filters,
                3,
                stride=stride,
            ),
            nn.ReLU(),
        ]
        for _ in range(depth):
            convs.extend(
                [
                    nn.Conv2d(num_filters, num_filters, 3, stride=1),
                    nn.ReLU(),
                ]
            )
        self.convs = nn.Sequential(*convs, nn.Flatten())

        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape)
            dummy_output = self.convs(dummy_input)
            conv_out_dim = dummy_output.shape[-1]

        self.linear = nn.Linear(conv_out_dim, self.feature_dim)
        self.layer_norm = nn.LayerNorm(self.feature_dim)

        if orth_init:
            self._orth_init()

    def _orth_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor, detach: bool = False) -> torch.Tensor:
        obs_normed = obs / 255.0
        h = self.convs(obs_normed)

        if detach:
            h = h.detach()

        return self.layer_norm(self.linear(h))


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_feature_dim: int = 50,
        normalized_image: bool = False,
        orth_init: bool = False,
    ):
        super().__init__(observation_space, 1)

        extractors: Dict[str, nn.Module] = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image):
                extractor = CustomCNN(subspace, cnn_feature_dim, orth_init=orth_init)
                extractors[key] = extractor
                total_concat_size += extractor.feature_dim
            else:
                print("Warning: non-image space not supported yet!")
                extractor = nn.Flatten()
                extractors[key] = extractor
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        features = []
        for key, extractor in self.extractors.items():
            extracted_features = extractor(observations[key])
            features.append(extracted_features)

        return torch.cat(features, dim=1)
