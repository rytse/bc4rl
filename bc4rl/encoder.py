from typing import Dict, List, Type

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict

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


class DeepBisimCNN(BaseFeaturesExtractor):
    """Convolutional encoder of pixels observations."""

    def __init__(
        self,
        observation_space: spaces.Box,
        feature_dim: int,
        # num_layers: int = 2,
        cnn_out_dim: int = 39,
        num_filters: int = 32,
        stride: int = 2,
    ):
        super().__init__(observation_space=observation_space, features_dim=feature_dim)

        self.feature_dim = feature_dim

        # self.num_layers = num_layers
        # self.convs = nn.Sequential(
        #     *(
        #         [
        #             nn.Conv2d(
        #                 observation_space.shape[0], num_filters, 3, stride=stride
        #             ),
        #             nn.ReLU(),
        #         ]
        #         + [nn.Conv2d(num_filters, num_filters, 3, stride=1), nn.ReLU()]
        #         * num_layers
        #     )
        # )
        # out_dim = {2: 39, 4: 35, 6: 31}[num_layers]
        # out_dim = 39
        # self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        # self.ln = nn.LayerNorm(self.feature_dim)

        self.net = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], num_filters, 3, stride=stride),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, 3, stride=1),
            nn.ReLU(),
            nn.Linear(num_filters * cnn_out_dim * cnn_out_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )

    def forward(self, obs: torch.Tensor, detach: bool = False) -> torch.Tensor:
        obs_normed = obs / 255.0
        # h_stacked = self.convs(obs_normed)
        # h = h_stacked.view(h_stacked.size(0), -1)

        # if detach:
        #     h = h.detach()

        # h_fc = self.fc(h)
        # out = self.ln(h_fc)

        return self.net(obs_normed)


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_feature_dim: int = 50,
        normalized_image: bool = False,
    ):
        super().__init__(observation_space, 1)

        self.extractors: torch.ModuleDict = {}

        total_concat_size = 0
        self.feature_sizes = {}
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image):
                extractor = DeepBisimCNN(subspace, cnn_feature_dim)
                self.extractors[key] = extractor
                self.feature_sizes[key] = extractor.feature_dim
                total_concat_size += extractor.feature_dim
            else:
                print("Warning: non-image space not supported yet!")
                extractor = nn.Flatten()
                self.extractors[key] = extractor
                self.feature_sizes[key] = get_flattened_obs_dim(subspace)
                total_concat_size += self.feature_sizes[key]

        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        # sample_tensor = next(iter(observations.values()))
        # batch_size = sample_tensor.size(0)
        # combined_features = torch.zeros(
        #     batch_size, self._features_dim, device=sample_tensor.device
        # )
    
        encoded_tensor_list = []
        # current_index = 0
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
            # extracted_features = extractor(observations[key])
            # feature_size = self.feature_sizes[key]
            # combined_features[:, current_index : current_index + feature_size] = (
            #     extracted_features
            # )
            # current_index += feature_size

        # return combined_features
        return torch.cat(encoded_tensor_list, dim=1)
