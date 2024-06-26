from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import Actor, SACPolicy


class FrozenActor(Actor):
    """
    Actor with frozen encoder gradients
    """

    def extract_features(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        features_extractor: BaseFeaturesExtractor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return super().extract_features(obs, features_extractor)


class BSACPolicy(SACPolicy):
    """
    Policy class with actor, critic, and shared feature extractor where the feature extractor is
    optimized with the critic, rather than the actor.
    """

    actor: FrozenActor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    encoder: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def make_actor(self, features_extractor: BaseFeaturesExtractor) -> FrozenActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return FrozenActor(**actor_kwargs).to(self.device)

    def _build(self, lr_schedule: Schedule) -> None:
        self.encoder = self.make_features_extractor()
        self.encoder_optimizer = self.optimizer_class(
            self.encoder.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        self.actor = self.make_actor(features_extractor=self.encoder)
        actor_params = [
            param
            for name, param in self.actor.named_parameters()
            if "features_extractor" not in name
        ]
        self.actor.optimizer = self.optimizer_class(
            actor_params,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        self.critic = self.make_critic(features_extractor=self.encoder)
        critic_params = [
            param
            for name, param in self.critic.named_parameters()
            if "features_extractor" not in name
        ]
        self.critic.optimizer = self.optimizer_class(
            critic_params,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def _predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return super()._predict(obs, deterministic)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return super().forward(obs, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.encoder.train(mode)

        self.training = mode


BSACMlpPolicy = BSACPolicy


class BSACCnnPolicy(BSACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


class BSACMultiInputPolicy(BSACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )
