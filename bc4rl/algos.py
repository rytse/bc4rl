from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rl_zoo3 import linear_schedule
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    ReplayBufferSamples,
    Schedule,
)
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac import SAC

from bc4rl.nn import MLP

from .encoder import CustomCNN, CustomMLP
from .policies import (
    BSACActor,
    BSACCnnPolicy,
    BSACMlpPolicy,
    BSACMultiInputPolicy,
    BSACPolicy,
)

SelfBSAC = TypeVar("SelfBSAC", bound="BSAC")


class BSAC(SAC):
    encoder_aliases: ClassVar[Dict[str, Type[nn.Module]]] = {
        "CustomMLP": CustomMLP,
        "CustomCNN": CustomCNN,
    }
    policy_aliases: ClassVar[Dict[str, Type[BSACPolicy]]] = {
        "BSACMlpPolicy": BSACMlpPolicy,
        "BSACCnnPolicy": BSACCnnPolicy,
        "BSACMultiInputPolicy": BSACMultiInputPolicy,
    }

    policy: BSACPolicy
    actor: BSACActor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[BSACPolicy]],
        env: Union[GymEnv, str],
        sac_lr: Union[float, Schedule] = 3e-4,
        bisim_lr: Union[str, float] = 3e-4,
        bisim_c: float = 0.5,
        bisim_k: float = 1.0,
        bisim_use_q: bool = False,
        bisim_grad_penalty: float = 1.0,
        features_extractor_class: Union[Type[BaseFeaturesExtractor], str] = CustomMLP,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        bisim_critic_kwargs: Optional[Union[Dict[str, Any], str]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.bisim_c = bisim_c
        self.bisim_k = bisim_k
        self.bisim_use_q = bisim_use_q
        self.bisim_grad_penalty = bisim_grad_penalty

        policy_kwargs = policy_kwargs if policy_kwargs is not None else {}
        policy_kwargs["share_features_extractor"] = True

        if isinstance(features_extractor_class, str):
            policy_kwargs["features_extractor_class"] = self.encoder_aliases[
                features_extractor_class
            ]
        else:
            policy_kwargs["features_extractor_class"] = features_extractor_class

        if isinstance(features_extractor_kwargs, str):
            features_extractor_kwargs = eval(features_extractor_kwargs)
            assert isinstance(features_extractor_kwargs, dict)
        elif features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        policy_kwargs["features_extractor_kwargs"] = features_extractor_kwargs

        if isinstance(sac_lr, str):
            lr_str, lr_val = sac_lr.split("_")
            assert lr_str == "lin"
            sac_lr = linear_schedule(float(lr_val))

        super().__init__(
            policy,
            env,
            sac_lr,
            buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        if isinstance(bisim_critic_kwargs, str):
            bisim_critic_kwargs = eval(bisim_critic_kwargs)
            assert isinstance(bisim_critic_kwargs, dict)
        elif bisim_critic_kwargs is None:
            bisim_critic_kwargs = {}
        self.bisim_critic = self.make_bisim_critic(
            self.encoder.features_dim, **bisim_critic_kwargs
        ).to(device)

        # Wasserstein critic estimation works better without momentum (see W-GAN paper), so we use
        # vanilla SGD
        self.bisim_critic_optimizer = optim.Adam(
            self.bisim_critic.parameters(),
            lr=float(bisim_lr),
        )

    def make_bisim_critic(
        self,
        feature_dim: int,
        net_arch: List[int] = [8],
        act: Type[nn.Module] = nn.ReLU,
        orth_init: bool = False,
    ) -> nn.Module:
        return MLP(feature_dim, 1, net_arch, act, orth_init)

    def _create_aliases(self) -> None:
        super()._create_aliases()
        self.encoder = self.policy.encoder
        self.encoder_optimizer = self.policy.encoder_optimizer

    def bisim_loss(
        self,
        replay_data: ReplayBufferSamples,
        target: torch.Tensor,
        n_samp: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zs = self.encoder(
            preprocess_obs(
                replay_data.observations.detach().requires_grad_(),
                self.observation_space,
            )
        )
        next_zs = self.encoder(
            preprocess_obs(
                replay_data.next_observations.detach().requires_grad_(),
                self.observation_space,
            )
        )
        target = target.detach().requires_grad_()
        critique = self.bisim_critic(next_zs)

        critique_grad = torch.autograd.grad(
            critique,
            next_zs,
            grad_outputs=torch.ones_like(critique),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_penalty = (critique_grad.norm(2, dim=1) - self.bisim_k).pow(2).mean()

        # Randomly sample n_samp pairs of zs and critique
        assert n_samp <= zs.shape[0]
        idx_i = torch.randperm(zs.shape[0])[:n_samp]
        idx_j = torch.randperm(zs.shape[0])[:n_samp]

        zs_i = zs[idx_i]
        zs_j = zs[idx_j]
        critique_i = critique[idx_i]
        critique_j = critique[idx_j]
        target_i = target[idx_i]
        target_j = target[idx_j]

        encoded_distance = torch.linalg.norm(zs_i - zs_j, ord=1, dim=1).unsqueeze(-1)
        reward_distance = torch.abs(target_i - target_j)
        critique_distance = torch.abs(critique_i - critique_j)
        bisim_distance = (
            1 - self.bisim_c
        ) * reward_distance + self.bisim_c / self.bisim_k * critique_distance
        bisim_loss = F.mse_loss(encoded_distance, bisim_distance)

        return bisim_loss, grad_penalty

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [
            self.actor.optimizer,
            self.critic.optimizer,
            self.encoder_optimizer,
        ]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        bisim_losses, grad_penalties = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Compute the target Q value
            with torch.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                # Compute the next Q values: min over all critics targets
                next_q_values = torch.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, torch.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = torch.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = torch.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            # self.encoder_optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            # self.encoder_optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            # Jointly optimize the encoder and bisim critic
            if self.bisim_use_q:
                agg_q_values = sum(current_q_values)
                assert isinstance(agg_q_values, torch.Tensor)
                target = agg_q_values
            else:
                target = replay_data.rewards
            bisim_loss, grad_penalty = self.bisim_loss(replay_data, target)
            bisim_losses.append(bisim_loss.item())
            grad_penalties.append(grad_penalty.item())

            self.encoder_optimizer.zero_grad()
            bisim_loss.backward(retain_graph=True)
            self.encoder_optimizer.step()

            bisim_critic_loss = bisim_loss + self.bisim_grad_penalty * grad_penalty
            self.bisim_critic_optimizer.zero_grad()
            bisim_critic_loss.backward()
            self.bisim_critic_optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/bisim_loss", np.mean(bisim_losses))
        self.logger.record("train/grad_penalty", np.mean(grad_penalties))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfBSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "BSAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfBSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["encoder", "bisim_critic"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts, saved_pytorch_variables = super()._get_torch_save_params()
        state_dicts += [
            "encoder",
            "encoder_optimizer",
            "bisim_critic",
            "bisim_critic_optimizer",
        ]
        return state_dicts, saved_pytorch_variables


class CustomSAC(SAC):
    def __init__(self, *args, **kwargs):
        kwargs["policy_kwargs"] = dict(
            share_features_extractor=True,
            features_extractor_class=CustomMLP,
            features_extractor_kwargs=dict(net_arch=[16], act=nn.ReLU, orth_init=True),
            net_arch=[400, 300],
        )
        super().__init__(*args, **kwargs)
