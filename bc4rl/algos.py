from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac import SAC

from .bisim import bisim_loss, gradient_penalty
from .policies import (BSACCnnPolicy, BSACMlpPolicy, BSACMultiInputPolicy,
                       BSACPolicy)


@dataclass
class BisimConfig:
    C: float
    K: float
    grad_penalty: float

    batch_size: int
    critic_training_steps: int


class BSAC(SAC):
    policy_aliases: ClassVar[Dict[str, Type[BSACPolicy]]] = {
        "BSACMlpPolicy": BSACMlpPolicy,
        "BSACCnnPolicy": BSACCnnPolicy,
        "BSACMultiInputPolicy": BSACMultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[BSACPolicy]],
        env: Union[GymEnv, str],
        bisim_config: BisimConfig,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
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
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
    ):
        policy_kwargs = policy_kwargs if policy_kwargs is not None else {}
        policy_kwargs["share_features_extractor"] = True

        self.bisim_config = bisim_config

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
        )

        self.bisim_critic = nn.Sequential(
            nn.Linear(self.policy.actor.features_extractor.features_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        ).to(device)

        # Bisim optimization works better without momentum, we use vanilla SGD
        if isinstance(learning_rate, float):
            sgd_lr = learning_rate
        elif callable(learning_rate):
            sgd_lr = learning_rate(1.0)
        else:
            raise ValueError("Invalid learning rate")
        self.bisim_critic_optimizer = optim.SGD(
            self.bisim_critic.parameters(), lr=sgd_lr
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        assert isinstance(self.policy, BSACPolicy)

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [
            self.actor.optimizer,
            self.critic.optimizer,
            self.policy.encoder_optimizer,
            # self.bisim_critic_optimizer, # momentum breaks critic training
        ]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        bs_critic_losses, encoder_losses, actor_losses, critic_losses = [], [], [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            assert self.replay_buffer is not None
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            replay_obs = preprocess_obs(
                replay_data.observations,
                self.replay_buffer.observation_space,
            )
            replay_next_obs = preprocess_obs(
                replay_data.next_observations,
                self.replay_buffer.observation_space,
            )
            replay_rewards = replay_data.rewards
            assert isinstance(replay_obs, torch.Tensor)
            assert isinstance(replay_next_obs, torch.Tensor)

            # Optimize the bisim critic
            for _ in range(self.bisim_config.critic_training_steps):
                bs_loss = bisim_loss(
                    replay_obs.detach(),
                    replay_next_obs.detach(),
                    replay_rewards.detach(),
                    self.policy.encoder,
                    self.bisim_critic,
                    self.bisim_config.C,
                    self.bisim_config.K,
                )
                grad_loss = gradient_penalty(
                    self.policy.encoder,
                    self.bisim_critic,
                    replay_obs.detach(),
                    self.bisim_config.K,
                )
                critic_loss = bs_loss + self.bisim_config.grad_penalty * grad_loss

                self.bisim_critic_optimizer.zero_grad()
                critic_loss.backward()
                self.bisim_critic_optimizer.step()
                bs_critic_losses.append(bs_loss.item())

            # Optimize encoder
            bs_loss = bisim_loss(
                replay_obs.detach(),
                replay_next_obs.detach(),
                replay_rewards.detach(),
                self.policy.encoder,
                self.bisim_critic,
                self.bisim_config.C,
                self.bisim_config.K,
            )
            self.policy.encoder_optimizer.zero_grad()
            bs_loss.backward()
            self.policy.encoder_optimizer.step()
            encoder_losses.append(bs_loss.item())

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
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/encoder_loss", np.mean(encoder_losses))
        self.logger.record("train/bisim_critic_loss", np.mean(bs_critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    # def _excluded_save_params(self) -> List[str]:
    #     return super()._excluded_save_params() + [
    #         "actor",
    #         "critic",
    #         "critic_target",
    #         "bisim_critic",
    #     ]

    # def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
    #     state_dicts = [
    #         "policy",
    #         "actor.optimizer",
    #         "critic.optimizer",
    #         "bisim_critic_optimizer",
    #     ]
    #     if self.ent_coef_optimizer is not None:
    #         saved_pytorch_variables = ["log_ent_coef"]
    #         state_dicts.append("ent_coef_optimizer")
    #     else:
    #         saved_pytorch_variables = ["ent_coef_tensor"]
    #     return state_dicts, saved_pytorch_variables
