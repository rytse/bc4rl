from copy import deepcopy
from functools import reduce
import math
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import PPO

from bc4rl.bppo.buffers import RolloutReplayBuffer, RolloutReplayBufferSamples
from bc4rl.bppo.nn import AntisymmetricNN
from bc4rl.utils import preprocess_and_detach_obs

SelfBPPO = TypeVar("SelfBPPO", bound="BPPO")


class BPPO(PPO):

    rollout_buffer: RolloutReplayBuffer

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        bisim_weight=0.1,
        bisim_critic_train_iters: int = 10,
        bisim_lr: Union[str, float] = 3e-4,
        bisim_c: float = 0.5,
        bisim_critic_kwargs: Optional[Union[Dict[str, Any], str]] = None,
        bisim_tau: float = 0.005,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            normalize_advantage,
            ent_coef,
            vf_coef,
            max_grad_norm,
            use_sde,
            sde_sample_freq,
            RolloutReplayBuffer,
            rollout_buffer_kwargs,
            target_kl,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

        assert (
            self.policy.share_features_extractor == True
        ), "Feature extractor must be shared"

        self.bisim_weight = bisim_weight
        self.bisim_critic_train_iters = bisim_critic_train_iters
        self.bisim_lr = bisim_lr
        self.bisim_c = bisim_c
        self.bisim_tau = bisim_tau

        if isinstance(bisim_critic_kwargs, str):
            bisim_critic_kwargs = eval(bisim_critic_kwargs)
            assert isinstance(bisim_critic_kwargs, dict)
        elif bisim_critic_kwargs is None:
            bisim_critic_kwargs = {}

        # TODO we're making assumptions about the structure of ActionSpace.
        assert isinstance(self.action_space, spaces.Discrete)
        assert isinstance(self.action_space.shape, tuple)
        self.bisim_critic = torch.compile(
            AntisymmetricNN(
                self.policy.features_dim,
                self.policy.features_dim,
                reduce(lambda a, b: a * b, self.action_space.shape, 1),
                16,
            ).to(device),
            mode="reduce-overhead",
        )

        # Wasserstein critic estimation works better without momentum (see W-GAN paper), so we use
        # vanilla SGD
        self.bisim_critic_optimizer = optim.Adam(
            self.bisim_critic.parameters(),
            lr=float(bisim_lr),
        )

        self.encoder = self.policy.features_extractor
        self.target_encoder = deepcopy(self.encoder)

        print("BPPO Policy:")
        print(self.policy)
        print()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Equivalent to `OnPolicyAlgorithm.collect_rollouts` but with `next_obs`
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add_with_next_obs(
                self._last_obs,  # type: ignore[arg-type]
                new_obs,
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def bisim_policy_loss(
        self,
        rollout_data: RolloutReplayBufferSamples,
        n_samp: Optional[int] = 256,
    ) -> torch.Tensor:
        """
        Compute the bisimulation loss, i.e. the difference between the bisimulation loss in the
        original space and the L2 distance in the latent space, assuming the bisimulation critic
        is the optimal "critic" in the Kantorovich-Rubinstein duality.
        """
        if n_samp is None or n_samp > rollout_data.observations.shape[0]:
            n_samp = rollout_data.observations.shape[0]

        zs = self.encoder(
            preprocess_and_detach_obs(
                rollout_data.observations,
                self.observation_space,
            )
        )
        with torch.no_grad():
            next_zs = self.target_encoder(
                preprocess_and_detach_obs(
                    rollout_data.next_observations,
                    self.observation_space,
                )
            ).detach()  # TODO torch.no_grad?
            actions = rollout_data.actions.detach()
        target = rollout_data.rewards.float().view(-1, 1)

        # Randomly sample n_samp pairs of zs and critique
        idx_i = torch.randperm(zs.shape[0])[:n_samp]
        idx_j = torch.randperm(zs.shape[0])[:n_samp]

        zs_i = zs[idx_i]
        zs_j = zs[idx_j]
        target_i = target[idx_i]  # todo try using value function like in BSAC?
        target_j = target[idx_j]
        encoded_distance = torch.linalg.norm(zs_i - zs_j, ord=1, dim=1).view(-1, 1)
        reward_distance = torch.abs(target_i - target_j)

        next_zs_i = next_zs[idx_i]
        next_zs_j = next_zs[idx_j]
        act_i = actions[idx_i]
        act_j = actions[idx_j]

        critique_distance = torch.abs(
            self.bisim_critic(next_zs_i, zs_i, act_i, zs_j, act_j)
            - self.bisim_critic(next_zs_j, zs_i, act_i, zs_j, act_j)
        ).view(-1, 1)

        bisim_distance = (
            1 - self.bisim_c
        ) * reward_distance + self.bisim_c * critique_distance

        return F.mse_loss(encoded_distance, bisim_distance)

    def bisim_critic_loss(
        self, rollout_data: RolloutReplayBufferSamples, n_samp: Optional[int] = 256
    ) -> torch.Tensor:
        """
        Compute the bisimulation critic loss, i.e. to get the optimal "critic" in the
        Kantorovich-Rubinstein duality to compute earth-mover's distance. Note this returns a
        negative value because we want to maximize the critic, not minimize it.
        """
        if n_samp is None or n_samp > rollout_data.observations.shape[0]:
            n_samp = rollout_data.observations.shape[0]

        # TODO torch.no_grad?
        with torch.no_grad():
            zs = self.encoder(
                preprocess_and_detach_obs(
                    rollout_data.observations,
                    self.observation_space,
                )
            ).detach()  # note: this term is *not* detached in bisim_policy_loss
            next_zs = self.target_encoder(
                preprocess_and_detach_obs(
                    rollout_data.next_observations,
                    self.observation_space,
                )
            ).detach()
            actions = rollout_data.actions.detach()

        idx_i = torch.randperm(next_zs.shape[0])[:n_samp]
        idx_j = torch.randperm(next_zs.shape[0])[:n_samp]
        zs_i = zs[idx_i]
        zs_j = zs[idx_j]
        next_zs_i = next_zs[idx_i]
        next_zs_j = next_zs[idx_j]
        act_i = actions[idx_i]
        act_j = actions[idx_j]

        critique = self.bisim_critic(
            next_zs_i, zs_i, act_i, zs_j, act_j
        ) - self.bisim_critic(next_zs_j, zs_i, act_i, zs_j, act_j)

        return -torch.mean(critique)

    def update_target_encoder(self):
        """
        Update the target encoder using an EMA of the current encoder.
        """
        for target_param, param in zip(
            self.target_encoder.parameters(), self.encoder.parameters()
        ):
            target_param.data.copy_(
                self.bisim_tau * param.data + (1.0 - self.bisim_tau) * target_param.data
            )

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        bisim_policy_losses, bisim_critic_losses = [], []
        critic_loss_ranges = []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get_with_next_obs(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                ppo_loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Train the (conditional) bisim critic
                min_critic_loss = float("inf")
                max_critic_loss = float("-inf")
                for _ in range(self.bisim_critic_train_iters):
                    bisim_critic_loss = self.bisim_critic_loss(rollout_data, None)
                    self.bisim_critic_optimizer.zero_grad()
                    bisim_critic_loss.backward()
                    min_critic_loss = min(min_critic_loss, bisim_critic_loss.item())
                    max_critic_loss = max(max_critic_loss, bisim_critic_loss.item())
                    self.bisim_critic_optimizer.step()
                    self.update_target_encoder()
                critic_loss_range = abs(max_critic_loss - min_critic_loss) / (
                    max_critic_loss + 1e-8
                )
                critic_loss_ranges.append(critic_loss_range)

                # Update policy, including encoder, given the best estimate of the bisim critic
                bisim_policy_loss = self.bisim_policy_loss(rollout_data)
                loss = ppo_loss + self.bisim_weight * bisim_policy_loss
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

                # Log
                bisim_policy_losses.append(bisim_policy_loss.item())
                bisim_critic_losses.append(bisim_critic_loss.item())

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/bisim_policy_loss", np.mean(bisim_policy_losses))
        self.logger.record("train/bisim_critic_loss", np.mean(bisim_critic_losses))
        self.logger.record(
            "train/critic_loss_range_percentage", np.max(critic_loss_ranges) * 100.0
        )
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", torch.exp(self.policy.log_std).mean().item()
            )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfBPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "BPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfBPPO:
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
            # "encoder_optimizer",
            "bisim_critic",
            "bisim_critic_optimizer",
        ]
        return state_dicts, saved_pytorch_variables
