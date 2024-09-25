from typing import Generator, NamedTuple, Optional

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class RolloutReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    next_observations: torch.Tensor  # new
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    rewards: torch.Tensor  # new


class RolloutReplayBuffer(RolloutBuffer):
    """
    Rollout buffer that also includes next_obs.
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def reset(self) -> None:
        self.next_observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32
        )
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """We shouldn't be able to call this method, we should be forced to use add_with_next_obs instead"""
        raise NotImplementedError

    def add_with_next_obs(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        if isinstance(self.observation_space, spaces.Discrete):
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
        self.next_observations[self.pos] = np.array(next_obs)

        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutReplayBufferSamples, None, None]:
        """We shouldn't be able to call this method, we should be forced to use get_with_next_obs instead"""
        raise NotImplementedError

    def get_with_next_obs(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutReplayBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "next_observations",  # (obviously) added next_observations
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "rewards",  # added rewards, regular PPO doesn't have this
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples_with_next_obs(
                indices[start_idx : start_idx + batch_size]
            )
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> RolloutReplayBufferSamples:
        """We shouldn't be able to call this method, we should be forced to use _get_samples_with_next_obs instead"""
        raise NotImplementedError

    def _get_samples_with_next_obs(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None,
    ) -> RolloutReplayBufferSamples:

        assert (
            self.observations.size == self.next_observations.size
        ), "Observations and next_observations must have the same size"
        assert np.all(
            batch_inds < len(self.observations)
        ), f"Invalid batch indices: max index {np.max(batch_inds)} >= buffer size {len(self.observations)}"

        data = (
            self.observations[batch_inds],
            self.next_observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.rewards[batch_inds].flatten(),
        )
        return RolloutReplayBufferSamples(*tuple(map(self.to_torch, data)))
