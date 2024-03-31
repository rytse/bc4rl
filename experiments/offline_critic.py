from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.preprocessing import preprocess_obs
from tqdm import tqdm

from bc4rl.bisim import bisim_loss, gradient_penalty

LOG_DIR = Path("logs")

N_REPLAY_EPISODES = 1_000
REPLAY_BUFFER_SIZE = N_REPLAY_EPISODES * 100
N_CRITIC_TRAINING_STEPS = N_REPLAY_EPISODES
N_ENCODER_TRAINING_STEPS = N_REPLAY_EPISODES
N_PAIR_EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

C = 0.5
K = 1.0
GRAD_PENALTY_WEIGHT = 10.0


env = gym.make("HalfCheetah-v4")

model_path = LOG_DIR / "best_model.zip"
model = SAC.load(model_path, env=env, device="cuda:1")

replay_buffer = ReplayBuffer(
    REPLAY_BUFFER_SIZE, env.observation_space, env.action_space, device=model.device
)

print("Populating the replay buffer...")
num_episodes = 0
with tqdm(total=N_REPLAY_EPISODES) as pbar:
    while num_episodes < N_REPLAY_EPISODES:
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.add(
                obs, new_obs, action, np.array([reward]), np.array([done]), [info]
            )

            obs = new_obs

            if done:
                num_episodes += 1
                pbar.update(1)

print("Training the bisim critic...")
encoder = model.policy.critic.features_extractor
critic = nn.Sequential(
    nn.Linear(encoder.features_dim, 256),
    nn.SiLU(),
    nn.Linear(256, 128),
    nn.SiLU(),
    nn.Linear(128, 128),
    nn.SiLU(),
    nn.Linear(128, 1),
).to(model.device)
critic_opt = optim.SGD(critic.parameters(), lr=LEARNING_RATE)
encoder_opt = optim.SGD(encoder.parameters(), lr=LEARNING_RATE)


for pair_epoch in range(N_PAIR_EPOCHS):
    print(f"Pair Epoch {pair_epoch + 1} / {N_PAIR_EPOCHS}")

    # Optimize the critic
    print("Optimizing the critic...")
    for step in tqdm(range(N_CRITIC_TRAINING_STEPS)):
        replay_data = replay_buffer.sample(BATCH_SIZE)
        replay_obs = preprocess_obs(
            replay_data.observations,
            replay_buffer.observation_space,
        )
        replay_next_obs = preprocess_obs(
            replay_data.next_observations,
            replay_buffer.observation_space,
        )
        replay_rewards = replay_data.rewards
        assert isinstance(replay_obs, torch.Tensor)
        assert isinstance(replay_next_obs, torch.Tensor)

        bs_loss = bisim_loss(
            replay_obs,
            replay_next_obs,
            replay_rewards,
            encoder,
            critic,
            C,
            K,
        )
        grad_loss = gradient_penalty(encoder, critic, replay_obs, K)

        loss = bs_loss + GRAD_PENALTY_WEIGHT * grad_loss

        critic_opt.zero_grad()
        loss.backward()
        critic_opt.step()

        if step % 100 == 0:
            print(
                f"Step: {step}, Bisim Loss: {bs_loss.item()}, Grad Loss: {grad_loss.item()}"
            )

    # Optimize the encoder
    print("Optimizing the encoder...")
    for step in tqdm(range(N_ENCODER_TRAINING_STEPS)):
        replay_data = replay_buffer.sample(BATCH_SIZE)
        replay_obs = preprocess_obs(
            replay_data.observations,
            replay_buffer.observation_space,
        )
        replay_next_obs = preprocess_obs(
            replay_data.next_observations,
            replay_buffer.observation_space,
        )
        replay_rewards = replay_data.rewards
        assert isinstance(replay_obs, torch.Tensor)
        assert isinstance(replay_next_obs, torch.Tensor)

        bs_loss = bisim_loss(
            replay_obs,
            replay_next_obs,
            replay_rewards,
            encoder,
            critic,
            C,
            K,
        )

        encoder_opt.zero_grad()
        bs_loss.backward()
        encoder_opt.step()

        if step % 100 == 0:
            print(f"Step: {step}, Bisim Loss: {bs_loss.item()}")
