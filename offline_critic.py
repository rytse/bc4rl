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

from bc4rl import bisim_loss, gradient_penalty

LOG_DIR = Path("logs")

K = 1.0
GRAD_PENALTY_WEIGHT = 10.0


env = gym.make("HalfCheetah-v4")

model_path = LOG_DIR / "best_model.zip"
model = SAC.load(model_path, env=env, device="cuda:1")

buffer_size = 1_000_000
replay_buffer = ReplayBuffer(
    buffer_size, env.observation_space, env.action_space, device=model.device
)

print("Populating the replay buffer...")
num_episodes = 0
total_episodes = 1_000
with tqdm(total=total_episodes) as pbar:
    while num_episodes < total_episodes:
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
critic_opt = optim.SGD(critic.parameters(), lr=1e-3)
num_training_steps = 10_000
batch_size = 256
for step in tqdm(range(num_training_steps)):

    samples = preprocess_obs(
        replay_buffer.sample(batch_size).observations, env.observation_space
    )
    assert isinstance(samples, torch.Tensor)

    bs_loss = bisim_loss(replay_buffer, encoder, critic, 0.5, 1.0, batch_size)
    grad_loss = gradient_penalty(critic, samples, K)

    loss = bs_loss + GRAD_PENALTY_WEIGHT * grad_loss

    critic_opt.zero_grad()
    loss.backward()
    critic_opt.step()

    if step % 100 == 0:
        print(
            f"Step: {step}, Bisim Loss: {bs_loss.item()}, Grad Loss: {grad_loss.item()}"
        )


final_loss = bisim_loss(replay_buffer, encoder, critic, 0.5, 1.0, 1000)
print(f"Final loss: {final_loss.item()}")
