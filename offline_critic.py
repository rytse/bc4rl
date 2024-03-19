from pathlib import Path

import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm

from bc4rl import bisim_loss

LOG_DIR = Path("logs")

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
critic = nn.Sequential(nn.Linear(encoder.features_dim, 1)).to(model.device)
critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
num_training_steps = 10_000
batch_size = 256
for step in tqdm(range(num_training_steps)):
    loss = bisim_loss(replay_buffer, encoder, critic, 0.5, 1.0, batch_size)
    critic_opt.zero_grad()
    loss.backward()
    critic_opt.step()

    if step % 100 == 0:
        print(f"Step: {step}, Loss: {loss.item()}")


final_loss = bisim_loss(replay_buffer, encoder, critic, 0.5, 1.0, 1000)
print(f"Final loss: {final_loss.item()}")
