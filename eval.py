from pathlib import Path

import gymnasium as gym
import imageio
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

LOG_DIR = Path("logs")
VIDEO_DIR = Path("logs/videos")

env = gym.make("HalfCheetah-v4")
eval_env = Monitor(gym.make("HalfCheetah-v4", render_mode="rgb_array"))

model_path = LOG_DIR / "best_model.zip"
model = SAC.load(model_path, env=env, device="cuda:1")

obs, info = eval_env.reset()
done = False
frames = []
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = eval_env.step(action)
    done = terminated or truncated
    frames.append(eval_env.render())
eval_env.close()  # https://github.com/google-deepmind/mujoco/issues/1186

imageio.mimsave(VIDEO_DIR / "evaluated_agent.mp4", frames, fps=20)
