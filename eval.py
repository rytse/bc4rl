from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from experiments.envs import get_env

LOG_DIR = Path("logs/old/experiment_manager_test/sac/LunarLanderContinuous-v2_4")
VIDEO_DIR = LOG_DIR / "videos"
VIDEO_DIR.mkdir(exist_ok=True)

train_env, eval_env = get_env("lunarlander", 1)
# env = gym.make("HalfCheetah-v4")
# eval_env = Monitor(gym.make("HalfCheetah-v4", render_mode="rgb_array"))

model_path = LOG_DIR / "best_model.zip"
model = SAC.load(model_path, env=eval_env, device="cuda:0")

obs = eval_env.reset()
done = False
frames = []
while not done:
    obs = np.array(obs)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, _ = eval_env.step(action)
    done = dones.any()

    frames.append(eval_env.render())
eval_env.close()  # https://github.com/google-deepmind/mujoco/issues/1186

imageio.mimsave(VIDEO_DIR / "evaluated_agent.mp4", frames, fps=20)
