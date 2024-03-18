from pathlib import Path

import gymnasium as gym
import imageio
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_DIR = Path("logs/videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Set up environment and callbacks
env = gym.make("HalfCheetah-v4")
eval_env = Monitor(gym.make("HalfCheetah-v4", render_mode="rgb_array"))
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=str(LOG_DIR),
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(LOG_DIR),
    log_path=str(LOG_DIR),
    eval_freq=10_000,
    deterministic=True,
    render=True,
)

# Train the agent
model = SAC(policy="MlpPolicy", env=env, device="cuda:1")
model.learn(
    total_timesteps=100_000,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True,
)

# Record a video of the trained agent
obs, info = eval_env.reset()
done = False
frames = []
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = eval_env.step(action)
    done = terminated or truncated
    frames.append(eval_env.render())

imageio.mimsave(VIDEO_DIR / "trained_agent.mp4", frames, fps=20)
