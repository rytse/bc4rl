from pathlib import Path

import gymnasium as gym
import imageio
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac.policies import MlpPolicy

from bsac import BSAC, BisimConfig
from encoder import CustomMLP

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_DIR = Path("logs/videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

N_TRAIN_TIME_STEPS = 100_000
N_CKPT_TIME_STEPS = N_TRAIN_TIME_STEPS // 10

C = 0.5
K = 1.0
GRAD_PENALTY = 10.0

BISIM_BATCH_SIZE = 256
N_CRITIC_TRAINING_STEPS = 100
N_ENCODER_TRAINING_STEPS = 100

# Set up environment and callbacks
env = gym.make("HalfCheetah-v4")
eval_env = Monitor(gym.make("HalfCheetah-v4", render_mode="rgb_array"))
checkpoint_callback = CheckpointCallback(
    save_freq=N_CKPT_TIME_STEPS,
    save_path=str(LOG_DIR),
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(LOG_DIR),
    log_path=str(LOG_DIR),
    eval_freq=N_CKPT_TIME_STEPS // 10,
    deterministic=True,
    render=True,
)

# Train the agent
model = BSAC(
    policy=MlpPolicy,
    env=env,
    bisim_config=BisimConfig(
        C=C,
        K=K,
        grad_penalty=GRAD_PENALTY,
        batch_size=BISIM_BATCH_SIZE,
        critic_training_steps=N_CRITIC_TRAINING_STEPS,
        encoder_training_steps=N_ENCODER_TRAINING_STEPS,
    ),
    policy_kwargs={
        "features_extractor_class": CustomMLP,
        "features_extractor_kwargs": {
            "features_dim": 256,
        },
        "share_features_extractor": True,
    },
    device="cuda:1",
)
model.learn(
    total_timesteps=N_TRAIN_TIME_STEPS,
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
eval_env.close()  # https://github.com/google-deepmind/mujoco/issues/1186

imageio.mimsave(VIDEO_DIR / "trained_agent.mp4", frames, fps=20)
