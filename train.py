from pathlib import Path

import click
import gymnasium as gym
import imageio
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy

from bsac import BSAC, BisimConfig

LOG_DIR = Path("logs")

N_TRAIN_TIME_STEPS = 200_000
N_CKPT_TIME_STEPS = N_TRAIN_TIME_STEPS // 10

REPLAY_BUFFER_SIZE = 300_000

C = 0.5
K = 1.0
GRAD_PENALTY = 10.0

BISIM_BATCH_SIZE = 2048
N_CRITIC_TRAINING_STEPS = 5
N_ENCODER_TRAINING_STEPS = 1


@click.command()
@click.argument("algo", type=click.Choice(["sac", "bsac"]))
@click.argument("policy", type=click.Choice(["cnn", "mlp"]))
@click.argument("env-id", type=str)
@click.option("-d", "--device", type=str, default="cuda:0", required=False)
@click.option("-s", "--log-suffix", type=str, default="", required=False)
def main(algo: str, policy: str, env_id: str, device: str, log_suffix: str):
    log_dir_str = f"{algo}_{policy}_{env_id}"
    if len(log_suffix) > 0:
        log_dir_str += f"_{log_suffix}"
    log_dir = LOG_DIR / log_dir_str
    tb_dir = log_dir / "tensorboard"
    video_dir = log_dir / "videos"
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id)
    eval_env = Monitor(gym.make(env_id, render_mode="rgb_array"))

    ckpt_cb = CheckpointCallback(
        save_freq=N_CKPT_TIME_STEPS,
        save_path=str(log_dir),
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir),
        log_path=str(log_dir),
        eval_freq=N_CKPT_TIME_STEPS // 10,
        deterministic=True,
        render=True,
    )

    if policy == "cnn":
        policy_type = CnnPolicy
    elif policy == "mlp":
        policy_type = MlpPolicy
    else:
        raise ValueError(f"Invalid policy: {policy}")

    if algo == "sac":
        model = SAC(
            policy=policy_type,
            env=env,
            buffer_size=REPLAY_BUFFER_SIZE,
            policy_kwargs={
                "share_features_extractor": True,
            },
            device=device,
            tensorboard_log=str(tb_dir),
        )
    elif algo == "bsac":
        model = BSAC(
            policy=policy_type,
            env=env,
            bisim_config=BisimConfig(
                C=C,
                K=K,
                grad_penalty=GRAD_PENALTY,
                batch_size=BISIM_BATCH_SIZE,
                critic_training_steps=N_CRITIC_TRAINING_STEPS,
                encoder_training_steps=N_ENCODER_TRAINING_STEPS,
            ),
            buffer_size=REPLAY_BUFFER_SIZE,
            policy_kwargs={
                "share_features_extractor": True,
            },
            device=device,
            tensorboard_log=str(tb_dir),
        )
    else:
        raise ValueError(f"Invalid algo: {algo}")

    # Train
    model.learn(
        total_timesteps=N_TRAIN_TIME_STEPS,
        callback=[ckpt_cb, eval_cb],
        progress_bar=True,
    )

    # Eval, record
    obs, _ = eval_env.reset()
    done = False
    frames = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        frames.append(eval_env.render())
    eval_env.close()  # https://github.com/google-deepmind/mujoco/issues/1186

    imageio.mimsave(video_dir / "trained_agent.mp4", frames, fps=20)


if __name__ == "__main__":
    main()
