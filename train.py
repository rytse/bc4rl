from pathlib import Path

import click
import imageio
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from experiments.algos import get_algo
from experiments.envs import get_env

LOG_DIR = Path("logs")

N_TRAIN_STEPS = 1_000_000
CKPT_RATIO = 10


@click.command()
@click.argument("algo-name", type=click.Choice(["sac", "bsac"]))
@click.argument("policy-type", type=str)
@click.argument("env-name", type=str)
@click.option("-d", "--device", type=str, default="cuda:0", required=False)
@click.option("-s", "--log-suffix", type=str, default="", required=False)
@click.option("-n", "--n-train-steps", type=int, default=N_TRAIN_STEPS, required=False)
@click.option("-c", "--ckpt-ratio", type=int, default=CKPT_RATIO, required=False)
def main(
    algo_name: str,
    policy_type: str,
    env_name: str,
    device: str,
    log_suffix: str,
    n_train_steps: int,
    ckpt_ratio: int,
):
    log_dir = LOG_DIR / (
        f"{algo_name}_{policy_type}_{env_name.replace('/', '_')}"
        + (f"_{log_suffix}" if len(log_suffix) > 0 else "")
    )
    tb_dir = log_dir / "tensorboard"
    video_dir = log_dir / "videos"
    best_path = log_dir / "best_model.zip"
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    train_env, eval_env = get_env(env_name)

    cb_freq = max(n_train_steps // ckpt_ratio // eval_env.num_envs, 1)
    ckpt_cb = CheckpointCallback(
        save_freq=cb_freq,
        save_path=str(log_dir),
        name_prefix="rl_model",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir),
        log_path=str(log_dir),
        eval_freq=cb_freq,
        n_eval_episodes=5,
        deterministic=True,
    )

    # Train
    algo = get_algo(algo_name, policy_type, train_env, device, tb_dir)
    if best_path in log_dir.iterdir():
        print(f"Loading best model from {best_path}")
        algo.set_parameters(str(best_path))

    algo.learn(
        total_timesteps=n_train_steps,
        callback=[ckpt_cb, eval_cb],
        progress_bar=True,
    )

    # Eval, record
    obs = eval_env.reset()
    done = False
    frames = []
    while not done:
        obs = np.array(obs)
        action, _ = algo.predict(obs, deterministic=True)
        obs, _, dones, _ = eval_env.step(action)
        done = dones.any()
        frames.append(eval_env.render())
    eval_env.close()  # https://github.com/google-deepmind/mujoco/issues/1186

    imageio.mimsave(video_dir / "trained_agent.mp4", frames, fps=20)


if __name__ == "__main__":
    main()
