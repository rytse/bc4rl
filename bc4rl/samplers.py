from typing import Any, Dict

import optuna
from rl_zoo3.hyperparams_opt import sample_sac_params


def sample_bsac_params(
    trial: optuna.Trial, _n_actions: int, _n_envs: int, _additional_args: dict
) -> Dict[str, Any]:
    """
    Sampler for BSAC hyperparams. Takes defaults from SAC and adds the missing ones.
    """
    hyperparams = {}
    hyperparams["sac_lr"] = trial.suggest_float("sac_lr", 1e-5, 1, log=True)
    hyperparams["bisim_lr"] = trial.suggest_float("bisim_lr", 1e-5, 1e-2, log=True)
    hyperparams["bisim_c"] = trial.suggest_float("bisim_c", 0.1, 0.9, log=True)
    hyperparams["bisim_use_q"] = trial.suggest_categorical("bisim_use_q", [True, False])
    hyperparams["bisim_grad_penalty"] = trial.suggest_float(
        "bisim_grad_penalty", 1e-3, 1e2, log=True
    )

    return hyperparams


def custom_sample_sac_params(
    trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict
) -> Dict[str, Any]:
    """
    Sampler for BSAC hyperparams. Takes defaults from SAC and adds the missing ones.
    """

    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    buffer_size = trial.suggest_categorical(
        "buffer_size", [int(5e3), int(1e4)]
    )
    learning_starts = trial.suggest_categorical(
        "learning_starts", [0, 1000, 10000, 20000]
    )
    # train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
    train_freq = trial.suggest_categorical(
        "train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512]
    )
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    # gradient_steps takes too much time
    # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    gradient_steps = train_freq
    # ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    ent_coef = "auto"
    # You can comment that out when not using gSDE
    log_std_init = trial.suggest_float("log_std_init", -4, 1)
    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        # Uncomment for tuning HER
        # "large": [256, 256, 256],
        # "verybig": [512, 512, 512],
    }[net_arch_type]

    target_entropy = "auto"
    # if ent_coef == 'auto':
    #     # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
    #     target_entropy = trial.suggest_float('target_entropy', -10, 10)

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "target_entropy": target_entropy,
        "policy_kwargs": dict(log_std_init=log_std_init, net_arch=net_arch),
    }

    if additional_args["using_her_replay_buffer"]:
        hyperparams = sample_her_params(
            trial, hyperparams, additional_args["her_kwargs"]
        )

    return hyperparams
