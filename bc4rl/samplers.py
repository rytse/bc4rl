from typing import Any, Dict

import optuna
from rl_zoo3.hyperparams_opt import sample_sac_params


def sample_bsac_params(
    trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict
) -> Dict[str, Any]:
    """
    Sampler for BSAC hyperparams. Takes defaults from SAC and adds the missing ones.
    """
    hyperparams = sample_sac_params(trial, n_actions, n_envs, additional_args)

    sac_lr = hyperparams["learning_rate"]
    hyperparams["sac_lr"] = sac_lr
    del hyperparams["learning_rate"]

    bisim_lr = trial.suggest_float("bisim_lr", 1e-5, 1e-2, log=True)
    hyperparams["bisim_lr"] = bisim_lr

    hyperparams["bisim_c"] = trial.suggest_float("bisim_c", 0.1, 0.9, log=True)
    # hyperparams["bism_k"] = trial.suggest_float("bisim_k", 0.1, 10.0, log=True)
    hyperparams["bisim_use_q"] = trial.suggest_categorical("bisim_use_q", [True, False])
    hyperparams["bisim_grad_penalty"] = trial.suggest_float(
        "bisim_grad_penalty", 1e-3, 1e2, log=True
    )

    return hyperparams
