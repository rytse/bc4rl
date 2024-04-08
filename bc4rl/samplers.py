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
