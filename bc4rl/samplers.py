from typing import Any, Dict

import optuna
from rl_zoo3.hyperparams_opt import sample_sac_params

from bc4rl.encoder import CustomMLP


def sample_bsac_params(
    trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict
) -> Dict[str, Any]:
    """
    Sampler for BSAC hyperparams. Takes defaults from SAC and adds the missing ones.
    """
    hyperparams = sample_sac_params(trial, n_actions, n_envs, additional_args)

    hyperparams["bisim_kwargs"] = dict(
        C=1.0, K=1.0, grad_penalty=10.0, critic_training_steps=1
    )
    hyperparams["policy_kwargs"]["share_features_extractor"] = True
    hyperparams["policy_kwargs"]["features_extractor_class"] = CustomMLP

    sac_lr = hyperparams["learning_rate"]
    hyperparams["sac_lr"] = sac_lr
    del hyperparams["learning_rate"]

    bisim_lr = trial.suggest_float("bisim_lr", 1e-5, 1e-2, log=True)
    hyperparams["bisim_lr"] = bisim_lr

    hyperparams["ent_coef"] = trial.suggest_categorical(
        "ent_coef", ["auto", 0.5, 0.1, 0.05, 0.01, 0.0001]
    )

    return hyperparams