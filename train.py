import argparse
from pathlib import Path

import click
from rl_zoo3 import ALGOS
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.hyperparams_opt import HYPERPARAMS_SAMPLER

from bc4rl.algos import BSAC
from bc4rl.samplers import sample_bsac_params


@click.command()
@click.argument("algo", type=str, default="bsac")
@click.argument("env", type=str, default="LunarLanderContinuous-v2")
@click.option("-d", "--device", type=str, default="cuda:0")
@click.option("-h", "--optimize_hyperparameters", is_flag=True)
def main(algo: str, env: str, device: str, optimize_hyperparameters: bool):
    ALGOS["bsac"] = BSAC
    HYPERPARAMS_SAMPLER["bsac"] = sample_bsac_params

    custom_hyperparam_path = Path(f"./hyperparams/{algo}.yml")
    if custom_hyperparam_path.exists():
        custom_hyperparam_path = str(custom_hyperparam_path)
    else:
        custom_hyperparam_path = None

    exp_manager = ExperimentManager(
        argparse.Namespace(),
        algo,
        env,
        log_folder="./logs",
        tensorboard_log="./tensorboard",
        optimize_hyperparameters=optimize_hyperparameters,
        n_trials=1_000,
        n_jobs=1,
        log_interval=10,
        show_progress=True,
        device=device,
        config=custom_hyperparam_path,
    )

    results = exp_manager.setup_experiment()
    if results is not None:
        model, _ = results
        if model is not None:
            print("Training model")
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)

        else:
            raise Exception("Loaded model is None")
    else:
        print("Optimizing hyperparameters")
        exp_manager.hyperparameters_optimization()


if __name__ == "__main__":
    main()
