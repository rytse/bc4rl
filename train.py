import argparse

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
@click.option("-h", "--optimize_hyperparameters", is_flag=False)
def main(algo: str, env: str, device: str, optimize_hyperparameters: bool):
    ALGOS["bsac"] = BSAC
    HYPERPARAMS_SAMPLER["bsac"] = sample_bsac_params

    exp_manager = ExperimentManager(
        argparse.Namespace(),
        algo,
        env,
        log_folder="./logs",
        tensorboard_log="./logs/tensorboard",
        optimize_hyperparameters=optimize_hyperparameters,
        n_trials=1_000,
        n_jobs=2,
        log_interval=10,
        device=device,
        config=f"./hyperparams/{algo}.yml",
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
