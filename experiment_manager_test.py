import argparse

import click
import rl_zoo3
from rl_zoo3.exp_manager import ExperimentManager

from bc4rl.algos import BSAC, BisimConfig
from bc4rl.encoder import CustomMLP


def main():
    rl_zoo3.ALGOS["bsac"] = BSAC

    exp_manager = ExperimentManager(
        argparse.Namespace(),
        "bsac",
        "LunarLanderContinuous-v2",
        log_folder="./logs",
        tensorboard_log="./logs/tensorboard",
        hyperparams={
            "bisim_config": BisimConfig(1.0, 1.0, 10.0, 2048, 20),
            "policy_kwargs": {
                "share_features_extractor": True,
                "features_extractor_class": CustomMLP,
                "net_arch": [400, 300],
            },
        },
        log_interval=1,
        device="cuda:1",
        config="./hyperparams/bsac.yml",
    )

    results = exp_manager.setup_experiment()
    if results is not None:
        model, _ = results
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)

        else:
            raise Exception("Loaded model is None")
    else:
        raise Exception("Failed to set up experiment.")


if __name__ == "__main__":
    main()
