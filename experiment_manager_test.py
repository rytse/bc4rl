import argparse

from rl_zoo3.exp_manager import ExperimentManager


def main():
    exp_manager = ExperimentManager(
        argparse.Namespace(),
        "sac",
        "LunarLanderContinuous-v2",
        "./logs/experiment_manager_test/",
        hyperparams={
            "policy_kwargs": {
                "share_features_extractor": True,
            },
        },
    )

    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparameters = results
        if model is not None:
            breakpoint()
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)
        else:
            raise Exception("Loaded model is None")
    else:
        raise Exception("Failed to set up experiment.")


if __name__ == "__main__":
    main()
