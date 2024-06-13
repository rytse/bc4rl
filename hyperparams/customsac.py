from bc4rl.encoder import CustomCombinedExtractor

hyperparams = {
    "dm_control/cheetah-run-v0": {
        "env_wrapper": "gymnasium.wrappers.PixelObservationWrapper",
        "frame_stack": 3,
        "n_timesteps": 1e6,
        "policy": "MultiInputPolicy",
        "policy_kwargs": {
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {"orth_init": True},
        },
        "buffer_size": 10_000,
        "batch_size": 64,
        "train_freq": 2,
        "gamma": 0.99,
        "learning_rate": 1.0e-5,
        "ent_coef": 0.9,
    }
}
