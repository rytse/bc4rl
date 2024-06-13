from bc4rl.encoder import CustomCombinedExtractor

hyperparams = {
    "dm_control/cheetah-run-v0": {
        "env_wrapper": "gymnasium.wrappers.PixelObservationWrapper",
        "frame_stack": 3,
        "n_timesteps": 2.0e6,
        # "n_envs": 12,
        "policy": "MultiInputPolicy",
        "policy_kwargs": {
            "features_extractor_class": CustomCombinedExtractor,
        },
        "buffer_size": 50_000,
        "batch_size": 128,
        "train_freq": 2,
        "gamma": 0.99,
        "learning_rate": 1.0e-5,
        "ent_coef": 0.9,
    }
}
