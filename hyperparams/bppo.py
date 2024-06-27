from bc4rl.encoder import CustomCombinedExtractor

hyperparams = {
    "dm_control/cheetah-run-v0": {
        "env_wrapper": "gymnasium.wrappers.PixelObservationWrapper",
        "frame_stack": 3,
        "n_envs": 1,
        "n_timesteps": 1_000_000,  # total training steps
        "n_steps": 4096,  # steps per epoch
        "learning_rate": 3e-4,  # overrided by pi_lr and vf_lr
        "batch_size": 64,
        "n_epochs": 80,
        "gamma": 0.99,
        "gae_lambda": 0.97,
        "clip_range": 0.2,
        "target_kl": 0.01,
        "policy": "MultiInputPolicy",
        "policy_kwargs": {
            "share_features_extractor": True,
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {"orth_init": True},
        },
    }
}
