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
        "policy": "BPPOMultiInputPolicy",
        "policy_kwargs": {
            "share_features_extractor": True,
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {"ortho_init": True},
        },
    },
    "atari": {
        "batch_size": 512,
        "clip_range": "lin_0.1",
        "ent_coef": 0.01,
        "frame_stack": 4,
        "learning_rate": "lin_2.5e-4",
        "n_envs": 8,
        "n_epochs": 4,
        "n_steps": 512,
        "n_timesteps": 10_000_000,
        "bisim_weight": 1e-6,
        "bisim_critic_train_iters": 10,
        "bisim_c": 0.75,
        "bisim_tau": 0.005,
        "policy": "CnnPolicy",
        "vf_coef": 0.5,
        "policy_kwargs": {
            "share_features_extractor": True,
        },
    },
}
