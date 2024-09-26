import torch.nn as nn

from bc4rl.encoder import CustomCombinedExtractor

hyperparams = {
    "dm_control/cheetah-run-v0": {
        "env_wrapper": "gymnasium.wrappers.PixelObservationWrapper",
        "frame_stack": 3,
        "n_envs": 2,
        "n_timesteps": 10_000_000,  # total training steps
        # "n_timesteps": 16_384,  # total training steps
        "n_steps": 256,  # steps per epoch
        "learning_rate": 0.000028728854414109447,
        "batch_size": 64,
        "n_epochs": 5,
        "gamma": 0.995,
        "gae_lambda": 0.80,
        "clip_range": 0.2,
        "target_kl": 0.01,
        "ent_coef": 0.01288369491630485,
        "max_grad_norm": 0.9,
        "vf_coef": 0.3749488155976731,
        "policy": "MultiInputPolicy",
        "policy_kwargs": {
            "activation_fn": nn.ReLU,
            "share_features_extractor": True,
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {"ortho_init": True},
        },
    },
    "atari": {
        "batch_size": 256,
        "clip_range": "lin_0.1",
        "ent_coef": 0.01,
        "frame_stack": 4,
        "learning_rate": "lin_2.5e-4",
        "n_envs": 8,
        "n_epochs": 4,
        "n_steps": 128,
        "n_timesteps": 10_000_000,
        "policy": "CnnPolicy",
        "vf_coef": 0.5,
        "policy_kwargs": {
            "share_features_extractor": True,
        },
    },
}
