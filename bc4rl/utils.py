import torch
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.type_aliases import PyTorchObs


def preprocess_and_detach_obs(obs: PyTorchObs, space) -> PyTorchObs:
    preprocessed = preprocess_obs(obs, space)
    if isinstance(preprocessed, torch.Tensor):
        return preprocessed.detach().requires_grad_()
    else:
        detached = {}
        for key, val in preprocessed.items():
            detached[key] = val.detach().requires_grad_()
        return detached
