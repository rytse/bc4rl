import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


# Define a function to create and wrap the environment
def make_env(env_name):
    def _init():
        env = gym.make(env_name)
        # Check if the environment follows the Gym interface
        check_env(env, warn=True)
        return env

    return _init


# Environment setup
env_name = "CarRacing-v2"
env = FrameStack(GrayScaleObservation(gym.make(env_name)), 4)
# vec_env = make_vec_env(env_name, n_envs=1, seed=0, wrapper_class=make_env)
# Stack frames
# frame_stacked_env = VecFrameStack(vec_env, n_stack=4)

# Initialize the agent
model = SAC("CnnPolicy", env, verbose=1, device="cpu", buffer_size=100_000)
# model = PPO("CnnPolicy", env, verbose=1, device="cpu")

# Train the agent
# total_timesteps = 100_000  # Adjust this value based on your computational budget
total_timesteps = 10  # Adjust this value based on your computational budget
model.learn(total_timesteps=total_timesteps)

# Save the trained model
model.save("ppo_carracing_v2")

print("Training complete. Model saved.")
