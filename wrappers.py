from gymnasium import Wrapper, Env, spaces

class RGBArrayObsWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=env.observation_space.shape[::-1] + (3,),
            dtype=env.observation_space.dtype,
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs.transpose(2, 0, 1), reward, done, info

    def reset(self):
        return self.env.reset().transpose(2, 0, 1