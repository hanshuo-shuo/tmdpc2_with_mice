import gym
import gymnasium
import numpy as np
import gym
import numpy as np
from typing import Tuple, Optional
import copy
    
class MypreyWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.model = env.model
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,), 
            dtype=np.float32
        )
        

    def see_predator(self):
        # if the predator is visible, then it is a safe zone
        if self.model.prey_data.predator_visible:
            return True
        else:
            return False

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action.copy())
        obs = obs.astype(np.float32)
        obs = copy.deepcopy(obs)
        # if self.see_predator():
        #     print('Predator is visible')
        # else:
        #     print('Predator is not visible')
        obs = np.delete(obs, -4)
        return obs, reward, False, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render()

    def reset(self, seed=None):
        obs, _ = self.env.reset()
        obs = obs.astype(np.float32)
        obs = copy.deepcopy(obs)
        obs = np.delete(obs, -4)
        return obs

    
