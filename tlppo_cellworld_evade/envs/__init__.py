from copy import deepcopy
import warnings
import numpy as np
import gym
import gymnasium
from envs.wrappers.time_limit import TimeLimit
from envs.wrappers.tensor import TensorWrapper
from envs.wrappers.gym import GymnasiumToGymWrapper
from envs.wrappers.entropy import uncertainty_wrapper_predator, EntropyCalculationWrapper, MypreyWrapper
import cellworld_gym as cwg

# filter warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 030_12_0063
# 21_05

def make_env(cfg, max_episode_steps):
    """
    Make Myosuite environment.
    """
    env = gymnasium.make("CellworldBotEvade-v0",
                         world_name=cfg.world_name,
                         use_lppos=False,
                         use_predator=True,
                         max_step=max_episode_steps,
                         time_step=cfg.time_step,
                         render=False,
                         real_time=False,
                         reward_function=cwg.Reward({"puffed": cfg.panelty, "finished": 1}),
                         action_type=cwg.BotEvadeEnv.ActionType.CONTINUOUS)
    # env.model.goal_threshold = -1 # NO exit close env
    # env = GymnasiumToGymWrapper(env, cfg)
    # env = Environment()
    # env = MypreyWrapper_v2(env, cfg)
    env = MypreyWrapper(env, cfg)
    # env = uncertainty_wrapper_predator(env, cfg)
    # env = EntropyCalculationWrapper(
    #     env,
    #     alpha=1.0,
    #     beta=1.0
    # )
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env.max_episode_steps = env._max_episode_steps
    return env

def make_prey_env(cfg):
    gym.logger.set_level(40)
    env = make_env(cfg, max_episode_steps=cfg.episode_length)
    env = TensorWrapper(env)
    cfg.obs_shape = {'state': env.observation_space.shape}
    cfg.action_dim = int(env.action_space.shape[0])
    # show the type of action space
    if isinstance(env.action_space, (gym.spaces.Box, gymnasium.spaces.Box)):
        print('Continuous action space')
    else:
        raise ValueError('Unknown action space type')
    print(type(cfg.action_dim))
    return env
