import os
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')
import copy
import hydra
from cellworld_game.video import save_video_output
import numpy as np
import torch
from termcolor import colored
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_prey_env
from tdmpc2 import TDMPC2
import pandas as pd
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task evaluation)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=mt80 model_size=1 checkpoint=/path/to/final_model.pt
		$ python evaluate.py task=mt30 model_size=3 checkpoint=/path/to/final_model_caucious.pt
	```
	"""
	# assert torch.cuda.is_available()
	# assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	# set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.model_size}', 'blue', attrs=['bold']))
	# print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	# Make environment
	env = make_prey_env(cfg)
	# Load agent
	print(colored('Loading agent...', 'yellow', attrs=['bold']))
	agent = TDMPC2(cfg)
	print(os.getcwd())
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)

	print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	scores = []
	tasks = [cfg.task]
	for task_idx, task in enumerate(tasks):
		task_idx = None
		ep_rewards, ep_successes = [], []
		# Q_var_list, ep_list= [], []
		obs_list = []
		action_list = []
		reward_list = []
		done_list = []
		next_obs_list = []
		prediction_error_list = []  # Add list to store prediction errors
		episode_list = []  # Add episode tracking
		step_list = []     # Add step tracking within each episode
		
		for i in tqdm(range(cfg.eval_episodes), desc='Evaluating episodes'):
			# save_video_output(env.model, "/Users/hanshuo/Documents/project/RL/tlppo_cellworld_evade/video")
			obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
			while not done:
				# Check if MPC is enabled and handle prediction error
				if hasattr(cfg, 'mpc') and cfg.mpc:
					action, prediction_error = agent.act(obs, t0=t==0, task=task_idx)
					prediction_error_list.append(prediction_error.item())
				else:
					action = agent.act(obs, t0=t==0, task=task_idx)
					prediction_error_list.append(0.0)  # Default value when MPC is disabled
				
				# add noise to action
				# action = action + np.random.normal(0, 0.2, size=action.shape)
				# ep_list.append(i+1)
				copied_obs = copy.deepcopy(obs.numpy())
				new_obs, reward, done, info = env.step(action)
				
				# Convert numpy arrays to flattened lists or strings for CSV compatibility
				obs_list.append(copied_obs.flatten().tolist())  # Flatten and convert to list
				action_list.append(action.numpy().flatten().tolist())  # Flatten and convert to list
				# save reward as float, now it is tensor
				reward_list.append(reward.item())
				done_list.append(done)
				next_obs_list.append(copy.deepcopy(new_obs.numpy()).flatten().tolist())  # Flatten and convert to list
				episode_list.append(i + 1)  # Track episode number
				step_list.append(t + 1)      # Track step number within episode
				
				obs = new_obs
				ep_reward += reward
				t += 1
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
		ep_rewards = np.mean(ep_rewards)
		ep_successes = np.mean(ep_successes)
		data = {
			"episode": episode_list,
			"step": step_list,
			"obs": obs_list,
			"action": action_list,
			"reward": reward_list,
			"done": done_list,
			"next_obs": next_obs_list,
			"prediction_error": prediction_error_list  # Add prediction error to saved data
		}
		# data = {
		# 	"ep": ep_list,
		# 	"Q_var": Q_var_list
		# }
		df = pd.DataFrame(data)
		# print(obs_list)
		data_name = cfg.save_name
		df.to_csv(f"/home/shv7753/tlppo_cellworld_evade/alex_data/{data_name}.csv", index=False)
		print(colored(f'  {task:<22}' \
			f'\tR: {ep_rewards:.01f}  ' \
			f'\tS: {ep_successes:.02f}', 'yellow'))

if __name__ == '__main__':
	evaluate()