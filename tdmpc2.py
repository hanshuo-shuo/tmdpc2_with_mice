import numpy as np
import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel


class TDMPC2:
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr * self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._Qs.parameters()},
			{'params': []}
		], lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2 * int(cfg.action_dim >= 20)  # Heuristic for large action spaces
		self.discount = self._get_discount(cfg.episode_length)

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length / self.cfg.discount_denom
		return min(max((frac - 1) / (frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp, map_location=torch.device('cpu'))
		self.model.load_state_dict(state_dict["model"])

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		new_obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
	
		# Check if MPC prediction error calculation is enabled
		if hasattr(self.cfg, 'mpc') and self.cfg.mpc:
			z = self.model.encode(new_obs, task)
			a, predicted_next_z = self.plan(z, t0=t0, eval_mode=eval_mode, task=task, return_prediction=True)
			# Calculate prediction error by comparing predicted next state with current state encoding
			current_z = self.model.encode(new_obs, task)
			prediction_error = F.mse_loss(predicted_next_z, current_z)
			return a.cpu(), prediction_error
	
		z = self.model.encode(new_obs, task)
		a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)

		return a.cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory."""
		# Regular value estimation for the trajectory
		G, discount = 0, 1
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * reward
			discount = discount * self.discount
		
		# Get final value using policy actions
		pi_actions = self.model.pi(z, task)[1]
		final_q = self.model.Q(z, pi_actions, task, return_type='min')
		final_value = G + discount * final_q
		
		return final_value

	@torch.no_grad()
	def plan(self, z, t0=False, eval_mode=False, task=None, return_prediction=False):
		"""
		Plan a sequence of actions using the learned world model.
		"""
		# Store original z for prediction if needed
		original_z = z.clone() if return_prediction else None
		
		# Sample policy trajectories
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon - 1):
				pi_actions[t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1] = self.model.pi(_z, task)[1]

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std * torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		Q_values_iteration = []
		
		# Iterate MPPI
		for i in range(self.cfg.iterations):
			# Sample actions
			actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
												  torch.randn(self.cfg.horizon,
															  self.cfg.num_samples - self.cfg.num_pi_trajs,
															  self.cfg.action_dim, device=std.device)) \
				.clamp(0, 1)

			# Compute elite actions using pre-computed variance penalty
			value = self._estimate_value(z, actions, task)
			value = value.nan_to_num_(0)

			# 获取当前状态(t=0)的Q值
			current_state_values = self.model.Q(z, actions[0], task, return_type='avg')
			current_state_variance = torch.var(current_state_values)
			# print(f"Iteration {i}, Current state Q-value variance: {current_state_variance.item():.4f}")
			Q_values_iteration.append(current_state_variance.item())

			# Continue with existing code
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature * (elite_value - max_value))
			score /= score.sum(0)
			mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			std = torch.sqrt(
				torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
				.clamp_(self.cfg.min_std, self.cfg.max_std)

		# Select action
		score = score.squeeze(1).cpu().numpy()
		selected_idx = np.random.choice(np.arange(score.shape[0]), p=score)
		selected_actions = elite_actions[:, selected_idx]
		self._prev_mean = mean
		a, std = selected_actions[0], std[0]
		if not eval_mode:
			a += std * torch.randn(self.cfg.action_dim, device=std.device) * 0.1
		
		if return_prediction:
			# Predict the next state using the selected action and original single state
			predicted_next_z = self.model.next(original_z, a.unsqueeze(0), task)
			return a.clamp_(0, 1), predicted_next_z
		else:
			return a.clamp_(0, 1)

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)
		_, pis, log_pis, _ = self.model.pi(zs, task)
		qs = self.model.Q(zs, pis, task, return_type='avg')
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.model.track_q_grad(True)

		return pi_loss.item()
	
	@torch.no_grad()
	def _td_target(self, next_z, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		pi = self.model.pi(next_z, task)[1]
		discount = self.discount
		return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)
	

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, task = buffer.sample()

		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task)

		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon + 1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t in range(self.cfg.horizon):
			z = self.model.next(z, action[t], task)
			consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho ** t
			zs[t + 1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t in range(self.cfg.horizon):
			reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho ** t
			for q in range(self.cfg.num_q):
				value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho ** t
		consistency_loss *= (1 / self.cfg.horizon)
		reward_loss *= (1 / self.cfg.horizon)
		value_loss *= (1 / (self.cfg.horizon * self.cfg.num_q))
		total_loss = (
				self.cfg.consistency_coef * consistency_loss +
				self.cfg.reward_coef * reward_loss +
				self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()

		# Update policy
		pi_loss = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value),
		}