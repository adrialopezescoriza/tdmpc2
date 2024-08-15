from collections import defaultdict
from tensordict.tensordict import TensorDict

import gym
import numpy as np
import torch


class TensorWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env):
		super().__init__(env)
		self._wrapped_vectorized = env.__class__.__name__ == 'Vectorized'
	
	def rand_act(self):
		if self._wrapped_vectorized:
			return self.env.rand_act()
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _try_f32_tensor(self, x):
		x = torch.from_numpy(x)
		if x.dtype == torch.float64:
			x = x.float()
		return x

	def _obs_to_tensor(self, obs, bs=()):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
			obs = TensorDict(obs, batch_size=bs)
		else:
			obs = self._try_f32_tensor(obs)
		return obs

	def reset(self, task_idx=None, **kwargs):
		if self._wrapped_vectorized:
			obs = self.env.reset(**kwargs)
			return self._obs_to_tensor(obs, self.cfg.num_envs)

		return self._obs_to_tensor(self.env.reset(**kwargs))
	
	def render(self, *args, **kwargs):
		return self.env.render(*args, **kwargs)

	def step(self, action, **kwargs):
		if self._wrapped_vectorized:
			obs, reward, done, info = self.env.step(action.numpy(), **kwargs)
			obs = self._obs_to_tensor(obs, self.cfg.num_envs)
		else:
			obs, reward, done, info = self.env.step(action.numpy())
			obs = self._obs_to_tensor(obs)
		if isinstance(info, tuple):
			info = {key: torch.stack([torch.tensor(d[key]) for d in info]) for key in info[0].keys()}
			if 'success' not in info.keys():
				info['success'] = torch.zeros(len(done))
		else:
			info = defaultdict(float, info)
			info['success'] = float(info['success'])
		return obs, torch.tensor(reward, dtype=torch.float32), torch.tensor(done), info
	
	def get_obs(self):
		if self._wrapped_vectorized:
			obs = self.env.get_obs()
			return self._obs_to_tensor(obs, self.cfg.num_envs)

		return self._obs_to_tensor(self.env.get_obs())
