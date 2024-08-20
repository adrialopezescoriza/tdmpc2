from collections import defaultdict
from tensordict.tensordict import TensorDict

import gymnasium as gym
import numpy as np
import torch


class TensorWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""
	def __init__(self, env):
		super().__init__(env)

	def _try_f32_tensor(self, x):
		x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
		if x.dtype == torch.float64:
			x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
			obs = TensorDict(obs, batch_size=self.num_envs)
		else:
			obs = self._try_f32_tensor(obs)
		return obs

	def reset(self, task_idx=None, **kwargs):
		obs = self.env.reset(**kwargs)
		return self._obs_to_tensor(obs)
	
	def render(self, *args, **kwargs):
		return self.env.render(*args, **kwargs)

	def step(self, action, **kwargs):
		obs, reward, done, info = self.env.step(action)
		obs = self._obs_to_tensor(obs)
		if isinstance(info, tuple):
			info = {key: torch.stack([torch.tensor(d[key]) for d in info]) for key in info[0].keys()}
			if 'success' not in info.keys():
				info['success'] = torch.zeros(len(done))
		else:
			info = defaultdict(float, info)
			info['success'] = info['success'].float()
		return obs, torch.tensor(reward, dtype=torch.float32), torch.tensor(done), info
	
	def get_obs(self):
		return self._obs_to_tensor(self.env.get_obs())
