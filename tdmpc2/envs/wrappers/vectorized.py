from copy import deepcopy

import gymnasium as gym

from gymnasium.vector import AsyncVectorEnv
import numpy as np
import torch

class Vectorized(gym.Wrapper):
	"""
	Vectorized environment for TD-MPC2 online training.
	"""

	def __init__(self, cfg, env_fn):
		self.cfg = cfg
		self.num_envs = cfg.num_envs

		def make():
			_cfg = deepcopy(cfg)
			_cfg.num_envs = 1
			_cfg.seed = cfg.seed + np.random.randint(1000)
			return env_fn(_cfg)

		print(f'Creating {cfg.num_envs} environments...')
		self.env = AsyncVectorEnv([make for _ in range(cfg.num_envs)])
		env = make()

		self.observation_space = env.observation_space
		self.action_space = env.action_space
		self.max_episode_steps = env.max_episode_steps

		# DrS specific
		if hasattr(env, "n_stages"):
			self.reward_mode = env.reward_mode
			self.n_stages = env.n_stages

	def rand_act(self):
		return torch.rand((self.cfg.num_envs, *self.action_space.shape)) * 2 - 1

	def reset(self):
		return self.env.reset()

	def step(self, action):
		obs, r, terminated, truncated, info = self.env.step(action.numpy())
		if "final_observation" in info.keys():
			obs = np.stack(info["final_observation"],axis=0)
			info = {k: [dic[k] for dic in info["final_info"]] for k in info["final_info"][0]}
		return obs, r, terminated, truncated, info 

	def render(self, *args, **kwargs):
		return self.env.call("render")[0]