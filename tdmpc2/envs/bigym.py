import numpy as np
import gymnasium as gym
import torch

from envs.tasks.bigym_stages import SUPPORTED_TASKS
from envs.wrappers.vectorized import Vectorized
from gymnasium.wrappers.rescale_action import RescaleAction

from envs.utils import convert_observation_to_space

class BiGymWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.observation_space = convert_observation_to_space(self.select_obs(self.get_observation()))

	def select_obs(self, obs):
		if self.cfg.obs == "state":
			return np.concatenate([v for v in obs.values()])
		processed = {"state": np.empty((0,))}
		for k, v in obs.items():
			if k.startswith("proprioception"):
				processed["state"] = np.concatenate((processed["state"], v))
			elif k.startswith("rgb"):
				processed[k] = v
			else:
				raise NotImplementedError
		return processed
	
	def rand_act(self):
		return self.action_space.sample().astype(np.float32)

	def reset(self, **kwargs):
		self._t = 0
		obs, info = super().reset(**kwargs)
		return self.select_obs(obs), info
	
	def get_penalties(self):
		return self.env.joint_vel_penalty()

	def step(self, action):
		reward = 0
		action = np.clip(action, self.action_space.low, self.action_space.high)
		action = action.numpy() if isinstance(action, torch.Tensor) else action
		for _ in range(self.cfg.action_repeat):
			obs, r, terminated, _, info = self.env.step(action)
			reward = r # Options: max, sum, min
		self._t += 1
		done = self._t >= self.max_episode_steps
		info["success"] = info["task_success"]
		return self.select_obs(obs), reward, False, done, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.render(*args, **kwargs)
	
	def get_obs(self, *args, **kwargs):
		return self.select_obs(self.get_observation())

def _make_env(cfg):
	"""
	Make Meta-World environment.
	"""
	env_id = cfg.task.split("-",1)[-1]
	if not cfg.task.startswith('bigym-') or env_id not in SUPPORTED_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	env = SUPPORTED_TASKS[env_id](
			obs_mode=cfg.obs, 
		    img_size=cfg.bigym.camera.image_size,
            render_mode=cfg.bigym.render_mode,
			start_seed=cfg.seed, 
		)
	cfg.bigym.obs = cfg.obs
	env = RescaleAction(env, -1.0, 1.0)
	env = BiGymWrapper(env, cfg.bigym)
	return env

def make_env(cfg):
	"""
	Make Vectorized BiGym environment.
	"""
	env = Vectorized(cfg, _make_env)
	cfg.action_penalty = cfg.bigym.get("action_penalty", False)
	return env