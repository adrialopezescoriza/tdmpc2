import numpy as np
import gymnasium as gym

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from envs.wrappers.vectorized import Vectorized
from envs.wrappers.mw_stages import getRewardWrapper

class MetaWorldWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.camera_name = "corner2"
		self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
		self.env._freeze_rand_vec = False
		self.max_episode_steps = cfg.max_episode_steps

	def reset(self, **kwargs):
		self._t = 0
		obs, info = super().reset(**kwargs)
		obs = obs.astype(np.float32)
		self.env.step(np.zeros(self.env.action_space.shape))
		return obs, info

	def step(self, action):
		reward = 0
		for _ in range(self.cfg.action_repeat):
			obs, r, terminated, _, info = self.env.step(action.copy())
			reward = r # Options: max, sum, min
		obs = obs.astype(np.float32)
		self._t += 1
		done = self._t >= self.max_episode_steps
		return obs, reward, terminated, done, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.render(*args, **kwargs).copy()

def _make_env(cfg):
	"""
	Make Meta-World environment.
	"""
	env_id = cfg.task.split("-")[1] + "-v2-goal-observable"
	if not cfg.task.startswith('mw-') or env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
		raise ValueError('Unknown task:', cfg.task)
	cfg.metaworld.reward_mode = cfg.task.split("-")[-1]
	if cfg.metaworld.reward_mode == "semi":
		cfg.metaworld.reward_mode = "semi_sparse"
	assert cfg.obs == 'state', 'This task only supports state observations.'
	env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=cfg.seed, render_mode=cfg.metaworld.render_mode)
	env = getRewardWrapper(env_id)(env, cfg.metaworld)
	env = MetaWorldWrapper(env, cfg.metaworld)
	return env

def make_env(cfg):
	"""
	Make Vectorized Meta-World environment.
	"""
	env = Vectorized(cfg, _make_env)
	return env