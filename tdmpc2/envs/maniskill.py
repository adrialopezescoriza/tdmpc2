import gymnasium as gym
import numpy as np
import torch
from envs.utils import convert_observation_to_space
from envs.wrappers.drS_reward import DrsRewardWrapper
from mani_skill.utils.common import flatten_state_dict

import mani_skill.envs
import envs.tasks.maniskill_stages


MANISKILL_TASKS = {
	'lift-cube': dict(
		env='LiftCube-v1',
		control_mode='pd_ee_delta_pose',
	),
	'pick-cube': dict(
		env='PickCube-v1',
		control_mode='pd_ee_delta_pose',
	),
	'pick-ycb': dict(
		env='PickSingleYCB-v1',
		control_mode='pd_ee_delta_pose',
	),
	'turn-faucet': dict(
		env='TurnFaucet-v1',
		control_mode='pd_ee_delta_pose',
	),
	'pick-place': dict(
		env='PickAndPlace_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense',
	),
	'stack-cube': dict (
		env='StackCube_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense', 
	),
	'peg-insertion': dict(
		env='PegInsertionSide_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense',
	),
	'lift-peg-upright': dict(
		env='LiftPegUpright_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense',
	),
	'poke-cube': dict(
		env='PokeCube_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense',
	),
	## Semi-sparse reward tasks with stage-indicators
	'pick-place-semi': dict (
		env='PickAndPlace_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse', 
	),
	'stack-cube-semi': dict (
		env='StackCube_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse', 
	),
	'peg-insertion-semi': dict (
		env='PegInsertionSide_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse', 
	),
	'lift-peg-upright-semi': dict(
		env='LiftPegUpright_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse',
	),
	'poke-cube-semi': dict(
		env='PokeCube_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse', 
	),
	'pick-place-drS': dict (
		env='PickAndPlace_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='drS', 
	),
	'stack-cube-drS': dict (
		env='StackCube_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='drS', 
	),
	'peg-insertion-drS': dict (
		env='PegInsertionSide_DrS_learn',
		control_mode='pd_ee_delta_pose',
		reward_mode='drS', 
	),
}

def select_obs(keys, obs):
	"""
	Processes observations on the first nested level of the obs dictionary

	Args:
		keys: The keys
		obs: An array or dictionary of more nested observations or observation spaces 
	"""
	if not isinstance(obs, dict):
		return obs
	processed = dict()
	for k in keys:
		if k == "agent":
			# Stack all states
			processed["state"] = flatten_state_dict(obs[k], use_torch=True)
		elif k == "image":
			# Only take rgb + Put channel dimension first
			processed["rgb_base"] = obs['sensor_data']['base_camera']['rgb'].permute(0,3,1,2)
			# processed["rgb_ext"] = obs['sensor_data']['ext_camera']['rgb'].permute(0,3,1,2)
			processed["rgb_hand"] = obs['sensor_data']['hand_camera']['rgb'].permute(0,3,1,2)
		else:
			return NotImplementedError
	return processed

class ManiSkillWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.action_space = env.single_action_space
		self.max_episode_steps = cfg.max_episode_steps

		self.obs_keys = cfg.get("obs_keys", None)

		if hasattr(self.env.observation_space, 'spaces'):
			# Dict
			obs_sample = {k: v[0] for k, v in self.get_obs().items()}
			self.observation_space = convert_observation_to_space(obs_sample)
		else:
			self.observation_space = gym.spaces.Box(
				low=np.full(
					env.single_observation_space.shape,
					-np.inf,
					dtype=np.float32),
				high=np.full(
					env.single_observation_space.shape,
					np.inf,
					dtype=np.float32),
				dtype=np.float32,
			)

	def rand_act(self):
		return torch.tensor(
			[self.action_space.sample().astype(np.float32) for _ in range(self.num_envs)],
			dtype=torch.float32, device=self.env.device)

	def reset(self, seed=None):
		self._t = 0
		obs, info = self.env.reset(seed=seed, options=None)
		return (select_obs(self.obs_keys, obs) if isinstance(obs, dict) else obs), info
	
	def step(self, action):
		for _ in range(self.cfg.action_repeat):
			obs, r, terminated, _, info = self.env.step(action)
			reward = r # Options: max, sum, min
		if isinstance(obs, dict):
			obs = select_obs(self.obs_keys, obs)
		self._t += 1
		done = torch.tensor([self._t >= self.max_episode_steps] * self.num_envs)
		return obs, reward, terminated, done, info
	
	def reward(self, **kwargs):
		return self.env.get_reward(obs=self.env.get_obs(), action=self.env.action_space.sample(), info=self.env.get_info())
	
	def get_obs(self, *args, **kwargs):
		return select_obs(self.obs_keys, self.env.get_obs())

	@property
	def unwrapped(self):
		return self.env.unwrapped
	
	def render(self, *args, **kwargs):
		if kwargs.get("render_all", False):
			return self.env.render()
		return self.env.render()[0].cpu().numpy()


def make_env(cfg):
	"""
	Make ManiSkill2 environment.
	"""
	if cfg.task not in MANISKILL_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	task_cfg = MANISKILL_TASKS[cfg.task]
	camera_resolution = dict(width=cfg.maniskill.camera.get("image_size", 64), height=cfg.maniskill.camera.get("image_size", 64))

	# WARNING: If one env is already in GPU, the other ones must also be in GPU
	env = gym.make(
		task_cfg['env'],
		obs_mode=cfg.obs,
		control_mode=task_cfg['control_mode'],
		num_envs=cfg.num_envs,
		reward_mode=task_cfg.get("reward_mode", None),
		render_mode='rgb_array',
		sensor_configs=camera_resolution,
		human_render_camera_configs=dict(width=384, height=384),
		reconfiguration_freq=1 if cfg.num_envs > 1 else None,
		sim_backend=cfg.maniskill.get("sim_backend", "auto"),
		render_backend="auto",
	)

	cfg.action_penalty = cfg.maniskill.action_penalty

	# DrS Reward Wrapper
	if task_cfg.get("reward_mode", None) == "drS":
		env = DrsRewardWrapper(env, cfg.drS_ckpt)
	env = ManiSkillWrapper(env, cfg.maniskill)
	return env
