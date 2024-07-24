import gym
import numpy as np
from envs.wrappers.time_limit import TimeLimit
from envs.wrappers.drS_reward import DrsRewardWrapper

import mani_skill2.envs
import envs.tasks.envs_with_stage_indicators


MANISKILL_TASKS = {
	'lift-cube': dict(
		env='LiftCube-v0',
		control_mode='pd_ee_delta_pose',
	),
	'pick-cube': dict(
		env='PickCube-v0',
		control_mode='pd_ee_delta_pose',
	),
	'stack-cube': dict(
		env='StackCube-v0',
		control_mode='pd_ee_delta_pose',
	),
	'pick-ycb': dict(
		env='PickSingleYCB-v0',
		control_mode='pd_ee_delta_pose',
	),
	'turn-faucet': dict(
		env='TurnFaucet-v0',
		control_mode='pd_ee_delta_pose',
	),
	'pick-place': dict(
		env='PickAndPlace_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense',
	),
	'peg-insertion': dict(
		env='PegInsertionSide_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense',
	),
	'open-cabinet': dict(
		env='OpenCabinetDoor_DrS_learn-v0',
		control_mode='base_pd_joint_vel_arm_pd_joint_vel',
		reward_mode='dense',
	),
	## Semi-sparse reward tasks with stage-indicators
	'pick-place-semi': dict (
		env='PickAndPlace_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse', 
	),
	'turn-faucet-semi': dict (
		env='TurnFaucet_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse', 
	),
	'stack-cube-semi': dict (
		env='StackCube_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse', 
	),
	'peg-insertion-semi': dict (
		env='PegInsertionSide_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse', 
	),
	'open-cabinet-semi': dict(
		env='OpenCabinetDoor_DrS_learn-v0',
		control_mode='base_pd_joint_vel_arm_pd_joint_vel',
		reward_mode='semi_sparse',
	),
	'pick-place-drS': dict (
		env='PickAndPlace_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='drS', 
	),
	'turn-faucet-drS': dict (
		env='TurnFaucet_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='drS', 
	),
	'stack-cube-drS': dict (
		env='StackCube_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='drS', 
	),
	'peg-insertion-drS': dict (
		env='PegInsertionSide_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='drS', 
	),
	'open-cabinet-drS': dict(
		env='OpenCabinetDoor_DrS_learn-v0',
		control_mode='base_pd_joint_vel_arm_pd_joint_vel',
		reward_mode='drS',
	),
}

def flatten_space(space):
	obs_shp = []
	for v in space.values():
		try:
			shp = np.prod(v.shape)
		except:
			shp = 1
		obs_shp.append(shp)
	obs_shp = (int(np.sum(obs_shp)),)
	return gym.spaces.Box(
		low=np.full(
			obs_shp,
			-np.inf,
			dtype=np.float32),
		high=np.full(
			obs_shp,
			np.inf,
			dtype=np.float32),
		dtype=np.float32,
	)

def select_obs(keys, obs):
	"""
	Processes observations on the first nested level of the obs dictionary

	Args:
		keys: The keys
		obs: An array or dictionary of more nested observations or observation spaces 
	"""
	processed = dict()
	is_space = isinstance(next(iter(obs.values())), gym.spaces.Space)
	flatten = lambda x: flatten_space(x) if is_space else np.concatenate(list(x.values()))

	for k in keys:
		if k == "agent":
			# Stack all states
			processed["state"] = flatten(obs[k])
		elif k == "image":
			# Only take base camera rgb + Put channel dimension first
			if is_space:
				shp = obs[k]['base_camera']['rgb'].shape
				processed["rgb"] = gym.spaces.Box(
					low=0, high=255, shape=(shp[-1],) + shp[:-1], dtype=np.uint8
				)
			else:
				processed["rgb"] = obs[k]['base_camera']['rgb'].transpose(2,0,1)
		else:
			return NotImplementedError
	return processed

class ManiSkillWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.action_space = gym.spaces.Box(
			low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
			high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
			dtype=self.env.action_space.dtype,
		)

		self.obs_keys = cfg.get("obs_keys", None)

		if hasattr(self.env.observation_space, 'spaces'):
			# Dict
			self.observation_space = gym.spaces.Dict(select_obs(self.obs_keys, self.env.observation_space.spaces))
		else:
			self.observation_space = self.env.observation_space

	def reset(self):
		obs = self.env.reset()
		return select_obs(self.obs_keys, obs) if isinstance(obs, dict) else obs

	
	def step(self, action):
		for _ in range(self.cfg.action_repeat):
			obs, r, _, info = self.env.step(action)
			reward = r # Options: max, sum, min
		if isinstance(obs, dict):
			obs = select_obs(self.obs_keys, obs)
		return obs, reward, False, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, args, **kwargs):
		return self.env.render(mode='cameras')


def make_env(cfg):
	"""
	Make ManiSkill2 environment.
	"""
	if cfg.task not in MANISKILL_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	task_cfg = MANISKILL_TASKS[cfg.task]
	env = gym.make(
		task_cfg['env'],
		obs_mode=cfg.obs,
		control_mode=task_cfg['control_mode'],
		render_camera_cfgs=dict(width=384, height=384),
		reward_mode=task_cfg.get("reward_mode", None),
		camera_cfgs=dict(width=cfg.maniskill.camera.get("render_size", 64), height=cfg.maniskill.camera.get("render_size", 64)),
	)
	
	# DrS Reward Wrapper
	if task_cfg.get("reward_mode", None) == "drS":
		env = DrsRewardWrapper(env, cfg.drS_ckpt)
	
	env = ManiSkillWrapper(env, cfg.maniskill)
	env = TimeLimit(env, max_episode_steps=100)
	env.max_episode_steps = env._max_episode_steps
	return env
