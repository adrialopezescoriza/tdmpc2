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
	'stack-cube': dict(
		env='StackCube_DrS_reuse-v0',
		control_mode='pd_ee_delta_pos',
		reward_mode='dense',
	),
	'peg-insertion': dict(
		env='PegInsertionSide_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
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
		control_mode='pd_ee_delta_pos',
		reward_mode='semi_sparse', 
	),
	'peg-insertion-semi': dict (
		env='PegInsertionSide_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
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
		control_mode='pd_ee_delta_pos',
		reward_mode='drS', 
	),
	'peg-insertion-drS': dict (
		env='PegInsertionSide_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='drS', 
	),
}


class ManiSkillWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.observation_space = self.env.observation_space
		self.action_space = gym.spaces.Box(
			low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
			high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
			dtype=self.env.action_space.dtype,
		)

	def reset(self):
		return self.env.reset()
	
	def step(self, action):
		# TODO: Revisit reward compunding with action repeat, this may not be the best way
		reward = -np.inf
		for _ in range(self.cfg.action_repeat):
			obs, r, _, info = self.env.step(action)
			reward = max(reward, r)
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
	assert cfg.obs == 'state', 'This task only supports state observations.'
	task_cfg = MANISKILL_TASKS[cfg.task]
	env = gym.make(
		task_cfg['env'],
		obs_mode='state',
		control_mode=task_cfg['control_mode'],
		render_camera_cfgs=dict(width=384, height=384),
		reward_mode=task_cfg.get("reward_mode", None),
	)
	
	# DrS Reward Wrapper
	if task_cfg.get("reward_mode", None) == "drS":
		env = DrsRewardWrapper(env, cfg.drS_ckpt)
	
	env = ManiSkillWrapper(env, cfg.maniskill)
	env = TimeLimit(env, max_episode_steps=100)
	env.max_episode_steps = env._max_episode_steps
	return env
