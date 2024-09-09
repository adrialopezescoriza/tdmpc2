import os
os.environ['MUJOCO_GL'] = 'osmesa'

import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm
from copy import deepcopy

from common.parser import parse_cfg
from common.seed import set_seed
from common.trajectory_saver import BaseTrajectorySaver
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True

class ObservationConverter(object):
	
	def __init__(self, cfg):
		self.obs_flag = (cfg.obs != cfg.obs_save)
		self.env_obs = None
		self.obs_type = cfg.obs_save
		if cfg.task.startswith("mw"):
			self.env_type = "metaworld"
		else:
			self.env_type = "maniskill"
			if self.obs_flag:
				cfg_obs = deepcopy(cfg)
				cfg_obs.obs = cfg_obs.obs_save
				self.env_obs = make_env(cfg_obs)
	
	def get_obs(self, env):
		if self.env_obs is not None:
			self.env_obs.set_state_dict(env.get_state_dict())
			return self.env_obs.get_obs()
		return env.get_obs(self.obs_type)
	
	def get_frame(self, env, obs):
		if hasattr(obs, "keys"):
			frame = None
			for k, v in obs.items():
				if k.startswith('rgb'):
					frame_ = v[0].permute(1,2,0).cpu().numpy()
					frame = frame_ if frame is None else np.concatenate((frame, frame_), axis=1)
			return frame
		return env.render()
	
	def reset(self, task_idx, seed, env):
		if self.env_obs is not None:
			return self.env_obs.reset(task_idx=task_idx, seed=seed)
		return env.get_obs(self.obs_type)

@hydra.main(config_name='eval', config_path='./config/')
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
		$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
		$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
		print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
		print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	print(f"Simulated observation: {cfg.obs}.")
	print(f"Saved observation: {cfg.obs_save}.")

	# Make agent environment
	env = make_env(cfg)

	# Observation converter
	obs_converter = ObservationConverter(cfg)

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)

	# Trajectory saver
	if cfg.save_trajectory:
		saver = BaseTrajectorySaver(cfg.num_envs, cfg.log_path, cfg.success_only, cfg.eval_episodes)
	
	# Evaluate
	if cfg.multitask:
		print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	if cfg.save_video:
		video_dir = os.path.join(cfg.log_path, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	tasks = [cfg.task]
	for task_idx, task in enumerate(tasks):
		if not cfg.multitask:
			task_idx = None
		ep_rewards, ep_successes = [], []
		with tqdm(total=cfg.eval_episodes) as pbar:
			while (saver.num_traj if cfg.save_trajectory else len(ep_rewards)) < cfg.eval_episodes:
				seed = np.random.RandomState().randint(2**32)
				obs, done, ep_reward, t = env.reset(task_idx=task_idx, seed=seed), torch.tensor(False), 0, 0
				obs_save = obs_converter.reset(task_idx, seed, env)
				if cfg.save_video:
					frames = [obs_converter.get_frame(env, obs_save)]
				while not done.all():
					prev_obs_save = deepcopy(obs_save)
					action = agent.act(obs, t0=t==0, task=task_idx, eval_mode=True).to(obs.device)
					obs, reward, done, info = env.step(action)
					obs_save = obs_converter.get_obs(env)
					ep_reward += reward
					t += 1
					if cfg.save_video:
						frames.append(obs_converter.get_frame(env, obs_save))
					if cfg.save_trajectory:
						terminated = done # Only terminate when truncated
						info = {k: v.cpu() for k,v in info.items()}
						saver.add_transition(
							prev_obs_save.cpu(),
							action.cpu(),
							obs_save.cpu(),
							reward.cpu(),
							terminated.cpu(),
							[dict(zip(info,t)) for t in zip(*info.values())])
				ep_rewards.append(ep_reward.tolist())
				ep_successes.append(info['success'].tolist())
				if cfg.save_video:
					imageio.mimsave(
						os.path.join(video_dir, f'{task}-{saver.num_traj}.mp4'), frames, fps=15)
				pbar.update((saver.num_traj if cfg.save_trajectory else len(ep_rewards)) - pbar.n)
		pbar.close()
		env.close()
		if cfg.save_trajectory:
			saver.save(env_id=task)
	
		ep_rewards = np.mean(ep_rewards)
		ep_successes = np.mean(ep_successes)
		print(colored(f'  {task:<22}' \
			f'\tR: {ep_rewards:.01f}  ' \
			f'\tS: {ep_successes:.02f}', 'yellow'))

if __name__ == '__main__':
	evaluate()
