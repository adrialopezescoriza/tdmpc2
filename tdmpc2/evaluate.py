import os
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.trajectory_saver import BaseTrajectorySaver
from envs import make_env, make_single_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True

def make_video(env, agent, task_idx):
	video_obs, done, t = env.reset(task_idx=task_idx), False, 0
	frames = [env.render()]
	while not done:
		video_action = agent.act(video_obs.unsqueeze(0), t0=t==0, task=task_idx)
		video_obs, _, done, _ = env.step(video_action.squeeze())
		frames.append(env.render())
	return frames

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

	# Make environment
	env = make_env(cfg)
	video_env = make_single_env(cfg, video_only=True) if cfg.save_video else None

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
	scores = []
	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	for task_idx, task in enumerate(tasks):
		if not cfg.multitask:
			task_idx = None
		ep_rewards, ep_successes = [], []
		while (saver.num_traj if cfg.save_trajectory else len(ep_rewards)) < cfg.eval_episodes:
			if cfg.save_video:
				frames = make_video(video_env, agent, task_idx)
				imageio.mimsave(
					os.path.join(video_dir, f'{task}-{saver.num_traj}.mp4'), frames, fps=15)
			obs, done, ep_reward, t = env.reset(task_idx=task_idx), torch.tensor(False), 0, 0
			while not done.all():
				prev_obs = obs
				action = agent.act(obs, t0=t==0, task=task_idx, eval_mode=True)
				obs, reward, done, info = env.step(action)
				ep_reward += reward
				t += 1
				if cfg.save_trajectory:
					terminated = done # Only terminate when truncated
					saver.add_transition(
						prev_obs.numpy(),
						action.numpy(),
						obs.numpy(),
						reward.numpy(),
						terminated.numpy(),
						[dict(zip(info,t)) for t in zip(*info.values())])
			ep_rewards.append(ep_reward.tolist())
			ep_successes.append(info['success'].tolist())
		if cfg.save_trajectory:
			saver.save(env_id=task)
	
		ep_rewards = np.mean(ep_rewards)
		ep_successes = np.mean(ep_successes)
		if cfg.multitask:
			scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)
		print(colored(f'  {task:<22}' \
			f'\tR: {ep_rewards:.01f}  ' \
			f'\tS: {ep_successes:.02f}', 'yellow'))
	if cfg.multitask:
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	evaluate()
