from time import time

import numpy as np
import torch
import termcolor
from tensordict.tensordict import TensorDict
from functools import partial

from .discriminator import Discriminator
from .disc_buffer import DiscriminatorBuffer
from trainer.base import Trainer


class DrsTrainer(Trainer):
	"""Trainer class for DrS training. Assumes semi-sparse reward environment."""

	def __init__(self, cfg, env, agent, buffer, logger):
		self.cfg = cfg
		self.env = env
		self.agent = agent
		self.replay_buffer = buffer
		self.logger = logger

		assert env.reward_mode in ["semi_sparse","drS"], "Reward mode is incompatible with DrS"

		# DrS specific
		self.disc = Discriminator(env, cfg.drS_discriminator, state_shape=(cfg.latent_dim,))
		self.stage_buffers = [DiscriminatorBuffer(
			cfg.drS_discriminator.buffer_size,
			env.observation_space,
			env.action_space,
			self.agent.device,
		) for _ in range(env.n_stages + 1)]

		if cfg.demo_path:
			from .data_utils import load_demo_dataset, load_dataset_as_td
			demo_dataset = load_dataset_as_td(cfg.demo_path)
			self.stage_buffers[-1].add(next_obs=torch.cat(demo_dataset)['obs'])

			if cfg.prefill_buffer_with_demos:
				[self.replay_buffer.add(_td.unsqueeze(0)) for _td in demo_dataset]
				termcolor.colored(f"Prefilled buffer with {len(demo_dataset)} trajectories", "green")

		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		
		print('Agent Architecture:', self.agent.model)
		print('Discriminator Architecture:', self.disc)
		print("Learnable parameters: {:,}".format(self.agent.model.total_params + self.disc.total_params))

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate agent."""
		ep_rewards, ep_max_rewards = [], []
		for i in range(max(1, self.cfg.eval_episodes  // self.cfg.num_envs)):
			obs, done, ep_reward, ep_max_reward, t = self.env.reset(), torch.tensor(False), 0, None, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done.any():
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				ep_max_reward = torch.maximum(ep_max_reward, reward) if ep_max_reward is not None else reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			assert done.all(), 'Vectorized environments must reset all environments at once.'
			ep_rewards.append(ep_reward)
			ep_max_rewards.append(ep_max_reward)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=torch.cat(ep_rewards).mean(),
			episode_max_reward=torch.cat(ep_max_rewards).mean(),
			episode_success=info['success'].float().mean(),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan')).repeat(self.cfg.num_envs)
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		), batch_size=(1, self.cfg.num_envs,))
		return td

	def train(self):
		"""Train agent and discriminator"""
		train_metrics, done, eval_next = {}, torch.tensor(True), True
		while self._step <= self.cfg.steps:

			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Save DrS and Agent periodically
			if self._step % self.cfg.save_freq == 0 and self._step > 0:
					print("Saving agent and discriminator checkpoints...")
					self.logger.save_agent(self.disc, identifier=f'drS_{self._step}')
					self.logger.save_agent(self.agent, identifier=f'agent_{self._step}')

			# Reset environment
			if done.any():
				assert done.all(), 'Vectorized environments must reset all environments at once.'
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					tds = torch.cat(self._tds)
					train_metrics.update(
						episode_reward=np.nansum(tds['reward'], axis=0).mean(),
						episode_max_reward=np.nanmax(tds['reward'], axis=0).mean(),
						episode_success=info['success'].float().nanmean(),
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.replay_buffer.add(tds)

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]
				self._observations = torch.empty((self.cfg.num_envs,0) + self.env.observation_space.shape)
				best_step = [0 for _ in range(self.cfg.num_envs)]
				best_reward = [-np.inf for _ in range(self.cfg.num_envs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward))
			self._observations = torch.cat((self._observations, obs.unsqueeze(1)), dim=1)

			# Get longest possible trajectory (DrS)
			for i, r in enumerate(reward):
				if r >= best_reward[i]:
					best_reward[i] = r
					best_step[i] = len(self._tds) - 1

			# Add experiences to corresponding disc buffer (DrS)
			if done.any():
				assert done.all(), 'Vectorized environments must reset all environments at once.'
				for i in range(self.cfg.num_envs):
					if info["success"][i]:
						stage_idx = self.env.n_stages
					elif self.env.n_stages > 1:
						stage_idx = int(best_reward[i])
					else:
						stage_idx = 0
					self.stage_buffers[stage_idx].add(self._observations[i, :best_step[i]])
			
			# Update discriminator and agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = int(self.cfg.seed_steps / self.cfg.steps_per_update)
					print('Pretraining agent on seed data...')
				else:
					num_updates = max(1, int(self.cfg.num_envs / self.cfg.steps_per_update))
				for _ in range(num_updates):
					disc_train_metrics = self.disc.update(self.stage_buffers,
										   encoder_function=partial(self.agent.model.encode, task=None))
					agent_train_metrics = self.agent.update(self.replay_buffer, self.disc.get_reward)
				train_metrics.update(disc_train_metrics)
				train_metrics.update(agent_train_metrics)

			self._step += self.cfg.num_envs
	
		self.logger.finish(self.agent)