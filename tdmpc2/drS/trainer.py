from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from functools import partial

from .discriminator import Discriminator
from .disc_buffer import DiscriminatorBuffer


class DrsTrainer():
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
			from .data_utils import load_demo_dataset
			demo_dataset = load_demo_dataset(cfg.demo_path, keys=['next_observations'])
			self.stage_buffers[-1].add(next_obs=demo_dataset['next_observations'])

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
		ep_rewards, ep_max_rewards, ep_successes = [], [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, ep_max_reward, t = self.env.reset(), False, 0, -np.inf, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				ep_max_reward = max(ep_max_reward, reward)
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_max_rewards.append(ep_max_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_max_reward=np.nanmean(ep_max_rewards),
			episode_success=np.nanmean(ep_successes),
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
			reward = torch.tensor(float('nan'))
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		), batch_size=(1,))
		return td

	def train(self):
		"""Train agent and discriminator"""
		train_metrics, done, eval_next = {}, True, True
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
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_max_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).max(),
						episode_success=info['success'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.replay_buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]
				self._observations = np.empty((0,) + self.env.observation_space.shape)
				best_step = 0
				best_reward = -np.inf

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward))
			self._observations = np.vstack((self._observations, obs))

			# Get longest possible trajectory (DrS)
			if reward >= best_reward:
				best_reward = reward
				best_step = len(self._tds) - 1

			# Add experience to corresponding disc buffer (DrS)
			if done:
				if info["success"]:
					stage_idx = self.env.n_stages
				elif self.env.n_stages > 1:
					stage_idx = int(best_reward)
				else:
					stage_idx = 0
				self.stage_buffers[stage_idx].add(self._observations[:best_step])
			
			# Update discriminator and agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					disc_train_metrics = self.disc.update(self.stage_buffers,
										   encoder_function=partial(self.agent.model.encode, task=None))
					agent_train_metrics = self.agent.update(self.replay_buffer, self.disc.get_reward)
				train_metrics.update(disc_train_metrics)
				train_metrics.update(agent_train_metrics)

			self._step += 1
	
		self.logger.finish(self.agent)