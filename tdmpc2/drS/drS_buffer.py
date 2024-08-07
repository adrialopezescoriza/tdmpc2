import torch
from copy import deepcopy
from tensordict.tensordict import TensorDict

from .stage_buffer import StageBuffer
from termcolor import colored

from .data_utils import load_dataset_as_td

def nanmax(tensor, dim=None, keepdim=False):
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    return output
	
class DrSBuffer():
	"""
	Ensemble of an offline dataloader and an online replay buffer.
	"""

	def __init__(self, cfg):
		self._stage_buffers = []
		self._device = torch.device('cuda')
		self.n_stages = cfg.n_stages

		self.horizon = cfg.horizon
		self.batch_size = cfg.batch_size

		# Create empty stage buffers
		for i in range(cfg.n_stages + 1):
			_cfg = deepcopy(cfg)
			if i == cfg.n_stages:
				# Last buffer = Success buffer (oversampled)
				_cfg.batch_size = int(cfg.batch_size * cfg.oversample_ratio) * (cfg.horizon + 1)
				_cfg.buffer_size = cfg.buffer_size // (cfg.n_stages + 1)
			else:
				_cfg.batch_size = int(cfg.batch_size * ((1 - cfg.oversample_ratio) / self.n_stages)) * (cfg.horizon + 1)
				_cfg.buffer_size = cfg.buffer_size // (cfg.n_stages + 1)
			self._stage_buffers.append(StageBuffer(_cfg))

		self._capacity = sum([buffer.cfg.buffer_size for buffer in self._stage_buffers])
		self._total_bytes = None

		# Fill last stage buffer with demos
		demo_dataset = load_dataset_as_td(cfg.demo_path)
		# WARNING: Make sure demonstrations contain same type of rewards as online environment!
		self._stage_buffers[-1].add(torch.stack(demo_dataset, dim=1))
		print(colored(f"Filled demo buffer with {self._stage_buffers[-1].num_eps} trajectories", "green"))

	@property
	def stage_buffers(self):
		return self._stage_buffers

	@property
	def num_eps(self):
		return sum([buffer.num_eps for buffer in self._stage_buffers])
	
	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity
	
	def _init(self, tds):
		"""Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		self._total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {self._total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = 'cuda' if 2.5*self._total_bytes < mem_free else 'cpu'
		print(f'Using {storage_device.upper()} memory for storage.')

	
	def add(self, tds):
		"""
		Add batch of trajectories to replay buffers
		args:
			tds: TensorDict with shape (TxB)
		return:
			stage_idx, best_steps
		"""
		# We take longest non-decreasing trajectory
		stage_indices, reversed_indices = nanmax(torch.flip(tds['reward'], dims=(0,)), dim=0)
		best_steps = len(tds) - reversed_indices

		if self._total_bytes is None:
				self._init(tds[:,0])

		for i, (step, stage_idx) in enumerate(zip(best_steps, stage_indices.int())):
			# Cut trajectory at the best step (last max reward)
			tds_ = tds[:step,i:i+1]
			self._stage_buffers[stage_idx].add(tds_)
		return self.num_eps
	
	def sample(self):
		"""Sample a batch of subsequences from all buffers. Oversample success data."""
		
		# Sample from success buffer
		obs, action, reward, task = self._stage_buffers[-1].sample()
		idx = 0

		# Loop buffers until we have all the samples
		while reward.shape[1] < self.batch_size:
			stage_buffer = self._stage_buffers[idx]

			# Only sample if buffer has enough data
			if stage_buffer.cfg.batch_size <= stage_buffer.n_elements:
				obs_, action_, reward_, task_ = stage_buffer.sample()
				obs, action, reward, task = torch.cat([obs, obs_], dim=1), \
											torch.cat([action, action_], dim=1), \
											torch.cat([reward, reward_], dim=1), \
											torch.cat([task, task_], dim=0) if task and task_ else None
			# Skip success buffer
			idx = (idx + 1) % self.n_stages

		# Return only batch
		return obs[:, :self.batch_size], action[:, :self.batch_size], reward[:, :self.batch_size], task[:, :self.batch_size] if task else None

	def sample_for_disc(self, stage_indices, batch_size : int):
		"""Sample same batch size from each buffer and flatten T dimension"""
		# Split batch size along number of buffers (uniform buffer sampling)
		bs_per_buffer = batch_size // len(stage_indices)
		n_elements, counter, tds = 0, 0, []

		# Loop buffers until we have all the samples
		while n_elements < batch_size:
			stage_idx = stage_indices[counter]
			stage_buffer = self._stage_buffers[stage_idx]

			# Take minimum of batch sizes
			bs = min(bs_per_buffer, stage_buffer.cfg.batch_size)

			# Only sample if buffer has enough data
			if bs_per_buffer <= stage_buffer.n_elements:
				td_ = stage_buffer.sample_single(batch_size=bs)
				tds.append(td_)
				n_elements += len(td_) # Is the same as batch_size

			if counter == len(stage_indices) and n_elements == 0:
				# Not enough data in buffers
				return None

			counter = (counter + 1) % len(stage_indices)

		# Cut to fit batch_size
		tds = torch.cat(tds, dim=0)[:batch_size]
		return tds


