from math import ceil
from common.logger import timeit

import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class StageBuffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda')
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0
		self._max_length = 0
		self._storage_device = None

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity
	
	@property
	def n_elements(self):
		if hasattr(self, "_buffer"):
			return len(self._buffer)
		return 0
	
	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps
	
	@property
	def max_length(self):
		"""Return the maximum length of episodes in the buffer."""
		return self._max_length
	
	@property
	def batch_size(self):
		return self._batch_size

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=True,
			prefetch=int(self.cfg.num_envs / self.cfg.steps_per_update),
			batch_size=self._batch_size,
		)
	
	def set_storage_device(self, device):
		self._storage_device = device

	def _init(self):
		"""Initialize the replay buffer."""
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=torch.device(self._storage_device))
		)

	def _to_device(self, *args, device=None):
		if device is None:
			device = self._device
		return (arg.to(device, non_blocking=True) \
			if arg is not None else None for arg in args)

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		obs = td['obs']
		action = td['action'][1:]
		reward = td['reward'][1:].unsqueeze(-1)
		task = td['task'][0] if 'task' in td.keys() else None
		return self._to_device(obs, action, reward, task)

	def add(self, td):
		"""
		Add a multi-env episode to the buffer.
		Expects `tds` to be a TensorDict with batch size TxB
		"""
		b_size = td.shape[1]
		td['episode'] = torch.ones_like(td['reward'], dtype=torch.int64) * torch.arange(self._num_eps, self._num_eps+b_size)
		td = td.permute(1, 0)
		if self._num_eps == 0:
			self._buffer = self._init()
		for i in range(b_size):
			self._buffer.extend(td[i])
			if td[i].shape[0] > self._max_length:
				self._max_length =  td[i].shape[0]
		self._num_eps += b_size
		return self._num_eps

	def sample(self, return_td=True):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		return td if return_td else self._prepare_batch(td)
	
	def sample_single(self, batch_size, return_td=True):
		"""Sample a single batch with no slicing.
		WARNING: action[0] -> obs[0] -> action[1] -> obs[1]"""
		td = torch.cat([self._buffer.sample(self._batch_size) for _ in range(ceil(batch_size / self._batch_size))], dim=0)
		return td if return_td else self._prepare_batch(td)