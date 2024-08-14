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
		self.batch_size = cfg.batch_size # Number of episodes per batch

		# Create empty stage buffers
		for i in range(cfg.n_stages + 1):
			_cfg = deepcopy(cfg)
			if i == cfg.n_stages:
				# Last buffer = Success buffer (oversampled)
				_cfg.batch_size = int(cfg.batch_size * cfg.oversample_ratio)
				_cfg.buffer_size = cfg.buffer_size // (cfg.n_stages + 1)
			else:
				_cfg.batch_size = int(cfg.batch_size * ((1 - cfg.oversample_ratio) / self.n_stages))
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
		tds = [self._stage_buffers[-1].sample().to(self._device)]
		idx, n_episodes = 0, tds[0].shape[1]

		# Loop buffers until we have all the samples
		while n_episodes < self.batch_size:
			stage_buffer = self._stage_buffers[idx]

			# Only sample if buffer has enough data
			if stage_buffer.cfg.batch_size <= stage_buffer.num_eps:
				td = stage_buffer.sample().to(self._device)
				tds.append(td)
				n_episodes += td.shape[1]
			# Skip success buffer
			idx = (idx + 1) % self.n_stages

		# TODO: 'task' needs to be cat at dim=0
		# Cut at batch size
		tds = torch.cat(tds, dim=1)[:, :self.batch_size]
		return self._prepare_batch(tds)

	def sample_for_disc(self, batch_size : int):
		"""Sample same batch size from each buffer and flatten T dimension. Returns list of obs"""
		obs = []
		for buffer in self._stage_buffers:
			# Only sample if buffer has enough data
			if batch_size <= buffer.n_elements:
				obs.append(buffer.sample_single(batch_size=batch_size)["obs"].to(self._device))
			else:
				obs.append(None)
		return obs


