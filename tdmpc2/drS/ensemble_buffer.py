import torch
from copy import deepcopy

from common.buffer import Buffer
from common.logger import timeit
from termcolor import colored
	
class EnsembleBuffer(Buffer):
	"""
	Ensemble of an offline dataloader and an online replay buffer.
	"""

	def __init__(self, cfg):
		_cfg1, _cfg2 = deepcopy(cfg), deepcopy(cfg)
		_cfg1.batch_size = int(cfg.batch_size * (1 - _cfg1.oversample_ratio))
		_cfg2.batch_size = int(cfg.batch_size - _cfg1.batch_size)
		super().__init__(_cfg1)

		# Load dataset into second replay buffer (ugly) TODO: This should be a normal dataloader
		from .data_utils import load_dataset_as_td
		demo_dataset = load_dataset_as_td(_cfg2.demo_path, num_traj=_cfg2.n_demos)
		_cfg2.buffer_size = len(demo_dataset) * len(demo_dataset[0]) # Offline buffer is not dynamically alocated
		# NOTE: Make sure demonstrations contain same type of rewards as online environment!
		self._offline_buffer = Buffer(_cfg2)
		for _td in demo_dataset:
			self._offline_buffer.add(_td.unsqueeze(1))
		print(colored(f"Filled demo buffer with {self._offline_buffer.num_eps} trajectories", "green"))

	def sample(self, return_td=False):
		"""Sample a batch of subsequences from the two buffers."""
		if return_td:
			raise NotImplementedError(f"TensorDict return not implemented for EnsembleBuffer")

		obs0, action0, reward0, task0 = self._offline_buffer.sample()
		obs1, action1, reward1, task1 = super().sample()
		return torch.cat([obs0, obs1], dim=1), \
			torch.cat([action0, action1], dim=1), \
			torch.cat([reward0, reward1], dim=1), \
			torch.cat([task0, task1], dim=0) if task0 and task1 else None
	
	# TODO: Need to revisit this to ensure some kind of diversity
	def sample_for_disc(self, batch_size : int):
		td0 = self._offline_buffer.sample_single(return_td=True)
		td1 = super().sample_single(return_td=True)
		
		# Concat + Shuffle
		tds = torch.cat([td0, td1], dim=0)

		obs_return = []
		for stage_index in range(self.cfg.n_stages):
			# Find indices where reward is equal to stage_index
			stage_indices = (tds["reward"] == stage_index).nonzero(as_tuple=True)[0]

			# Success indices
			success_indices = (tds["stage"][stage_indices] > stage_index).nonzero(as_tuple=True)[0]
			fail_indices = (tds["stage"][stage_indices] <= stage_index).nonzero(as_tuple=True)[0]

			# Cut at minimum to avoid data imbalance
			success_indices = success_indices[:min(len(success_indices), len(fail_indices))]
			fail_indices = fail_indices[:min(len(success_indices), len(fail_indices))]
			
			# Extract observations for those indices
			obs_return.append({"success_data": tds["obs"][success_indices], "fail_data": tds["obs"][fail_indices]})
		
		return obs_return