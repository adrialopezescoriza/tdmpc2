import torch
import numpy as np

class DiscriminatorBuffer(object):
    # can be optimized by create a buffer of size (n_traj, len_traj, dim)
    def __init__(self, buffer_size, obs_space, action_space, device):
        self.buffer_size = buffer_size
        self.next_observations = np.zeros((self.buffer_size,) + obs_space.shape, dtype=obs_space.dtype)
        self.device = device
        self.pos = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.buffer_size if self.full else self.pos

    def add(self, next_obs):
        l = next_obs.shape[0]
        
        while self.pos + l >= self.buffer_size:
            self.full = True
            k = self.buffer_size - self.pos
            self.next_observations[self.pos:] = next_obs[:k]
            self.pos = 0
            next_obs = next_obs[k:]
            l = next_obs.shape[0]
            
        self.next_observations[self.pos:self.pos+l] = next_obs.copy()
        self.pos = (self.pos + l) % self.buffer_size
        self.next_observations = np.float32(self.next_observations)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            next_observations=self.next_observations[idxs],
        )
        return {k: torch.tensor(v).to(self.device) for k,v in batch.items()}