import torch
import numpy as np
from tensordict.tensordict import TensorDict

class DiscriminatorBuffer(object):
    # can be optimized by create a buffer of size (n_traj, len_traj, dim)
    def __init__(self, buffer_size, obs_space, action_space, device):
        self.buffer_size = buffer_size
        if hasattr(obs_space, "spaces"):
            self.next_observations = TensorDict({k : torch.zeros((self.buffer_size,) + v.shape) 
                                                 for k, v in obs_space.spaces.items()}, batch_size=(self.buffer_size,)).float()
        else:
            self.next_observations = torch.zeros((self.buffer_size,) + obs_space.shape).float()
        self.device = device
        self.pos = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.buffer_size if self.full else self.pos

    def add(self, next_obs):
        '''
            Adds an element into the existing circular buffer
            next_obs: torch.Tensor or tensordict.TensorDict
        '''
        l = next_obs.shape[0]
        
        while self.pos + l >= self.buffer_size:
            self.full = True
            k = self.buffer_size - self.pos
            self.next_observations[self.pos:] = next_obs[:k]
            self.pos = 0
            next_obs = next_obs[k:]
            l = next_obs.shape[0]
            
        self.next_observations[self.pos:self.pos+l] = next_obs.copy() if isinstance(next_obs, np.ndarray) else next_obs.clone()
        self.pos = (self.pos + l) % self.buffer_size
        self.next_observations = self.next_observations.float()

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            next_observations=self.next_observations[idxs],
        )
        return {k: v.to(self.device) for k,v in batch.items()}