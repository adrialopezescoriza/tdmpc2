import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .data_utils import sample_from_multi_buffers


class Discriminator(nn.Module):
    def __init__(self, envs, cfg):
        super().__init__()
        self.n_stages = envs.n_stages
        state_shape = np.prod(envs.observation_space.shape)
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_shape, 32),
                nn.Sigmoid(),
                nn.Linear(32, 1),
            ) for _ in range(self.n_stages)
        ])
        self.trained = [False] * self.n_stages
        self._cfg = cfg
        self.device = torch.device('cuda')
        self.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=cfg.disc_lr)

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def set_trained(self, stage_idx):
        self.trained[stage_idx] = True

    def forward(self, next_s, stage_idx):
        net = self.nets[stage_idx]
        return net(next_s)

    def update(self, stage_buffers):
        for stage_idx in range(self.n_stages):
            success_data = sample_from_multi_buffers(stage_buffers[stage_idx+1:], self._cfg.batch_size)
            if not success_data:
                break
            fail_data = sample_from_multi_buffers(stage_buffers[:stage_idx+1], self._cfg.batch_size)

            disc_next_obs = torch.cat([fail_data['next_observations'], success_data['next_observations']], dim=0)
            disc_labels = torch.cat([
                torch.zeros((self._cfg.batch_size, 1), device=self.device), # fail label is 0
                torch.ones((self._cfg.batch_size, 1), device=self.device), # success label is 1
            ], dim=0)

            logits = self(disc_next_obs, stage_idx)
            disc_loss = F.binary_cross_entropy_with_logits(logits, disc_labels)
            
            self.optimizer.zero_grad()
            disc_loss.backward()
            self.optimizer.step()

            pred = logits.detach() > 0

            self.set_trained(stage_idx)

            return {
                "discriminator_loss": float(disc_loss.mean().item()),
            }

    def get_reward(self, next_s, stage_idx):
        '''
            next_s : torch.Tensor (length, batch_size, state_dim)
            stage_idx : torch.Tensor (length, batch_size, 1)
        '''
        with torch.no_grad():
            bs = stage_idx.shape[:-1]
            
            stage_rewards = [
                torch.tanh(self(next_s, stage_idx=i)) if self.trained[i] else torch.zeros(*bs + (1,), device=next_s.device)
            for i in range(self.n_stages)]
            stage_rewards = torch.cat(stage_rewards + [torch.zeros(*bs + (1,), device=next_s.device)], dim=-1) # Dummy stage_reward for success state (always 0)

            k = 3
            reward = k * stage_idx + torch.gather(stage_rewards, -1, stage_idx.long()) # Selects stage index for each reward
            reward = reward / (k * self.n_stages) # reward is in (0, 1]
            #reward = reward - 2 # make the reward negative
            reward = reward + 1 # make the reward positive

            return reward