import gym
import gym.envs
import numpy as np
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_stages = env.n_stages
        state_shape = np.prod(env.observation_space.shape)
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_shape, 32),
                nn.Sigmoid(),
                nn.Linear(32, 1),
            ) for _ in range(self.n_stages)
        ])
        self.trained = [False] * self.n_stages

    def set_trained(self, stage_idx):
        self.trained[stage_idx] = True

    def set_trained_all(self):
        [self.set_trained(idx) for idx in range(self.n_stages)]

    def forward(self, next_s, stage_idx):
        net = self.nets[stage_idx]
        return net(next_s)

    def get_reward(self, next_s, success):
        with torch.no_grad():
            bs = next_s.shape[0]
            if not torch.is_tensor(success):
                success = torch.tensor(success, device=next_s.device)
                success = success.reshape(bs, 1)
            if self.n_stages > 1:
                stage_idx = torch.cat([next_s[:, -(self.n_stages-1):], success], dim=1).sum(dim=1)
            else:
                stage_idx = success.squeeze(-1)
            
            stage_rewards = [
                torch.tanh(self(next_s, stage_idx=i)) if self.trained[i] else torch.zeros(bs, 1, device=next_s.device)
            for i in range(self.n_stages)]
            stage_rewards = torch.cat(stage_rewards + [torch.zeros(bs, 1, device=next_s.device)], dim=1)

            k = 3
            reward = k * stage_idx + stage_rewards[torch.arange(bs), stage_idx.long()]
            reward = reward / (k * self.n_stages) # reward is in (0, 1]
            #reward = reward - 2 # make the reward negative
            reward = reward + 1 # make the reward positive

            return reward
        
class DrsRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.envs, disc_ckpt):
        super().__init__(env)
        
        self.disc = Discriminator(env)
        checkpoint = torch.load(disc_ckpt)

        try:
            self.disc.load_state_dict(checkpoint['discriminator'])
        except:
            raise ValueError("DrS Error: Discriminator checkpoint does not match environment")

        self.disc.set_trained_all() # Checkpoint is trained

    def _try_f32_tensor(self, x):
        x = torch.from_numpy(x)
        if x.dtype == torch.float64:
            x = x.float()
        return x
    
    def _obs_to_tensor(self, obs):
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = self._try_f32_tensor(obs[k])
        else:
            obs = self._try_f32_tensor(obs)
        return obs

    def step(self, action):
        next_obs, _, done, info = self.env.step(action)
        reward = self.disc.get_reward(self._obs_to_tensor(next_obs).unsqueeze(0), info['success'])
        reward = reward.item()
        return next_obs, reward, done, info
