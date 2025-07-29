import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["LAZY_LEGACY_OP"] = "0"
import warnings

warnings.filterwarnings("ignore")

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm
from copy import deepcopy

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
from common.discriminator import Discriminator

import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import FormatStrFormatter

import torchrl

torch.backends.cudnn.benchmark = True

T_MAX = 50

def add_reward_text(frame, reward):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"R: {reward:.2f}"
    position = (10, 30)
    font_scale = 0.5
    color = (255, 255, 0)
    thickness = 1
    line_type = cv2.LINE_AA
    return cv2.putText(frame.copy(), text, position, font, font_scale, color, thickness, line_type)

def plot_reward_evolution(rewards):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(rewards, color='red')
    # ax.set_title("Reward Evolution")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.grid(True)

    # Set fixed axis limits
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(-0.5, 3.5)

    # Set tick formatters
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))  # No decimals
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # One decimal

    # Add spacing around plot
    plt.subplots_adjust(left=0.18, bottom=0.18, right=0.95, top=0.95)

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

class ObservationConverter:
    def __init__(self, cfg):
        self.obs_flag = cfg.obs != cfg.obs_save
        self.env_obs = None
        self.obs_type = cfg.obs_save
        if cfg.task.startswith("mw"):
            self.env_type = "metaworld"
        elif cfg.task.startswith("robosuite"):
            self.env_type = "robosuite"
        else:
            self.env_type = "maniskill"
            if self.obs_flag:
                cfg_obs = deepcopy(cfg)
                cfg_obs.obs = cfg_obs.obs_save
                self.env_obs = make_env(cfg_obs)

    def get_obs(self, env):
        if self.env_obs is not None:
            self.env_obs.set_state_dict(env.get_state_dict())
            return self.env_obs.get_obs()
        return env.get_obs(self.obs_type)

    def get_frame(self, env, obs, render_obs=True):
        if hasattr(obs, "keys") and render_obs:
            frame = None
            for k, v in obs.items():
                if k.startswith("rgb"):
                    frame_ = v[0].permute(1, 2, 0).cpu().numpy()
                    frame = frame_ if frame is None else np.concatenate((frame, frame_), axis=1)
            return frame
        return env.render()

    def reset(self, task_idx, seed, env):
        if self.env_obs is not None:
            return self.env_obs.reset(task_idx=task_idx, seed=seed)
        return env.get_obs(self.obs_type)

@hydra.main(config_name="eval", config_path="./config/")
def evaluate(cfg: dict):
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
    print(colored(f"Checkpoint: {cfg.checkpoint}", "blue", attrs=["bold"]))

    env = make_env(cfg)
    obs_converter = ObservationConverter(cfg)
    
    agent = TDMPC2(cfg)
    assert os.path.exists(cfg.checkpoint)
    agent.load(cfg.checkpoint)

    discriminator = Discriminator(env, cfg.discriminator, cfg.latent_dim)
    if cfg.disc_checkpoint:
        discriminator.load(cfg.disc_checkpoint)
        [discriminator.set_trained(i) for i in range(env.n_stages)]
    else:
        discriminator = None

    if cfg.save_video:
        video_dir = os.path.join(cfg.log_path, "videos")
        os.makedirs(video_dir, exist_ok=True)

    for ep in range(cfg.eval_episodes):
        seed = np.random.RandomState().randint(2**32)
        obs, done, ep_reward, t = env.reset(seed=seed), torch.tensor(False), 0, 0
        obs_save = obs_converter.reset(None, seed, env)
        rewards, video_frames = [], []

        while not done.all():
            action = agent.act(obs, t0=(t == 0), task=None, eval_mode=True).to(obs.device)
            obs, reward, done, info = env.step(action)
            reward = discriminator.get_reward(agent.model.encode(obs, None).squeeze(), reward[0].int()) if discriminator else reward
            obs_save = obs_converter.get_obs(env)
            ep_reward += reward
            t += 1
            rewards.append(reward.item())

            if t >= T_MAX:
                break

            if cfg.save_video:
                frame = obs_converter.get_frame(env, obs_save, cfg.render_obs)
                reward_plot = plot_reward_evolution(rewards)
                h = max(frame.shape[0], reward_plot.shape[0])
                frame = cv2.resize(frame, (frame.shape[1], h))
                reward_plot = cv2.resize(reward_plot, (reward_plot.shape[1], h))
                side_by_side = np.concatenate((frame, reward_plot), axis=1)
                video_frames.append(side_by_side)

        print(colored(f"Episode {ep+1} Reward: {ep_reward.item():.2f}", "yellow"))

        if cfg.save_video:
            out_path = os.path.join(video_dir, f"{cfg.task}-ep{ep+1}.mp4")
            imageio.mimsave(out_path, video_frames, fps=5)

    env.close()

if __name__ == "__main__":
    evaluate()