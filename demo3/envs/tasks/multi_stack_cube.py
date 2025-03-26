from typing import Any, Dict, Union
import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("StackNCubes-v1", max_episode_steps=100)
class StackNCubesEnv(BaseEnv):
    """
    Task: Stack N cubes in order (bottom to top).
    - Each cube must be placed correctly on the previous one.
    - Cubes are randomized in position and rotation.
    - Success: all cubes stacked in order, static, and ungrasped.
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, num_cubes=3, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.num_cubes = num_cubes
        self.max_episode_steps = 50 * num_cubes  # Adjust max steps based on number of cubes
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Array of possible colors for the cubes. Clors are 0: RED, 1: GREEN, 2: BLUE, 3: YELLOW, 4: CYAN, 5: MAGENTA
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]

        self.cubes = []
        for i in range(self.num_cubes):
            color = colors[i % len(colors)] + [1] # Transparency
            cube = actors.build_cube(
                self.scene,
                half_size=0.02,
                color=color,
                name=f"cube_{i}",
                initial_pose=sapien.Pose(p=[0, 0, 0.1 + 0.05 * i]),
            )
            self.cubes.append(cube)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            region = [[-0.2, -0.2], [0.2, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.01

            for i, cube in enumerate(self.cubes):
                xy = sampler.sample(radius, 100)
                xyz = torch.zeros((b, 3), device=self.device)
                xyz[:, :2] = xy
                xyz[:, 2] = 0.02
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
                cube.set_pose(Pose.create_from_pq(xyz, qs))

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            for i, cube in enumerate(self.cubes):
                obs[f"cube_{i}_pose"] = cube.pose.raw_pose
                obs[f"tcp_to_cube_{i}_pos"] = cube.pose.p - self.agent.tcp.pose.p
        return obs

    def evaluate(self):
        B = self.num_envs
        success = torch.ones((B,), dtype=torch.bool, device=self.device)
        result = {}

        # Check if each cube is grasped
        for i, cube in enumerate(self.cubes):
            grasped_flag = self.agent.is_grasping(cube)  # (B,)
            result[f"grasped_{i}"] = grasped_flag

        # Ordered stacking constraints
        stacked_flags = [None] * self.num_cubes
        for i in range(self.num_cubes):
            if i == 0:
                # Assume the base is always valid (table)
                stacked_flags[i] = torch.ones((B,), dtype=torch.bool, device=self.device)
                continue

            below = self.cubes[i - 1]
            above = self.cubes[i]
            offset = above.pose.p - below.pose.p  # (B, 3)

            xy_flag = (
                torch.linalg.norm(offset[..., :2], axis=1)
                <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
            )
            z_flag = torch.abs(offset[..., 2] - self.cube_half_size[2] * 2) <= 0.01
            static_flag = above.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            grasp_flag = result[f"grasped_{i}"]  # (B,)

            pair_success = (
                stacked_flags[i - 1]
                & xy_flag
                & z_flag
                & static_flag
                & (~grasp_flag)
            )

            stacked_flags[i] = pair_success
            success &= pair_success

            result[f"xy_flag_{i}"] = xy_flag
            result[f"pair_success_{i}"] = pair_success

        result["success"] = success.bool()
        return result


    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        B = action.shape[0]
        device = self.device
        reward = torch.zeros((B,), device=device)
        tcp_pose = self.agent.tcp.pose.p  # (B, 3)

        # Count how many cubes are already stacked correctly
        stacked_count = torch.zeros((B,), dtype=torch.long, device=device)
        stacked_mask = torch.ones((B,), dtype=torch.bool, device=device)

        for i in range(1, self.num_cubes):
            pair_key = f"pair_success_{i}"
            pair_success = info[pair_key]  # (B,)
            new_stacked = stacked_mask & pair_success
            stacked_count += new_stacked.long()
            stacked_mask = new_stacked

        reward += 10.0 * stacked_count.float()

        # If all cubes stacked, return
        fully_stacked = stacked_count == (self.num_cubes - 1)
        if fully_stacked.all():
            return reward

        # Process next cube to stack (per environment)
        next_cube_idx = stacked_count + 1  # (B,)
        target_pose_all = torch.stack([cube.pose.p for cube in self.cubes])  # (num_cubes, B, 3)
        below_pose_all = torch.stack([self.cubes[i - 1].pose.p for i in range(1, self.num_cubes)])  # (num_cubes-1, B, 3)

        target_pose = torch.stack([target_pose_all[i, b] for b, i in enumerate(next_cube_idx)])  # (B, 3)
        below_pose = torch.stack([below_pose_all[i - 1, b] for b, i in enumerate(next_cube_idx)])  # (B, 3)
        goal_pos = below_pose + torch.tensor([0, 0, self.cube_half_size[2] * 2], device=device)

        # Reach reward
        dist_to_tcp = torch.linalg.norm(tcp_pose - target_pose, dim=1)
        reach_reward = 1.0 * (1 - torch.tanh(5 * dist_to_tcp))

        # Grasp reward
        grasp_all = torch.stack([info[f"grasped_{i}"] for i in range(self.num_cubes)])  # (num_cubes, B)
        is_grasped = torch.stack([grasp_all[i, b] for b, i in enumerate(next_cube_idx)])  # (B,)
        grasp_reward = torch.where(is_grasped, torch.full_like(reward, 1.5), torch.zeros_like(reward))

        # Place reward
        dist_to_goal = torch.linalg.norm(target_pose - goal_pos, dim=1)
        place_reward = 2.0 * (1 - torch.tanh(5 * dist_to_goal))
        placed = dist_to_goal < 0.02  # (B,)

        # Static and ungrasped reward
        is_static_all = torch.stack(
            [self.cubes[i].is_static(lin_thresh=1e-2, ang_thresh=0.5) for i in range(self.num_cubes)]
        )  # (num_cubes, B)
        is_static = torch.stack([is_static_all[i, b] for b, i in enumerate(next_cube_idx)])  # (B,)
        ungrasped = ~is_grasped
        final_reward = torch.where(
            placed & ungrasped & is_static,
            torch.full_like(reward, 2.5),
            torch.zeros_like(reward),
        )

        reward += reach_reward + grasp_reward + place_reward + final_reward
        return reward


    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        max_reward = 10.0 * (self.num_cubes - 1)
        return self.compute_dense_reward(obs, action, info) / max_reward
