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

        for i in range(1, self.num_cubes):
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

            stacked_flag = xy_flag & z_flag

            result[f"stacked_flag_{i}"] = stacked_flag
            result[f"pair_success_{i}"] = stacked_flag & static_flag & (~grasp_flag)
            success = success & result[f"pair_success_{i}"]

        result["success"] = success.bool()
        return result


    def compute_2_cube_reward(self, cubeA_pos, cubeB_pos, cubeA_vel_linear, cubeA_vel_angular, is_cubeA_grasped, is_cubeA_on_cubeB):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # grasp and place reward
        goal_xyz = torch.hstack(
            [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

        reward[is_cubeA_grasped] = (4 + place_reward)[is_cubeA_grasped]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        is_cubeA_grasped = is_cubeA_grasped
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(cubeA_vel_linear, axis=1)
        av = torch.linalg.norm(cubeA_vel_angular, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[is_cubeA_on_cubeB] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[is_cubeA_on_cubeB]

        # Success
        reward[is_cubeA_on_cubeB & (~is_cubeA_grasped)] = 8

        return reward
    
    
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Count how many cubes are already stacked correctly
        stacked_count = torch.zeros((self.num_envs,), dtype=torch.int, device=self.device)
        stacked_mask = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)

        for i in range(1, self.num_cubes):
            pair_success = info[f"pair_success_{i}"]  # (B,)
            new_stacked = stacked_mask & pair_success
            stacked_count += new_stacked.long()
            stacked_mask = new_stacked

        stacked_reward = 8.0 * stacked_count.float()

        # Compute reward for the next non-stacked cube for each environment
        next_cube_idx = stacked_count + 1  # The next cube to stack is the one after the last stacked one
        next_reward = torch.zeros((self.num_envs,), device=self.device)

        # For the environments that have not stacked all cubes, compute the reward for the next cube
        if (next_cube_idx < self.num_cubes).any():
            cubesPos = torch.stack([cube.pose.p for cube in self.cubes], dim=0)  # (num_cubes, 3)
            cubesLinVel = torch.stack([cube.linear_velocity for cube in self.cubes], dim=0)
            cubesAngVel = torch.stack([cube.angular_velocity for cube in self.cubes], dim=0)  # (num_cubes, 3)

            next_cube_idx_mask = next_cube_idx < self.num_cubes

            cubeA_pos = cubesPos[next_cube_idx[next_cube_idx_mask], torch.arange(self.num_envs)]
            cubeB_pos = cubesPos[next_cube_idx[next_cube_idx_mask]-1, torch.arange(self.num_envs)]
            cubeA_lin_vel = cubesLinVel[next_cube_idx[next_cube_idx_mask], torch.arange(self.num_envs)]
            cubeA_ang_vel = cubesAngVel[next_cube_idx[next_cube_idx_mask], torch.arange(self.num_envs)]

            is_cubeA_grasped = torch.stack([info[f"grasped_{i}"][idx] for idx, i in enumerate(next_cube_idx[next_cube_idx_mask])])
            is_cubeA_on_cubeB = torch.stack([info[f"pair_success_{i}"][idx] for idx, i in enumerate(next_cube_idx[next_cube_idx_mask])])

            next_reward[next_cube_idx_mask] = self.compute_2_cube_reward(
                cubeA_pos, cubeB_pos, cubeA_lin_vel, cubeA_ang_vel, is_cubeA_grasped, is_cubeA_on_cubeB
            )

        return stacked_reward + next_reward
            


    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        max_reward = 8.0 * (self.num_cubes - 1)
        return self.compute_dense_reward(obs, action, info) / max_reward
