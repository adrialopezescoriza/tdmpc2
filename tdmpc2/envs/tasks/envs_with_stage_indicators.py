from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
import numpy as np
import torch

class DrS_BaseEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ("dense", "sparse", "semi_sparse", "drS")

    def compute_stage_indicator(self):
        raise NotImplementedError()
    
    def get_reward(self, **kwargs):
        if self._reward_mode == "sparse":
            eval_info = self.evaluate(**kwargs)
            return float(eval_info["success"])
        elif self._reward_mode == "dense":
            return self.compute_dense_reward(**kwargs)
        elif self._reward_mode == "semi_sparse" or self._reward_mode == "drS":
            # reward build from stage indicators
            return self.compute_semi_sparse_reward(**kwargs)
        else:
            raise NotImplementedError(self._reward_mode)
        
    def compute_semi_sparse_reward(self, **kwargs):
        stage_indicators = self.compute_stage_indicator()
        eval_info = self.evaluate()
        return sum(stage_indicators.values()) + eval_info["success"].float()
    
    @property
    def _default_sensor_configs(self):
        # Define all the cameras needed for the environment
        pose_ext = sapien_utils.look_at(eye=[0.6, 0.7, 0.6], target=[0.0, 0.0, 0.35]) # NOTE: Same as render camera
        pose_base = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose_base, width=128, height=128, fov=np.pi / 2, near=0.01, far=100),
            CameraConfig("ext_camera", pose=pose_ext, width=128, height=128, fov=1, near=0.01, far=100),
        ]

############################################
# Pick And Place
############################################

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv

@register_env("PickAndPlace_DrS_learn", max_episode_steps=100)
class PickAndPlace_DrS_learn(DrS_BaseEnv, PickCubeEnv):
    def __init__(self, *args, **kwargs):
        self.n_stages = 3
        super().__init__(*args, **kwargs)

    def check_obj_placed(self):
        obj_to_goal_pos = self.goal_pos - self.obj_pose.p
        return np.linalg.norm(obj_to_goal_pos) <= self.goal_thresh

    def compute_stage_indicator(self):
        eval_info = self.evaluate()
        return {
            'is_grasped': (eval_info['is_grasped']).float(),
            'is_obj_placed': (eval_info['is_obs_placed']).float(),
        }

############################################
# Stack Cube
############################################

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv

@register_env("StackCube_DrS_learn", max_episode_steps=100)
class StackCube_DrS_learn(DrS_BaseEnv, StackCubeEnv):
    def __init__(self, *args, **kwargs):
        self.n_stages = 3
        super().__init__(*args, **kwargs)

    def compute_stage_indicator(self):
        eval_info = self.evaluate()
        return {
            'is_grasped': (torch.logical_or(eval_info["is_cubeA_grasped"], eval_info["success"])).float(), # allow releasing the cube when stacked
            'is_cube_A_placed': (torch.logical_or(eval_info["is_cubeA_on_cubeB"], eval_info["success"])).float(),
        }

############################################
# Peg Insertion
############################################

from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv

@register_env("PegInsertionSide_DrS_learn", max_episode_steps=100)
class PegInsertionSide_DrS_learn(DrS_BaseEnv, PegInsertionSideEnv):
    def __init__(self, *args, **kwargs):
        self.n_stages = 3
        super().__init__(*args, **kwargs)

    def is_peg_pre_inserted(self):
        peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
        peg_head_wrt_goal_yz_dist = torch.linalg.norm(
            peg_head_wrt_goal.p[:, 1:], axis=1
        )
        peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
        peg_wrt_goal_yz_dist = torch.linalg.norm(peg_wrt_goal.p[:, 1:], axis=1)

        # stage 3 passes if peg is correctly oriented in order to insert into hole easily
        pre_inserted = (peg_head_wrt_goal_yz_dist < 0.01) & (
            peg_wrt_goal_yz_dist < 0.01
        )
        return pre_inserted

    def compute_stage_indicator(self):
        success = self.evaluate()["success"]
        return {
            'is_correctly_grasped': (torch.logical_or(self.agent.is_grasping(self.peg, max_angle=20), success)).float(), # do this to allow releasing the peg when inserted
            'is_peg_pre_inserted': (torch.logical_or(self.is_peg_pre_inserted(), success)).float(),
        }

    @property
    def _default_sensor_configs(self):
        # Define all the cameras needed for the environment
        pose_ext = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4]) # NOTE: Same as render camera
        pose_base = sapien_utils.look_at([0, -0.3, 0.2], [0, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose_base, width=128, height=128, fov=np.pi / 2, near=0.01, far=100),
            CameraConfig("ext_camera", pose=pose_ext, width=128, height=128, fov=1, near=0.01, far=100),
        ]