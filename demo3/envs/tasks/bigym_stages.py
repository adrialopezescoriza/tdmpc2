from bigym.envs.pick_and_place import StoreBox, PickBox
from bigym.envs.reach_target import ReachTarget
from bigym.envs.manipulation import StackBlocks, FlipCup
from bigym.envs.cupboards import DrawerTopOpen, WallCupboardOpen
from bigym.envs.move_plates import MovePlate
from bigym.envs.dishwasher import DishwasherCloseTrays
from bigym.bigym_env import BiGymEnv, CONTROL_FREQUENCY_MAX, CONTROL_FREQUENCY_MIN
# CONTROL_FREQUENCY_MAX (500Hz) = 40 * CONTROL_FREQUENCY_MIN(20Hz)

from bigym.action_modes import TorqueActionMode, JointPositionActionMode, PelvisDof
from bigym.utils.observation_config import ObservationConfig, CameraConfig

from transforms3d.quaternions import mat2quat

from collections import OrderedDict
import numpy as np
from gymnasium.spaces import Box
import copy

def look_at(eye, target, up=(0, 0, 1)):

    def normalize_vector(x, eps=1e-6):
        x = np.asarray(x)
        assert x.ndim == 1, x.ndim
        norm = np.linalg.norm(x)
        if norm < eps:
            return np.zeros_like(x)
        else:
            return x / norm

    forward = normalize_vector(np.array(target) - np.array(eye))
    up = normalize_vector(up)
    left = np.cross(up, forward)
    up = np.cross(forward, left)
    rotation = np.stack([forward, left, up], axis=1)
    return mat2quat(rotation).tolist()

class BiGymStages(BiGymEnv):
    def __init__(self, obs_mode, img_size, ext_camera = {"pos": None, "quat" : None}, *args, **kwargs):
        if obs_mode.startswith("rgb"):
            observation_config=ObservationConfig(
                cameras=[
                    CameraConfig(
                        name="head",
                        rgb=True,
                        depth=False,
                        resolution=(img_size, img_size),
                    ),
                    CameraConfig(
                        name="external",
                        rgb=True,
                        depth=False,
                        resolution=(img_size, img_size),
                        pos=ext_camera["pos"],
                        quat = ext_camera["quat"]
                    ),
                ],
            )
        elif obs_mode == "state":
            observation_config=ObservationConfig(
                cameras=[],
                privileged_information=True,
            )
        else:
            raise NotImplementedError
        
        super().__init__(
            observation_config=observation_config,
            *args, **kwargs
        )
    
    def _reward(self) -> float:
        """Get current episode reward."""
        stage_indicators = self.compute_stage_indicators()
        return len(stage_indicators)+1 if self.success else sum(stage_indicators.values())
    
    def _fail(self) -> bool:
        """Check if the episode is failed (Never)"""
        return False
    
    def compute_stage_indicators(self):
        raise NotImplementedError

    def reset(self, **kwargs):
        res = super().reset(**kwargs)
        if hasattr(self, "action_space_"):
            self.action_space = self.action_space_
        return res

########################################
########   Wall Cupboard Open   ########
########################################
class WallCupboardOpenStages(BiGymStages, WallCupboardOpen):
    def __init__(self, obs_mode, img_size, *args, **kwargs):
        self.n_stages = 3
        self.reward_mode = "semi_sparse"
        self.max_episode_steps = 150
        action_mode=JointPositionActionMode(floating_base=True, floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ], absolute=True)

        super().__init__(
            obs_mode=obs_mode,
            img_size=img_size,
            action_mode=action_mode,
            control_frequency=CONTROL_FREQUENCY_MIN,
            ext_camera={"pos":[0,-2,3], "quat":None},
            *args, 
            **kwargs,
        )

    def compute_stage_indicators(self):
        is_cupboard_grasped_one_gripper = np.any([self.robot.is_gripper_holding_object(self.cabinet_wall, side) for side in self.robot.grippers])
        is_cupboard_grasped_two_gripper = np.all([self.robot.is_gripper_holding_object(self.cabinet_wall, side) for side in self.robot.grippers])
        return {
            "stage1": is_cupboard_grasped_one_gripper or self.success,
            "stage2": is_cupboard_grasped_two_gripper or self.success,
        }


########################################
######## DishWasher Close Trays ########
########################################
class DishwasherCloseTraysStages(BiGymStages, DishwasherCloseTrays):
    def __init__(self, obs_mode, img_size, *args, **kwargs):
        self.n_stages = 2
        self.reward_mode = "semi_sparse"
        self.max_episode_steps = 100
        action_mode=JointPositionActionMode(floating_base=True, floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ])

        super().__init__(
            obs_mode=obs_mode,
            img_size=img_size,
            action_mode=action_mode,
            control_frequency=CONTROL_FREQUENCY_MIN * 6,
            ext_camera={"pos":[0,-2,3], "quat":None},
            *args, 
            **kwargs,
        )

    def compute_stage_indicators(self):
        return {
            "stage1": np.isclose(self.dishwasher.get_state()[1:], 0, atol=self._TOLERANCE).any()
        }


########################################
########    Drawer Top Open     ########
########################################
class DrawerTopOpenStages(BiGymStages, DrawerTopOpen):
    def __init__(self, obs_mode, img_size, *args, **kwargs):
        self.n_stages = 2
        self.reward_mode = "semi_sparse"
        self.max_episode_steps = 150
        action_mode=JointPositionActionMode(floating_base=True, floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ], absolute=True)

        # Modify external camera orientation
        quat = [ 0.3649717, -0.2778159, -0.1150751, 0.8811196 ]
        quat = [quat[3]] + quat[:3]

        super().__init__(
            obs_mode=obs_mode,
            img_size=img_size,
            action_mode=action_mode,
            control_frequency=CONTROL_FREQUENCY_MIN,
            ext_camera={"pos":[-1,-1.5,2], "quat":quat},
            *args, 
            **kwargs,
        )

        # Resize action space to fit demos
        low_ = np.array([-0.05256215, -0.01346442, -0.02014803, -0.02447689, -0.48408347,
                        -0.16367014, -0.33427486, -0.83284396,  0.        , -1.4986415 ,
                        -0.2914733 , -0.1270883 , -0.6966172 , -1.3383255 ,  0.        ,
                            0.        ])
        
        high_ = np.array([0.02163532, 0.02605202, 0.00349079, 0.01902353, 0.18264526,
                        0.01864652, 0.10289277, 0.04801869, 0.4592447 , 0.14056844,
                        0.27309757, 0.47346738, 1.4556292 , 0.24234204, 0.        ,
                        1.        ])
        
        tolerance = 1e-5
        
        self.action_space_ = Box(low=low_ - tolerance, high=high_ + tolerance)
        self.action_space = copy.deepcopy(self.action_space_)

    def compute_stage_indicators(self):
        is_drawer_grasped = np.any([self.robot.is_gripper_holding_object(self.cabinet_drawers, side) for side in self.robot.grippers])
        is_pulling_drawer = self.cabinet_drawers.get_state()[-1] > 1e-1 and is_drawer_grasped
        return {
            "stage1": is_pulling_drawer or self.success,
        }

########################################
#######       Move Plate        ########
########################################
class MovePlateStages(BiGymStages, MovePlate):
    def __init__(self, obs_mode, img_size, *args, **kwargs):
        self.n_stages = 2
        self.reward_mode = "semi_sparse"
        self.max_episode_steps = 150
        action_mode=JointPositionActionMode(floating_base=True, floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.RZ])

        super().__init__(
            obs_mode=obs_mode,
            img_size=img_size,
            action_mode=action_mode,
            control_frequency=CONTROL_FREQUENCY_MIN,
            ext_camera={"pos":[0.5,-1,2], "quat":None},
            *args, 
            **kwargs,
        )

        low_ = np.array([-0.02, -0.08, -0.05,
                         -0.18, -0.1, -0.1,
                         -0.3, -0.12, -0.1,
                         -0.07, -0.1, -0.1,
                         -0.058934748557960805,  0.        ,  0.        ])
        
        high_ = np.array([0.015, 0.04, 0.1,
                          0.1, 0.054908952743412336, 0.07382039380207843,
                          0.17238096489878027, 0.12, 0.07,
                          0.07, 0.1, 0.1,
                          0.057050697620818144, 1.        , 0.        ])

        # low_ = np.array([-0.01962075, -0.07761819, -0.06895841, 
        #                   -0.26554358, -0.18311483, -0.26119497, 
        #                   -0.29295182, -0.49999997, -0.2771536 ,
        #                   -0.14540102,, -0.14482735, -0.2845109 , 
        #                   -0.5536907 ,  0.        ,  0.        ])
        
        # high_ = np.array([0.0175525 , 0.03667733, 0.14617348,
        #                   0.09228504, 0.08773132, 0.12075103, 
        #                   0.26808703, 0.631871  , 0.103172, 
        #                   0.10693253, 0.12817591, 0.23806447, 
        #                   0.42442662, 1.        , 0.        ])

        tolerance = 1e-5
        
        self.action_space_ = Box(low=low_ - tolerance, high=high_ + tolerance)
        self.action_space = copy.deepcopy(self.action_space_)

    def compute_stage_indicators(self):
        plate = self.plates[0]
        plate_pose = plate.get_pose()

        is_plate_grasped = np.array([self.robot.is_gripper_holding_object(plate, side) for side in self.robot.grippers]).any()
        is_plate_lifted = not (plate.is_colliding(self.rack_start) and plate_pose[2] > self.rack_start.get_pose()[2])

        is_plate_above_target = np.allclose(plate_pose[:2], self.rack_target.get_pose()[:2], atol=0.2)
        is_plate_on_target = plate.is_colliding(self.rack_target)

        return {
            "stage1": (is_plate_grasped and is_plate_lifted) or is_plate_above_target,
            # "stage2": ((is_plate_lifted or is_plate_on_target) and is_plate_grasped) or is_plate_above_target,
            # "stage3": is_plate_above_target,
        }

########################################
#######         Flip Cup        ########
########################################
from pyquaternion import Quaternion
class FlipCupStages(BiGymStages, FlipCup):
    def __init__(self, obs_mode, img_size, *args, **kwargs):
        self.n_stages = 4
        self.reward_mode = "semi_sparse"
        self.max_episode_steps = 200
        action_mode=JointPositionActionMode(floating_base=True, floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ], absolute=True)

        super().__init__(
            obs_mode=obs_mode,
            img_size=img_size,
            action_mode=action_mode,
            control_frequency=CONTROL_FREQUENCY_MIN,
            ext_camera={"pos":[0,-2,3], "quat":None},
            *args, 
            **kwargs,
        )

    def compute_stage_indicators(self):
        is_cup_lifted = not self.cup.is_colliding(self.cabinet.counter)
        is_cup_grasped = np.any([self.robot.is_gripper_holding_object(self.cup, side) for side in self.robot.grippers])
        is_cup_above_counter = self.cup.get_pose()[2] > self.cabinet.counter.get_position()[2]

        up = np.array([0, 0, 1])
        cup_up = Quaternion(self.cup.body.get_quaternion()).rotate(up)
        angle_to_up = np.arccos(np.clip(np.dot(cup_up, up), -1.0, 1.0))
        is_cup_flipping = (angle_to_up < np.deg2rad(90)) and is_cup_grasped

        is_cup_flipped = angle_to_up < np.deg2rad(30)
        is_cup_flipped_on_counter = is_cup_flipped and self.cup.is_colliding(self.cabinet.counter)

        return {
            "stage1": (is_cup_lifted and is_cup_grasped and is_cup_above_counter) or is_cup_flipping or is_cup_flipped,
            "stage2": ((is_cup_flipping or is_cup_flipped) and is_cup_grasped),
            "stage3": is_cup_flipped_on_counter,
        }

########################################
####### Reach Target (no demos) ########
########################################
class ReachTargetStages(BiGymStages, ReachTarget):
    def __init__(self, obs_mode, img_size, *args, **kwargs):
        self.n_stages = 1
        self.reward_mode = "semi_sparse"
        self.max_episode_steps = 50
        action_mode=JointPositionActionMode(floating_base=True, floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.RZ])

        super().__init__(
            obs_mode=obs_mode,
            img_size=img_size,
            action_mode=action_mode,
            control_frequency=CONTROL_FREQUENCY_MIN,
            ext_camera={"pos":[0,-2,3], "quat":None},
            *args, 
            **kwargs,
        )

    def compute_stage_indicators(self):
        return {}

# VERY HARD TASKS

#########################
####### Pick Box ########
#########################
class PickBoxStages(BiGymStages, PickBox):
    def __init__(self, obs_mode, img_size, *args, **kwargs):
        self.n_stages = 4
        self.reward_mode = "semi_sparse"
        self.max_episode_steps = 500
        action_mode=JointPositionActionMode(floating_base=True, floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ])

        super().__init__(
            obs_mode=obs_mode,
            img_size=img_size,
            action_mode=action_mode,
            control_frequency=CONTROL_FREQUENCY_MIN,
            ext_camera={"pos":[0.7,-2.5,2.5], "quat":None},
            *args, 
            **kwargs,
        )

        # Resize action space
        low_ = np.array([-0.01609539, -0.0205579 , -0.0255008 , -0.05639558, -0.27723205,
                        -0.20132488, -0.14135206, -0.4655914 , -0.22796647, -0.56218374,
                        -0.16436169, -0.09876764, -0.5635499 , -0.8177019 ,  0.        ,
                            0.        ])
        
        high_ = np.array([0.02267784, 0.02073249, 0.10178855, 0.06766292, 0.39900678,
                        0.05100107, 0.05727863, 0.6134118 , 0.50238436, 0.70846236,
                        0.19214949, 0.16949272, 0.70000005, 0.65224624, 0.        ,
                        0.        ])
        
        low_[:-2] = -np.maximum(np.abs(low_[:-2]), np.abs(high_[:-2]))
        high_[:-2] = np.maximum(np.abs(low_[:-2]), np.abs(high_[:-2]))
        
        self.action_space_ = Box(low=low_ - 0.01, high=high_ + 0.01)
        self.action_space = copy.deepcopy(self.action_space_)

    def compute_stage_indicators(self):
        box_pose = self.box.get_pose()
        counter_pose = self.cabinet_base.counter.get_position()

        is_object_taken = (not self.box.is_colliding(self.floor)) and np.all([np.allclose(self.box.get_pose()[:3], self.robot.get_hand_pos(side), atol=0.2) for side in self.robot.grippers]) 
        is_object_lifted = box_pose[2] > counter_pose[2]
        is_object_above_counter = np.allclose(box_pose[:2], counter_pose[:2], atol=0.4) and is_object_lifted
        
        return {
            "stage_1": float(is_object_taken or is_object_above_counter),
            "stage_2": float(is_object_lifted or self.success),
            "stage_3": float(is_object_above_counter or self.success),
        }

SUPPORTED_TASKS = OrderedDict(
    (
        ("reach-target-semi",           ReachTargetStages),
        ("pick-box-semi",               PickBoxStages),
        ("flip-cup-semi",               FlipCupStages), 
        ("drawer-top-open-semi",        DrawerTopOpenStages), 
        ("move-plate-semi",             MovePlateStages), 
        ("wall-cupboard-open-semi",     WallCupboardOpenStages), 
        ("dishwasher-close-trays-semi", DishwasherCloseTraysStages),              
    )
)