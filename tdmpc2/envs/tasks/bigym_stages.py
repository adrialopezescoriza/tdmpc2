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

from collections import OrderedDict
import numpy as np

class BiGymStages(BiGymEnv):
    def __init__(self, obs_mode, img_size, *args, **kwargs):
        if obs_mode.startswith("rgb"):
            observation_config=ObservationConfig(
                cameras=[
                    CameraConfig(
                        name="head",
                        rgb=True,
                        depth=False,
                        resolution=(img_size, img_size),
                    )
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

########################################
########   Wall Cupboard Open   ########
########################################
class WallCupboardOpenStages(BiGymStages, WallCupboardOpen):
    def __init__(self, obs_mode, img_size, *args, **kwargs):
        self.n_stages = 3
        self.reward_mode = "semi_sparse"
        self.max_episode_steps = 200
        action_mode=JointPositionActionMode(floating_base=True, floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ], absolute=True)

        super().__init__(
            obs_mode=obs_mode,
            img_size=img_size,
            action_mode=action_mode,
            control_frequency=CONTROL_FREQUENCY_MIN * 4,
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
            control_frequency=CONTROL_FREQUENCY_MIN * 4,
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
        self.max_episode_steps = 200
        action_mode=JointPositionActionMode(floating_base=True, floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ], absolute=True)

        super().__init__(
            obs_mode=obs_mode,
            img_size=img_size,
            action_mode=action_mode,
            control_frequency=CONTROL_FREQUENCY_MIN * 4,
            *args, 
            **kwargs,
        )

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
        self.n_stages = 5
        self.reward_mode = "semi_sparse"
        self.max_episode_steps = 200
        action_mode=JointPositionActionMode(floating_base=True, floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.RZ])

        super().__init__(
            obs_mode=obs_mode,
            img_size=img_size,
            action_mode=action_mode,
            control_frequency=CONTROL_FREQUENCY_MIN * 4,
            *args, 
            **kwargs,
        )

    def compute_stage_indicators(self):
        plate = self.plates[0]
        plate_pose = plate.get_pose()

        is_plate_grasped = np.array([self.robot.is_gripper_holding_object(plate, side) for side in self.robot.grippers]).any()
        is_plate_lifted = not (plate.is_colliding(self.rack_start) and plate_pose[2] > self.rack_start.get_pose()[2])

        is_plate_above_target = np.allclose(plate_pose[:2], self.rack_target.get_pose()[:2], atol=0.2)
        is_plate_on_target = plate.is_colliding(self.rack_target)

        return {
            "stage1": is_plate_grasped or is_plate_above_target,
            "stage2": ((is_plate_lifted or is_plate_on_target) and is_plate_grasped) or is_plate_above_target,
            "stage3": is_plate_above_target,
            "stage4": is_plate_on_target,
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
            control_frequency=CONTROL_FREQUENCY_MIN * 4,
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
        self.max_episode_steps = 100
        action_mode=JointPositionActionMode(floating_base=True, floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.RZ])

        super().__init__(
            obs_mode=obs_mode,
            img_size=img_size,
            action_mode=action_mode,
            control_frequency=CONTROL_FREQUENCY_MIN * 4,
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
        self.max_episode_steps = 200
        action_mode=JointPositionActionMode(floating_base=True, floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ])

        super().__init__(
            obs_mode=obs_mode,
            img_size=img_size,
            action_mode=action_mode,
            control_frequency=CONTROL_FREQUENCY_MIN * 4,
            *args, 
            **kwargs,
        )

    def compute_stage_indicators(self):
        box_pose = self.box.get_pose()
        counter_pose = self.cabinet_base.counter.get_position()

        is_object_taken = not self.box.is_colliding(self.floor)
        is_object_lifted = box_pose[2] > counter_pose[2]
        is_object_above_counter = np.allclose(box_pose[:2], counter_pose[:2], atol=0.4) and is_object_lifted
        
        return {
            "stage_1": float(is_object_taken or self.success),
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