from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.registration import register_env
import numpy as np
from collections import OrderedDict

class DrS_BaseEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ("dense", "sparse", "semi_sparse")

    def compute_stage_indicator(self):
        raise NotImplementedError()

    def _get_obs_state_dict(self) -> OrderedDict:
        ret = super()._get_obs_state_dict()
        ret['extra'].update(self.compute_stage_indicator())
        return ret
    
    def get_reward(self, **kwargs):
        if self._reward_mode == "sparse":
            eval_info = self.evaluate(**kwargs)
            return float(eval_info["success"])
        elif self._reward_mode == "dense":
            return self.compute_dense_reward(**kwargs)
        elif self._reward_mode == "semi_sparse":
            # reward build from stage indicators
            return self.compute_semi_sparse_reward(**kwargs)
        else:
            raise NotImplementedError(self._reward_mode)
        
    def compute_semi_sparse_reward(self, info, **kwargs):
        stage_indicators = self.compute_stage_indicator()
        eval_info = self.evaluate(**kwargs)
        return sum(stage_indicators.values()) + float(eval_info["success"])


############################################
# Pick And Place
############################################

from mani_skill2.envs.pick_and_place.pick_single import PickSingleYCBEnv, PickSingleEGADEnv
from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv

@register_env("PickAndPlace_DrS_learn-v0", max_episode_steps=100)
class PickAndPlace_DrS_learn(PickSingleYCBEnv, DrS_BaseEnv):
    def check_obj_placed(self):
        obj_to_goal_pos = self.goal_pos - self.obj_pose.p
        return np.linalg.norm(obj_to_goal_pos) <= self.goal_thresh

    def compute_stage_indicator(self):
        return {
            'is_grasped': float(self.agent.check_grasp(self.obj)),
            'is_obj_placed': float(self.check_obj_placed()),
        }

@register_env("PickAndPlace_DrS_reuse-v0", max_episode_steps=100)
class PickAndPlace_DrS_reuse(PickCubeEnv, PickAndPlace_DrS_learn):
    pass


############################################
# Turn Faucet
############################################

from mani_skill2.envs.misc.turn_faucet import (
    TurnFaucetEnv, transform_points, load_json
)
from mani_skill2 import PACKAGE_ASSET_DIR

class TurnFaucetEnv_DrS(TurnFaucetEnv, DrS_BaseEnv):
    
    def _get_obs_extra(self) -> OrderedDict:
        ret = super()._get_obs_extra()
        T = self.target_link.pose.to_transformation_matrix()
        pcd = transform_points(T, self.target_link_pcd)
        T1 = self.lfinger.pose.to_transformation_matrix()
        T2 = self.rfinger.pose.to_transformation_matrix()
        pcd1 = transform_points(T1, self.lfinger_pcd)
        pcd2 = transform_points(T2, self.rfinger_pcd)
        ret.update(
            handle_center=np.mean(pcd, axis=0),
            lfinger_center=np.mean(pcd1, axis=0),
            rfinger_center=np.mean(pcd2, axis=0),
            target_joint_qvel=self.faucet.get_qvel()[self.target_joint_idx],
        )
        return ret

    def _initialize_task(self):
        super()._initialize_task()
        self._last_angle = self.current_angle

    def step_action(self, action):
        self._last_angle = self.current_angle
        super().step_action(action)
    
    def compute_stage_indicator(self):
        delta_angle = self.current_angle - self._last_angle
        success = self.evaluate()['success']

        return {
            'handle_move': float((delta_angle > 1e-3) or success),
        }
    

@register_env("TurnFaucet_DrS_learn-v0", max_episode_steps=100)
class TurnFaucetEnv_DrS_learn(TurnFaucetEnv_DrS):
    def __init__(
        self,
        *args,
        model_ids = (),
        **kwargs,
    ):
        model_ids = ('5028','5063','5034','5000','5006','5039','5056','5020','5027','5041')
        super().__init__(*args, model_ids=model_ids, **kwargs)

@register_env("TurnFaucet_DrS_reuse-v0", max_episode_steps=100)
class TurnFaucetEnv_DrS_reuse(TurnFaucetEnv_DrS):
    def __init__(
        self,
        *args,
        model_ids = (),
        **kwargs,
    ):
        model_json = f"{PACKAGE_ASSET_DIR}/partnet_mobility/meta/info_faucet_train.json"
        model_db = load_json(model_json)
        exclude_model_ids = ('5028','5063','5034','5000','5006','5039','5056','5020','5027','5041')
        model_ids = sorted(model_db.keys())
        model_ids = [model_id for model_id in model_ids if model_id not in exclude_model_ids]
        super().__init__(*args, model_ids=model_ids, **kwargs)