import gymnasium as gym
import numpy as np
import numpy.typing as npt

SUPPORTED_REWARD_MODES = ("dense", "sparse", "semi_sparse", "drS")

def getRewardWrapper(task: str):
    if task.startswith("assembly"):
        return Assembly_DrS
    if task.startswith("pick-place"):
        return PickAndPlace_DrS
    if task.startswith("peg-insert-side"):
        return PegInsertSide_DrS
    if task.startswith("pick-place-wall"):
        return PickAndPlaceWall_DrS
    if task.startswith("stick-push"):
        return StickPush_DrS
    if task.startswith("stick-pull"):
        return StickPull_DrS
    raise NotImplementedError(f"Task {task} is not supported yet.")

class MetaWorldRewardWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, cfg):
        super().__init__(env)
        if cfg.reward_mode not in SUPPORTED_REWARD_MODES:
            self.reward_mode = SUPPORTED_REWARD_MODES[0]
        else:
            self.reward_mode = cfg.reward_mode

    def step(self, action: npt.NDArray[np.float32]):
        obs, rew, termindated, truncated, info = self.env.step(action)
        if self.reward_mode == "sparse":
            rew  = float(info["success"])
        elif self.reward_mode == "dense":
            rew = rew
        elif self.reward_mode == "semi_sparse" or self.reward_mode == "drS":
            rew = self.compute_semi_sparse_reward(info)
        else:
            raise NotImplementedError(self.reward_mode)
        return obs, rew, termindated, truncated, info 

    def compute_stage_indicator(self):
        raise NotImplementedError()
        
    def compute_semi_sparse_reward(self, info):
        stage_indicators = self.compute_stage_indicator(info)
        assert len(stage_indicators.keys()) <= self.n_stages
        reward = sum(stage_indicators.values())
        assert reward.is_integer(), "Semi-sparse reward is not an integer"
        return reward

############################################
# Assembly (Hard)
############################################
class Assembly_DrS(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2
    
    def compute_stage_indicator(self, eval_info):
        return {
            'is_grasped': float(eval_info['grasp_success'] or eval_info['success']),
            'success': float(eval_info['success'])
        }
    
############################################
# Pick And Place (Hard)
############################################
class PickAndPlace_DrS(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2
    
    def compute_stage_indicator(self, eval_info):
        return {
            'is_grasped': float(eval_info['grasp_success'] or eval_info['success']),
            'success': float(eval_info['success'])
        }
    
############################################
# Peg Insert Side (Medium)
############################################
class PegInsertSide_DrS(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2
    
    def compute_stage_indicator(self, eval_info):
        return {
            'is_grasped': float(eval_info['grasp_success'] or eval_info['success']),
            'success': float(eval_info['success'])
        }
    
############################################
# Stick Pull (Very Hard)
############################################
class StickPull_DrS(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 3
    
    def compute_stage_indicator(self, eval_info):
        return {
            'is_grasped': float(eval_info['grasp_success'] or eval_info['success']),
            'in_place': float(np.floor(eval_info['in_place_reward']) or eval_info['success']),
            'success': float(eval_info['success'])
        }
    
############################################
# Stick Push (Very Hard)
############################################
class StickPush_DrS(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 3
    
    def compute_stage_indicator(self, eval_info):
        return {
            'is_grasped': float(eval_info['grasp_success'] or eval_info['success']),
            'in_place': float(np.floor(eval_info['in_place_reward']) or eval_info['success']),
            'success': float(eval_info['success'])
        }
    
############################################
# Pick And Place (Hard)
############################################
class PickAndPlaceWall_DrS(MetaWorldRewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = 2
    
    def compute_stage_indicator(self, eval_info):
        return {
            'is_grasped': float(eval_info['grasp_success'] or eval_info['success']),
            'success': float(eval_info['success'])
        }