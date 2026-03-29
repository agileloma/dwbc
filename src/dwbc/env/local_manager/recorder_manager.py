# # local_recorder_manager.py
# from copy import deepcopy
# from dataclasses import dataclass
# from typing import Dict, Any, Callable, Optional
# import torch
# import numpy as np

# @dataclass
# class RecorderTermCfg:
#     """记录项配置"""
#     func: Callable
#     weight: float = 1.0
#     history_length: int = 1
#     record_pre_step: bool = True
#     record_post_step: bool = True
#     record_pre_reset: bool = True
#     record_post_reset: bool = True

# class RecorderManager:
#     """完整的记录管理器实现"""
    
#     def __init__(self, cfg: Dict[str, RecorderTermCfg], env):
#         self.cfg = deepcopy(cfg)
#         # super().__init__(env=env)

#         self.env = env
#         # self.active_terms = cfg
#         self.active_terms: Dict[str, RecorderTermCfg] = cfg  # 添加类型提示
#         self._initialize_buffers()
        
#     def _initialize_buffers(self):
#         """初始化所有记录缓冲区"""
#         self.pre_step_buf = {}
#         self.post_step_buf = {}
#         self.pre_reset_buf = {}
#         self.post_reset_buf = {}
#         self.episode_sums = {}
#         self.step_counter = 0
        
#         for term_name in self.active_terms:
#             self.pre_step_buf[term_name] = []
#             self.post_step_buf[term_name] = []
#             self.pre_reset_buf[term_name] = []
#             self.post_reset_buf[term_name] = []
#             self.episode_sums[term_name] = torch.zeros(
#                 self.env.num_envs, device=self.env.device
#             )
    
#     def record_pre_step(self):
#         """记录步开始前的状态"""
#         self.step_counter += 1
#         for term_name, term_cfg in self.active_terms.items():
#             if term_cfg.record_pre_step:
#                 try:
#                     value = term_cfg.func(
#                         self.env, 
#                         phase='pre_step',
#                         step=self.step_counter
#                     )
#                     self.pre_step_buf[term_name].append(value)
#                 except Exception as e:
#                     print(f"Record pre_step error - {term_name}: {e}")
    
#     def record_post_step(self):
#         """记录步完成后的状态"""
#         for term_name, term_cfg in self.active_terms.items():
#             if term_cfg.record_post_step:
#                 try:
#                     value = term_cfg.func(
#                         self.env,
#                         phase='post_step',
#                         step=self.step_counter
#                     )
#                     self.post_step_buf[term_name].append(value)
                    
#                     # 更新episode累计和
#                     if isinstance(value, torch.Tensor):
#                         self.episode_sums[term_name] += value
#                     elif isinstance(value, (int, float)):
#                         self.episode_sums[term_name] += value
                        
#                 except Exception as e:
#                     print(f"Record post_step error - {term_name}: {e}")
    
#     def record_pre_reset(self, env_ids):
#         """记录重置前的状态"""
#         for term_name, term_cfg in self.active_terms.items():
#             if term_cfg.record_pre_reset:
#                 try:
#                     value = term_cfg.func(
#                         self.env,
#                         env_ids=env_ids,
#                         phase='pre_reset',
#                         step=self.step_counter
#                     )
#                     self.pre_reset_buf[term_name].append({
#                         'env_ids': env_ids.cpu().numpy() if torch.is_tensor(env_ids) else env_ids,
#                         'value': value,
#                         'step': self.step_counter
#                     })
#                 except Exception as e:
#                     print(f"Record pre_reset error - {term_name}: {e}")
    
#     def record_post_reset(self, env_ids):
#         """记录重置后的状态"""
#         for term_name, term_cfg in self.active_terms.items():
#             if term_cfg.record_post_reset:
#                 try:
#                     value = term_cfg.func(
#                         self.env,
#                         env_ids=env_ids,
#                         phase='post_reset',
#                         step=self.step_counter
#                     )
#                     self.post_reset_buf[term_name].append({
#                         'env_ids': env_ids.cpu().numpy() if torch.is_tensor(env_ids) else env_ids,
#                         'value': value,
#                         'step': self.step_counter
#                     })
                    
#                     # 重置episode累计和
#                     if torch.is_tensor(env_ids):
#                         self.episode_sums[term_name][env_ids] = 0
                        
#                 except Exception as e:
#                     print(f"Record post_reset error - {term_name}: {e}")
    
#     def get_statistics(self, term_name=None):
#         """获取记录统计信息"""
#         stats = {}
#         target_terms = [term_name] if term_name else self.active_terms.keys()
        
#         for name in target_terms:
#             if name in self.episode_sums:
#                 stats[name] = {
#                     'mean': self.episode_sums[name].mean().item(),
#                     'std': self.episode_sums[name].std().item(),
#                     'max': self.episode_sums[name].max().item(),
#                     'min': self.episode_sums[name].min().item(),
#                 }
#         return stats
    
#     def clear_history(self):
#         """清空历史记录(保持episode累计和)"""
#         self.pre_step_buf.clear()
#         self.post_step_buf.clear()
#         self.pre_reset_buf.clear()
#         self.post_reset_buf.clear()
#         self._initialize_buffers()  # 重新初始化缓冲区


# # # 预定义的记录函数
# # def record_reward(env, **kwargs):
# #     """记录奖励"""
# #     return env.reward_buf.clone()

# # def record_arm_reward(env, **kwargs):
# #     """记录机械臂奖励"""
# #     return env.arm_reward_buf.clone() if hasattr(env, 'arm_reward_buf') else None

# # def record_episode_length(env, **kwargs):
# #     """记录episode长度"""
# #     return env.episode_length_buf.clone()

# # def record_action(env, **kwargs):
# #     """记录动作"""
# #     return env.action_manager.action.clone()

# # def record_termination(env, **kwargs):
# #     """记录终止原因"""
# #     phase = kwargs.get('phase', '')
# #     if phase == 'pre_reset':
# #         return {
# #             'terminated': env.reset_terminated.clone(),
# #             'time_outs': env.reset_time_outs.clone()
# #         }
# #     return None