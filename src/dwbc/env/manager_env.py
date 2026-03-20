from mjlab.envs import ManagerBasedRlEnv
from . import local_manager
import torch


class B2Z1ManagerRLEnv(ManagerBasedRlEnv):
    """Configuration for the locomotion velocity-tracking environment."""
    def __init__(self, cfg, render_mode, device, **kwargs):
        super().__init__(cfg=cfg, device=device, render_mode=render_mode, **kwargs)
        self._sim_step_counter = 0
        self.arm_reward_buf = None

    def load_managers(self):
        super().load_managers()
        self.reward_manager = local_manager.RewardManager(self.cfg.rewards, self)
        self.observation_manager = local_manager.ObservationManager(self.cfg.observations, self)
        # self.recorder_manager = local_manager.RecorderManager(self.cfg.recorders, self)
        self._configure_gym_env_spaces()

    # ---------- 新增属性 ----------
    @property
    def num_prop(self) -> int:
        """返回 proprioceptive 观测的原始维度（不含历史堆叠）。"""
        actor_dim = self.observation_manager.group_obs_dim["actor"][0]  # 改为 "actor"
        return actor_dim // self.cfg.num_history

    @property
    def num_priv(self) -> int:
        """返回特权观察的总维度。"""
        return self.observation_manager.compute_priv_dim()

    @property
    def num_history(self) -> int:
        """返回历史步数，从配置中读取。"""
        return self.cfg.num_history


    def step(self, action) :
        
        obs_dict, reward, terminated, timeout, extras = super().step(action)
        total_reward, arm_reward = self.reward_manager.get_last_rewards()  
        dones = (terminated | timeout).to(dtype=torch.long)
        return obs_dict, total_reward, arm_reward, dones, extras


