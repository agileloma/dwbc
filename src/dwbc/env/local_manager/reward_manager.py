# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py

from __future__ import annotations
import torch
from mjlab.managers import RewardManager as RewardManagerBase

class RewardManager(RewardManagerBase):

    def __init__(self, cfg, env, scale_by_dt=True):
        super().__init__(cfg, env, scale_by_dt=scale_by_dt)
        self._arm_reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self._last_total_reward = None
        self._last_arm_reward = None

    def compute(self, dt: float) -> torch.Tensor:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # reset computation
        self._reward_buf[:] = 0.0
        self._arm_reward_buf[:] = 0.0 
        scale = dt if self._scale_by_dt else 1.0

        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs, strict=False)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * scale
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 根据名字前缀区分奖励
            if name.startswith("end_effector"):  
                self._arm_reward_buf += value
            else:
                self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value

            # Update current reward for this step.
            self._step_reward[:, term_idx] = value / scale

        self._last_total_reward = self._reward_buf.clone()
        self._last_arm_reward = self._arm_reward_buf.clone()

        return self._reward_buf
    
    def get_last_rewards(self):
        """返回最近一次计算的（总奖励，臂奖励）元组。"""
        if self._last_total_reward is None or self._last_arm_reward is None:
            raise RuntimeError("Rewards not computed yet. Call compute() first.")
        return self._last_total_reward, self._last_arm_reward