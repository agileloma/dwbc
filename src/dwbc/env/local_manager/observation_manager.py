# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers import ObservationManager as ObservationManagerBase


class ObservationManager(ObservationManagerBase):

    def compute_obs(self):
        """计算并返回特权观察的维度总和"""
        num_priv: int = 0  # 特权观察的总维度
        
        for group_name in self._group_obs_term_names:
            # 验证组名
            if group_name not in self._group_obs_term_names:
                raise ValueError(
                    f"Unable to find the group '{group_name}' in the observation manager."
                    f" Available groups are: {list(self._group_obs_term_names.keys())}"
                )
            
            # 获取当前组的观察项
            group_term_names = self._group_obs_term_names[group_name]
            obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])
            
            # 只统计特权观察的维度
            for term_name, term_cfg in obs_terms:
                if term_name.startswith("priv_"):
                    # 计算特权观察的维度
                    obs_value = term_cfg.func(self._env, **term_cfg.params).clone()
                    num_priv += obs_value.shape[1]
        
        # 返回特权观察的总维度
        return num_priv
