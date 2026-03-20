# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from mjlab.managers import ObservationManager as ObservationManagerBase


class ObservationManager(ObservationManagerBase):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.num_history = env.cfg.num_history


    def compute_priv_dim(self) -> int:
        """返回特权观察组（例如 "privileged"）的总维度。

        假设配置中存在名为 "privileged" 的观察组。
        如果不存在，返回 0。

        根据组配置（是否 concatenate）自动计算总维度。
        """
        if "privileged" not in self._group_obs_dim:
            return 0

        group_dims = self._group_obs_dim["privileged"]
        # 如果组未 concatenate，group_dims 是一个列表，每个元素是一个 term 的维度元组
        if isinstance(group_dims, list) and all(isinstance(d, tuple) for d in group_dims):
            total = 0
            for term_dims in group_dims:
                total += int(np.prod(term_dims))
            return total
        else:
            # 已 concatenate，直接计算乘积
            return int(np.prod(group_dims))