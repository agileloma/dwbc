# Copyright (c) 2026, Harbin Institute of Technology, Weihai.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from dwbc.env_cfgs import make_b2z1_flat_env_cfg
from dwbc.rl_cfg import b2z1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Walk-Flat-B2Z1",
  env_cfg=make_b2z1_flat_env_cfg(),
  play_env_cfg=make_b2z1_flat_env_cfg(),
  rl_cfg=b2z1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)