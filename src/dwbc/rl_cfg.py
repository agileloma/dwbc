# Copyright (c) 2026, Harbin Institute of Technology, Weihai.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


"""RL configuration for Unitree B2Z1 velocity task."""

from mjlab.rl import (
    B2Z1RslRlModelCfg,
    B2Z1RslRlOnPolicyRunnerCfg,
    B2Z1RslRlPpoAlgorithmCfg,
)


def b2z1_ppo_runner_cfg() -> B2Z1RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree B2Z1 velocity task."""
  return B2Z1RslRlOnPolicyRunnerCfg(
    actor=B2Z1RslRlModelCfg(
      num_arm_actions= 6,
      num_leg_actions= 12,
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      # 自定义参数（维度由训练脚本动态填充）
      leg_control_head_hidden_dims=(128, 64),
      arm_control_head_hidden_dims=(128, 64),
      class_name = "ActorWithEncoders" ,
      priv_encoder_dims=(64, 18),
      activation_out="tanh",
    ),
    critic=B2Z1RslRlModelCfg(
      num_arm_actions= 6,
      num_leg_actions= 12,
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      critic_leg_control_head_hidden_dims=(128, 64),
      critic_arm_control_head_hidden_dims=(128, 64),
      class_name = "CriticWithEncoders",
    ),
    algorithm=B2Z1RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      # 新增参数
      mixing_schedule=(0.5, 2000, 4000),
      dagger_update_freq=20,
      priv_reg_coef_schedual=(0.0, 0.0, 0, 1),  # 若不使用正则化，设为全0
      eps=1e-5,
    ),
    experiment_name="b2z1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )