# Copyright (c) 2026, Harbin Institute of Technology, Weihai.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


"""Unitree B2Z1 velocity environment configurations."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.managers.observation_manager import ObservationTermCfg, ObservationGroupCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from dwbc import mdp 

from mjlab.sensor.contact_sensor import ContactSensorCfg, ContactMatch

from dwbc.mdp.velocity_command import UniformVelocityCommandCfg
from dwbc.mdp.pose_command import UniformPoseCommandCfg
from dwbc.b2z1.b2z1_constants import B2Z1_ACTION_SCALE, get_b2z1_robot_cfg

def make_b2z1_flat_env_cfg() -> ManagerBasedRlEnvCfg:
  """Core configuration for B2Z1 flat ground walking."""

  ##
  # Scene
  ##
  
  FOOT_GEOM_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


  scene = SceneCfg(
    num_envs=1,
    extent=2.0,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": get_b2z1_robot_cfg()},    
  )

    # 足端接触传感器（检测脚与地面的接触力，用于计算空中时间）
  feet_ground_cfg = ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(
          mode="geom",
          pattern=FOOT_GEOM_NAMES,
          entity="robot",
      ),
      secondary=ContactMatch(
          mode="body",
          pattern="terrain",   # 地形 body 名称，需与 TerrainEntityCfg 生成的一致，若不确定可改为 None
      ),
      fields=("found", "force"),
      reduce="netforce",       # 每个足部所有接触点的合力
      num_slots=1,
      track_air_time=True,     # 开启空气时间追踪
  )

  # 非足端接触传感器（检测除脚以外的其他部分是否触地）
  # nonfoot_ground_cfg = ContactSensorCfg(
  #     name="nonfoot_ground_touch",
  #     primary=ContactMatch(
  #         mode="geom",
  #         entity="robot",
  #         # 匹配所有碰撞几何体（假设命名规则为 "*_collision" 或以数字结尾）
  #         # 注：您的 XML 中碰撞几何体可能没有显式名称，此正则可能需要根据实际情况调整
  #         pattern=r".*_collision\d*$",
  #         exclude=FOOT_GEOM_NAMES,
  #     ),
  #     secondary=ContactMatch(
  #         mode="body",
  #         pattern="terrain",
  #     ),
  #     fields=("found",),
  #     reduce="none",
  #     num_slots=1,
  # )

  # 将传感器添加到场景的 sensors 元组中
  # 注意：SceneCfg 需要有 sensors 字段，如果没有则动态创建
  if not hasattr(scene, "sensors"):
      scene.sensors = ()
  scene.sensors += (feet_ground_cfg, )
      # 将传感器添加到场景，只保留足端传感器


  viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="base_link",
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
  )

  sim = SimulationCfg(
    nconmax=None,
    njmax=300,
    contact_sensor_maxmatch=64,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
      ccd_iterations=50,
    ),
  )

  ##
  # Observations
  ##


  actor_terms = {
      "base_linear_velocity": ObservationTermCfg(
          func=mdp.builtin_sensor,
          params={"sensor_name": "robot/imu_lin_vel"},
      ),
      "base_angular_velocity": ObservationTermCfg(
          func=mdp.builtin_sensor,
          params={"sensor_name": "robot/imu_ang_vel"},
      ),
      "projected_gravity": ObservationTermCfg(
          func=mdp.projected_gravity,
      ),
      "leg_joint_positions": ObservationTermCfg(
          func=mdp.joint_pos_rel,
          params={
              "asset_cfg": SceneEntityCfg(
                  "robot",
                  joint_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint")
              )
          },
      ),
      "leg_joint_velocities": ObservationTermCfg(
          func=mdp.joint_vel_rel,
          params={
              "asset_cfg": SceneEntityCfg(
                  "robot",
                  joint_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint")
              )
          },
      ),
      "actions": ObservationTermCfg(
          func=mdp.last_action,
      ),
      "leg_commands": ObservationTermCfg(
          func=mdp.generated_commands,
          params={"command_name": "twist"},
      ),
      "arm_pose_command": ObservationTermCfg(
          func=mdp.generated_commands,
          params={"command_name": "arm_pose"},
      ),
      "arm_joint_positions": ObservationTermCfg(
          func=mdp.joint_pos_rel,
          params={
              "asset_cfg": SceneEntityCfg(
                  "robot",
                  joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
              )
          },
      ),
      "arm_joint_velocities": ObservationTermCfg(
          func=mdp.joint_vel_rel,
          params={
              "asset_cfg": SceneEntityCfg(
                  "robot",
                  joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
              )
          },
      ),
      # "foot_contacts": ObservationTermCfg(
      #     func=mdp.sensor_data,
      #     params={"sensor_name": "feet_ground_contact", "field": "found"},
      # ),
  }
  
  
  critic_terms = {
    **actor_terms,
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##
  # actions: dict[str, ActionTermCfg] = {
  #   "joint_positions": JointPositionActionCfg(
  #     entity_name="robot",
  #     actuator_names=(".*",),
  #     scale=B2Z1_ACTION_SCALE,
  #     # scale=0.5,  # Override per-robot.
  #     use_default_offset=True,
  #   )
  # }

  actions: dict[str, ActionTermCfg] = {
    "joint_positions": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", 
        ),
        scale={
            # 保留腿部的自动缩放（或手动指定）
            ".*_hip_joint": 0.25 * 200 / 250,   # 示例值
            ".*_thigh_joint": 0.25 * 200 / 250,
            ".*_calf_joint": 0.25 * 320 / 300,
            # 臂关节手动指定较大值
            "joint[1-6]": 0.5,   # 直接设为 0.5 弧度
        },
        use_default_offset=True,
    )
  } 

  joint_pos_action = actions["joint_positions"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  # joint_pos_action.scale = B2Z1_ACTION_SCALE

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    # leg
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      is_B2Z1=True,  # 对应原代码的 is_Go2ARM=True
      curriculum_coeff=1000,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.2, 1.0), 
        lin_vel_y=(-0.5, 0.5), 
        ang_vel_z=(-0.5, 0.5),
        heading=(-0.0, 0.0)
        ),
      
      ranges_final=UniformVelocityCommandCfg.Ranges(
          lin_vel_x=(0.1, 0.8), 
          lin_vel_y=(-0.5, 0.5), 
          ang_vel_z=(-0.5, 0.5),
          heading=(-0.0, 0.0)
        ),
      
      ranges_init=UniformVelocityCommandCfg.Ranges(
          lin_vel_x=(0.1, 0.35), 
          lin_vel_y=(-0.1, 0.1), 
          ang_vel_z=(-0.1, 0.1),
          heading=(-0.0, 0.0)
      )
    ),  
    # arm
    "arm_pose": UniformPoseCommandCfg(      
      entity_name="robot",
      body_name="link06",          
      resampling_time_range=(3.0, 8.0), 
      is_B2Z1=True,  # 对应原代码的 is_Go2ARM=True
      curriculum_coeff=1000,   

      # 初始训练范围（简单）
      ranges_init=UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.5),      # 小范围，靠近身体
            pos_y=(-0.05, 0.05),    # 几乎在中心线
            pos_z=(0.35, 0.4),       # 中等高度小范围
            roll=(-0.0, 0.0),        # 无滚转
            pitch=(-0.0, 0.0),       # 无俯仰  
            yaw=(-0.0, 0.0),         # 无偏航
        ),
        
      # 中期训练范围（中等难度）
      ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),        # 前后范围
            pos_y=(-0.35, 0.35),      # 左右范围
            pos_z=(0.1, 0.55),        # 高度范围
            roll=(-0.0, 0.0),         # 仍无滚转（与原始一致）
            pitch=(-0.35, 0.35),      # ±20度俯仰
            yaw=(-0.35, 0.35),        # ±20度偏航
        ),
        
      # 最终训练范围（完整难度）
      ranges_final=UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),         # 保持相同位置范围
            pos_y=(-0.35, 0.35),
            pos_z=(0.1, 0.55),
            roll=(-0.0, 0.0),          # 仍无滚转
            pitch=(-0.35, 0.35),       # ±20度俯仰
            yaw=(-0.35, 0.35),         # ±20度偏航
        ),

      make_quat_unique=True,               
    ),
  }


  ##
  # Rewards
  ##
  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity, 
      weight=2.0,
      params={
        "command_name": "twist", 
        "std": math.sqrt(0.25)
      }
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={
        "command_name": "twist", 
        "std": math.sqrt(0.5)
      }
    ),
    "upright": RewardTermCfg(
      func=mdp.flat_orientation,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link")), 
        "std": math.sqrt(0.2)
      }
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2, 
      weight=-0.01,
    ),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits, 
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    ),
    
    # 腿部专用奖励项
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,  
      weight=-0.02,  
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      }
    ),

    "leg_joint_deviation": RewardTermCfg(
      func=mdp.joint_deviation_l1,  
      weight=-0.03,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_thigh_joint", ".*_calf_joint")),
      }
    ),


    "leg_joint_torques":RewardTermCfg(
      func=mdp.joint_torques_l2,  
      weight=-2.5e-7,
       params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                actuator_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint")
            )
        },
    ),


    "leg_joint_vel": RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-1e-4,  # 速度惩罚通常比力矩惩罚大一些
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint")
            )
        },
    ),

    "leg_joint_acc": RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-1.0e-6,#-2.5e-7,  # 加速度惩罚通常比速度惩罚小
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint")
            )
        },
    ),

    "height_reward": RewardTermCfg(
      func=mdp.base_height_l2,  
      weight=-2.0,
      params={"target_height": 0.5}
    ),

    "feet_height_body": RewardTermCfg(
        func=mdp.feet_height_body,
        weight=-1.0,#-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_calf"),
            "tanh_mult": 2.0,
            "target_height": -0.5,
            "command_name": "twist",  
        },
    ),

    "feet_air_time": RewardTermCfg(
        func=mdp.feet_air_time,
        weight=0.5, #0.3,
        params={
            "sensor_name": "feet_ground_contact",  
            "threshold_min": 0.1,
            "threshold_max": 0.3,
            "command_name": "twist",
            "command_threshold": 0.5,
        },
    ),
    
    "feet_slip": RewardTermCfg(
        func=mdp.feet_slip,
        weight=-0.1, #-0.2,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.01,
            "asset_cfg": SceneEntityCfg("robot", site_names=("FL_foot_site", "FR_foot_site", "RL_foot_site", "RR_foot_site")),
        },
    ),

    # arm
    "end_effector_position_tracking": RewardTermCfg(
        func=mdp.track_pose_position,        
        weight=1.0,
        params={
            "command_name": "arm_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="link06"),
            "std": 0.1,                    
        }
    ),
    "end_effector_orientation_tracking": RewardTermCfg(
        func=mdp.track_pose_orientation,      
        weight=0.5,
        params={
            "command_name": "arm_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="link06"),
            "std": 0.2,
        }
    ),
    
    "end_effector_action_rate": RewardTermCfg(
      func=mdp.action_rate_l2, 
      weight=-0.01,
    ),
        # ===== 机械臂关节偏差惩罚 =====
        # 机械臂的关节偏差惩罚（负权重，但比腿部更严格）
    "end_effector_joint_deviation": RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.05,#-0.08,  # 机械臂需要更精确的位置控制，权重更高
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(
                    "joint1",  # 腰部
                    "joint2",  # 肩关节
                    "joint3",  # 肘关节
                    "joint4",  # 腕部角度
                    "joint5",  # 前臂旋转
                    "joint6",  # 腕部旋转
                    # "jointGripper",  # 夹爪（如果需要）
                )
            )
        },
    ),

    # ===== 机械臂关节惩罚 =====
    # 机械臂关节力矩惩罚
    "end_effector_joint_torques": RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-5.0e-7,#-2.0e-6,#-5.0e-7,  # 机械臂需要更精确的力矩控制，权重提高2倍
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(
                    "joint1", "joint2", "joint3",
                    "joint4", "joint5", "joint6"
                )
            )
        },
    ),
    
    # 机械臂关节速度惩罚
    "end_effector_joint_vel": RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-2e-4,  # 机械臂速度控制更重要，权重提高2倍
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(
                    "joint1", "joint2", "joint3",
                    "joint4", "joint5", "joint6"
                )
            )
        },
    ),
  
    # 机械臂关节加速度惩罚
    "end_effector_acc": RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.0e-6,#-2.0e-5,#-5.0e-7,  # 机械臂加速度控制更重要，权重提高2倍
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(
                    "joint1", "joint2", "joint3",
                    "joint4", "joint5", "joint6"
                )
            )
        },
    ),


  }




  ##
  # Events
  ##

  ##
  # Terminations
  ##
  terminations = {
    "time_out": TerminationTermCfg(
      func=mdp.time_out, 
      time_out=True
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation, 
      params={"limit_angle": math.radians(70.0)}
    )
  }

  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    sim=sim,
    viewer=viewer,
    rewards=rewards,
    terminations=terminations,
    commands=commands,
    decimation=4,
    episode_length_s=20.0,
  )