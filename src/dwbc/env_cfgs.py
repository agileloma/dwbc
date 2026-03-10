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

from dwbc.mdp.velocity_command import UniformVelocityCommandCfg
from dwbc.mdp.pose_command import UniformPoseCommandCfg
from dwbc.b2z1.b2z1_constants import B2Z1_ACTION_SCALE, get_b2z1_robot_cfg

def make_b2z1_flat_env_cfg() -> ManagerBasedRlEnvCfg:
  """Core configuration for B2Z1 flat ground walking."""

  ##
  # Scene
  ##

  scene = SceneCfg(
    num_envs=1,
    extent=2.0,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": get_b2z1_robot_cfg()},
  )

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
    "joint_positions": ObservationTermCfg(
      func=mdp.joint_pos_rel,
    ),
    "joint_velocities": ObservationTermCfg(
      func=mdp.joint_vel_rel
    ),
    "actions": ObservationTermCfg(
      func=mdp.last_action,
    ),
    "commands": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "arm_pose_command": ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "arm_pose"},
    ),
    "arm_joint_positions": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"])}
    ),
    "arm_joint_velocities": ObservationTermCfg(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"])}
    ),


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
  # commands: dict[str, CommandTermCfg] = {
  #   "twist": UniformVelocityCommandCfg(
  #     entity_name="robot",
  #     resampling_time_range=(3.0, 8.0),
  #     ranges=UniformVelocityCommandCfg.Ranges(
  #       lin_vel_x=(-1.0, 1.0), 
  #       lin_vel_y=(-1.0, 1.0),
  #       ang_vel_z=(-1.0, 1.0),
  #     )
  #   )
  # }

  commands: dict[str, CommandTermCfg] = {
    # leg
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0), 
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
      )
    ),  
    # arm
    "arm_pose": UniformPoseCommandCfg(      
      entity_name="robot",
      body_name="link06",          
      resampling_time_range=(3.0, 8.0),      
      ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.6),               
            pos_y=(-0.3, 0.3),               
            pos_z=(0.0, 0.4),                
            roll=(-0.5, 0.5),              
            pitch=(-0.5, 1.0),           
            yaw=(-1.0, 1.0),                    
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
    ),
    
    # arm
    "track_arm_position": RewardTermCfg(
        func=mdp.track_pose_position,        
        weight=1.0,
        params={
            "command_name": "arm_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="link06"),
            "std": 0.1,                    
        }
    ),
    "track_arm_orientation": RewardTermCfg(
        func=mdp.track_pose_orientation,      
        weight=0.5,
        params={
            "command_name": "arm_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="link06"),
            "std": 0.2,
        }
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