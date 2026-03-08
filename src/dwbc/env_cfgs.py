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

from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

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
    )
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
  actions: dict[str, ActionTermCfg] = {
    "joint_positions": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=B2Z1_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##
  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0), 
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
      )
    )
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
    )
  }

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