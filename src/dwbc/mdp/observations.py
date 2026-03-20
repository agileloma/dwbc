from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def body_mass(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get mass of specified bodies."""
    asset = env.scene[asset_cfg.name]
    # asset.data.model.body_mass: shape (num_envs, 1, num_bodies)
    masses = asset.data.model.body_mass.squeeze(1)  # -> (num_envs, num_bodies)
    if asset_cfg.body_ids is not None:
        masses = masses[:, asset_cfg.body_ids]
    return masses

def joint_torques(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get joint torques (actuator forces in joint space)."""
    asset = env.scene[asset_cfg.name]
    # asset.data.data.qfrc_actuator: shape (num_envs, nv)
    torques = asset.data.data.qfrc_actuator[:, asset.indexing.joint_v_adr]  # (num_envs, num_joints)
    if asset_cfg.joint_ids is not None:
        torques = torques[:, asset_cfg.joint_ids]
    return torques


def contact_sensor_field(env: ManagerBasedRlEnv, sensor_name: str, field: str) -> torch.Tensor:
    """Get a specific field from a contact sensor."""
    sensor = env.scene[sensor_name]
    data = sensor.data
    if field == "found":
        return data.found.float()
    elif field == "force":
        return data.force
    elif field == "torque":
        return data.torque
    elif field == "dist":
        return data.dist
    elif field == "pos":
        return data.pos
    elif field == "normal":
        return data.normal
    elif field == "tangent":
        return data.tangent
    else:
        raise ValueError(f"Unknown field '{field}' for contact sensor.")