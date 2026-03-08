# Copyright (c) 2026, Harbin Institute of Technology, Weihai.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.os import update_assets
from mjlab.utils.actuator import ElectricActuator, reflected_inertia


##
# MJCF and assets.
##

_HERE_ = Path(__file__).parent

B2Z1_XML: Path = (
    _HERE_ / "xmls" / "b2z1.xml"
)
assert B2Z1_XML.exists()

def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, B2Z1_XML.parent / "assets", meshdir)
  return assets

def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(B2Z1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec

##
# Actuator config.
##

HIP_EFFORT_LIMIT = 200.0
KNEE_EFFORT_LIMIT = 320.0
ARM_EFFORT_LIMIT = 30.0

# Leg rotor inertia.
# Ref: https://github.com/unitreerobotics/unitree_ros/blob/master/robots/b2_description/urdf/b2_description.urdf#L156
# Extracted Ixx (rotation along x-axis).
LEG_ROTOR_INERTIA = 0.000144463

# Arm rotor inertia.
ARM_ROTOR_INERTIA = 0.000144463

# Gearbox.
HIP_GEAR_RATIO = 9.0
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5
ARM_GEAR_RATIO = 60.0

HIP_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(LEG_ROTOR_INERTIA, HIP_GEAR_RATIO),
  velocity_limit=23.0,
  effort_limit=HIP_EFFORT_LIMIT,
)
KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(LEG_ROTOR_INERTIA, KNEE_GEAR_RATIO),
  velocity_limit=14.0,
  effort_limit=KNEE_EFFORT_LIMIT,
)

ARM_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ARM_ROTOR_INERTIA, ARM_GEAR_RATIO),
  velocity_limit=3.14,
  effort_limit=ARM_EFFORT_LIMIT,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIP = HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_HIP = 2 * DAMPING_RATIO * HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_KNEE = KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_KNEE = 2 * DAMPING_RATIO * KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_ARM = ARM_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_ARM = 2 * DAMPING_RATIO * ARM_ACTUATOR.reflected_inertia * NATURAL_FREQ

B2Z1_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=STIFFNESS_HIP,
  damping=DAMPING_HIP,
  effort_limit=HIP_ACTUATOR.effort_limit,
  armature=HIP_ACTUATOR.reflected_inertia,
)

B2Z1_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
  effort_limit=KNEE_ACTUATOR.effort_limit,
  armature=KNEE_ACTUATOR.reflected_inertia,
)

B2Z1_ARM_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("joint[1-6]",),
  stiffness=STIFFNESS_ARM,
  damping=DAMPING_ARM,
  effort_limit=ARM_ACTUATOR.effort_limit,
  armature=ARM_ACTUATOR.reflected_inertia,
)

##
# Keyframe.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.5),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    ".*hip_joint": 0.0,
    ".*thigh_joint": 1.28,
    ".*calf_joint": -2.84,
    "joint*": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

B2Z1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    B2Z1_HIP_ACTUATOR_CFG,
    B2Z1_KNEE_ACTUATOR_CFG,
    B2Z1_ARM_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)

def get_b2z1_robot_cfg() -> EntityCfg:
  """Get a fresh B2Z1 robot configuration instance.
  
  Return a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
      spec_fn=get_spec,
      articulation=B2Z1_ARTICULATION,
  )

B2Z1_ACTION_SCALE: dict[str, float] = {}
for a in B2Z1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    B2Z1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_b2z1_robot_cfg())

  viewer.launch(robot.spec.compile())