from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,      # 已存在
    wrap_to_pi,      # 已存在
)

# 尝试从 mjlab 数学库导入四元数函数，若不存在则手动实现
try:
    from mjlab.utils.lab_api.math import (
        quat_mul,
        quat_conj,
        quat_from_euler_xyz,
        quat_unique,
    )
except ImportError:
    # 手动实现四元数乘法、共轭、欧拉角转四元数、唯一化
    def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return torch.stack([w, x, y, z], dim=-1)

    def quat_conj(q: torch.Tensor) -> torch.Tensor:
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return torch.stack([qw, qx, qy, qz], dim=-1)

    def quat_unique(q: torch.Tensor) -> torch.Tensor:
        return torch.where(q[..., :1] < 0, -q, q)


if TYPE_CHECKING:
    import viser
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class UniformPoseCommand(CommandTerm):
    """Command generator for uniform pose tracking.

    Generates pose commands (position + orientation) in the robot's base frame.
    Positions are sampled uniformly from specified ranges, orientations are sampled
    uniformly in euler angles and converted to quaternions.
    """

    cfg: UniformPoseCommandCfg

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        # 获取机器人和目标 body
        self.robot: Entity = env.scene[cfg.entity_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # 命令缓冲区：在基座坐标系下的期望位姿 [x, y, z, qw, qx, qy, qz]
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0  # 单位四元数

        # 世界坐标系下的命令位姿（用于度量）
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        # 度量
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        # GUI 相关
        self._gui_enabled: viser.GuiCheckboxHandle | None = None
        self._gui_sliders: list[viser.GuiSliderHandle] = []
        self._gui_get_env_idx: Callable[[], int] | None = None

    @property
    def command(self) -> torch.Tensor:
        """Desired pose command in base frame. Shape (num_envs, 7)."""
        return self.pose_command_b

    def _update_metrics(self) -> None:
        """Compute position and orientation errors in world frame."""
        # 获取基座位姿
        root_pos = self.robot.data.root_link_pos_w
        root_quat = self.robot.data.root_link_quat_w

        # 局部命令
        local_pos = self.pose_command_b[:, :3]
        local_quat = self.pose_command_b[:, 3:]

        # 转换到世界坐标系
        rot_mat = matrix_from_quat(root_quat)  # (num_envs, 3, 3)
        world_pos = root_pos + torch.einsum('nij,nj->ni', rot_mat, local_pos)
        world_quat = quat_mul(root_quat, local_quat)
        self.pose_command_w = torch.cat([world_pos, world_quat], dim=-1)

        # 当前身体在世界坐标系下的位姿
        body_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]  # (num_envs, 7)
        body_pos = body_pose_w[:, :3]
        body_quat = body_pose_w[:, 3:7]

        # 位置误差
        pos_error = body_pos - world_pos
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)

        # 方向误差（旋转角度）
        q_rel = quat_mul(quat_conj(body_quat), world_quat)
        angle_error = 2 * torch.acos(torch.abs(q_rel[:, 0]))
        angle_error = torch.clamp(angle_error, max=torch.pi)
        self.metrics["orientation_error"] = angle_error

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Resample pose commands for specified environments."""
        if len(env_ids) == 0:
            return

        r = torch.empty(len(env_ids), device=self.device)

        # 采样位置
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

        # 采样欧拉角
        euler = torch.zeros(len(env_ids), 3, device=self.device)
        euler[:, 0] = r.uniform_(*self.cfg.ranges.roll)
        euler[:, 1] = r.uniform_(*self.cfg.ranges.pitch)
        euler[:, 2] = r.uniform_(*self.cfg.ranges.yaw)

        # 转换为四元数
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        if self.cfg.make_quat_unique:
            quat = quat_unique(quat)
        self.pose_command_b[env_ids, 3:] = quat

    def _update_command(self) -> None:
        """No continuous update for pose commands."""
        pass

    # GUI 实时控制（与 velocity_command 一致）
    def create_gui(
        self,
        name: str,
        server: "viser.ViserServer",
        get_env_idx: Callable[[], int],
    ) -> None:
        """Create GUI sliders to manually set pose command for a selected environment."""
        from viser import Icon

        with server.gui.add_folder(name.capitalize()):
            enabled = server.gui.add_checkbox("Enable", initial_value=False)

            pos_x = server.gui.add_slider(
                "pos_x", min=self.cfg.ranges.pos_x[0], max=self.cfg.ranges.pos_x[1],
                step=0.01, initial_value=0.0
            )
            pos_y = server.gui.add_slider(
                "pos_y", min=self.cfg.ranges.pos_y[0], max=self.cfg.ranges.pos_y[1],
                step=0.01, initial_value=0.0
            )
            pos_z = server.gui.add_slider(
                "pos_z", min=self.cfg.ranges.pos_z[0], max=self.cfg.ranges.pos_z[1],
                step=0.01, initial_value=0.0
            )
            roll = server.gui.add_slider(
                "roll", min=self.cfg.ranges.roll[0], max=self.cfg.ranges.roll[1],
                step=0.01, initial_value=0.0
            )
            pitch = server.gui.add_slider(
                "pitch", min=self.cfg.ranges.pitch[0], max=self.cfg.ranges.pitch[1],
                step=0.01, initial_value=0.0
            )
            yaw = server.gui.add_slider(
                "yaw", min=self.cfg.ranges.yaw[0], max=self.cfg.ranges.yaw[1],
                step=0.01, initial_value=0.0
            )

            sliders = [pos_x, pos_y, pos_z, roll, pitch, yaw]

            zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)
            @zero_btn.on_click
            def _(_) -> None:
                for s in sliders:
                    s.value = 0.0

        self._gui_enabled = enabled
        self._gui_sliders = sliders
        self._gui_get_env_idx = get_env_idx

    def compute(self, dt: float) -> None:
        """Override command if GUI enabled."""
        super().compute(dt)
        if self._gui_enabled is not None and self._gui_enabled.value:
            assert self._gui_get_env_idx is not None
            idx = self._gui_get_env_idx()

            pos = torch.tensor([s.value for s in self._gui_sliders[:3]], device=self.device)
            euler = torch.tensor([s.value for s in self._gui_sliders[3:6]], device=self.device)

            quat = quat_from_euler_xyz(euler[0], euler[1], euler[2])
            if self.cfg.make_quat_unique:
                quat = quat_unique(quat)

            self.pose_command_b[idx, :3] = pos
            self.pose_command_b[idx, 3:] = quat


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
    """Configuration for UniformPoseCommand."""

    entity_name: str
    """Name of the robot entity in the scene."""

    body_name: str
    """Name of the body (e.g., end-effector) to track."""

    @dataclass
    class Ranges:
        """Sampling ranges for position and euler angles."""
        pos_x: tuple[float, float]
        pos_y: tuple[float, float]
        pos_z: tuple[float, float]
        roll: tuple[float, float]
        pitch: tuple[float, float]
        yaw: tuple[float, float]

    ranges: Ranges
    """Ranges for uniform sampling."""

    make_quat_unique: bool = True
    """Ensure quaternion has positive real part (unique representation)."""

    def build(self, env: ManagerBasedRlEnv) -> UniformPoseCommand:
        return UniformPoseCommand(self, env)