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

        # ===== 新增：世界坐标系下的 z 轴高度缓冲区 =====
        self.pose_command_w_z = torch.zeros(self.num_envs, 1, device=self.device)
        """World frame z-coordinate for B2Z1 pose commands."""

        # 度量
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        # 保存环境引用和训练步数相关参数
        self.env = env
        # self.num_env_step = 24  # 默认值

        if hasattr(self.cfg, 'curriculum_coeff') and self.cfg.curriculum_coeff is not None:
            # 直接使用基础配置
            from dwbc.rl_cfg import b2z1_ppo_runner_cfg
            cfg_runner = b2z1_ppo_runner_cfg()
            self.num_env_step = cfg_runner.num_steps_per_env  # 24
            # 先不考虑粗糙地形

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
       
        # B2Z1 特殊处理：如果启用，使用世界坐标系下的 z 高度
        if hasattr(self.cfg, 'is_B2Z1') and self.cfg.is_B2Z1:
            # 对于 B2Z1，使用预先存储的世界坐标系 z 高度
            world_pos[:, 2] = self.pose_command_w_z[:, 0]
            # 重新计算局部 z 坐标（用于后续可能的计算）
            self.pose_command_b[:, 2] = self.pose_command_w_z[:, 0] - root_pos[:, 2]

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

        # 初始化变量
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        r = torch.empty(len(env_ids), device=self.device)
        r_1 = torch.empty(1, device=self.device)

        # B2Z1 训练模式（带课程学习）
        if hasattr(self.cfg, 'is_B2Z1') and self.cfg.is_B2Z1:
            # 课程学习进度计算
            count = torch.tensor(
                self.env.common_step_counter / self.num_env_step / self.cfg.curriculum_coeff,
                device=self.device
            )
            count = torch.clamp(count, 0, 1)  # 确保在 [0,1] 范围内

            # 位置采样（课程学习）- x, y 在基座坐标系
            self.pose_command_b[env_ids, 0] = (
                r.uniform_(*self.cfg.ranges_init.pos_x) * (1 - count) +
                r.uniform_(*self.cfg.ranges_final.pos_x) * count
            )
            self.pose_command_b[env_ids, 1] = (
                r.uniform_(*self.cfg.ranges_init.pos_y) * (1 - count) +
                r.uniform_(*self.cfg.ranges_final.pos_y) * count
            )
            
            # z 坐标在世界坐标系采样，然后转换
            self.pose_command_w_z[env_ids, 0] = (
                r.uniform_(*self.cfg.ranges_init.pos_z) * (1 - count) +
                r.uniform_(*self.cfg.ranges_final.pos_z) * count
            )
            self.pose_command_b[env_ids, 2] = (
                self.pose_command_w_z[env_ids, 0] - self.robot.data.root_link_pos_w[env_ids, 2]
            )

            # 可达性检查和重采样
            for i_idx, env_id in enumerate(env_ids):
                # 计算臂长（末端到基座的距离）
                length_arm = torch.norm(torch.stack([
                    self.pose_command_b[env_id, 0],
                    self.pose_command_b[env_id, 1],
                    self.pose_command_b[env_id, 2]
                ]))
                

                # # 获取当前位姿
                # x = self.pose_command_b[env_id, 0]
                # y = self.pose_command_b[env_id, 1]
                # z = self.pose_command_w_z[env_id, 0]



                # 检查是否在可达范围内
                while ((length_arm > 0.7) or (length_arm < 0.3) or 
                       (self.pose_command_b[env_id, 0] < 0.45 and 
                        torch.abs(self.pose_command_b[env_id, 1]) < 0.2)):
                    

                # # 检查是否安全
                # while not self._is_pose_safe(x, y, z, length_arm):

                    # 重采样
                    self.pose_command_b[env_id, 0] = (
                        r_1.uniform_(*self.cfg.ranges_init.pos_x) * (1 - count) +
                        r_1.uniform_(*self.cfg.ranges_final.pos_x) * count
                    )
                    self.pose_command_b[env_id, 1] = (
                        r_1.uniform_(*self.cfg.ranges_init.pos_y) * (1 - count) +
                        r_1.uniform_(*self.cfg.ranges_final.pos_y) * count
                    )
                    self.pose_command_w_z[env_id, 0] = (
                        r_1.uniform_(*self.cfg.ranges_init.pos_z) * (1 - count) +
                        r_1.uniform_(*self.cfg.ranges_final.pos_z) * count
                    )
                    self.pose_command_b[env_id, 2] = (
                        self.pose_command_w_z[env_id, 0] - self.robot.data.root_link_pos_w[env_id, 2]
                    )
                    
                    # 更新参数
                    x = self.pose_command_b[env_id, 0]
                    y = self.pose_command_b[env_id, 1]
                    z = self.pose_command_w_z[env_id, 0]


                    # 重新计算臂长
                    length_arm = torch.norm(torch.stack([
                        self.pose_command_b[env_id, 0],
                        self.pose_command_b[env_id, 1],
                        self.pose_command_b[env_id, 2]
                    ]))

            # 横滚角采样（通常设为0或小范围）
            euler_angles[:, 0] = (
                r.uniform_(*self.cfg.ranges_init.roll) * (1 - count) +
                r.uniform_(*self.cfg.ranges_final.roll) * count
            )

            # 获取位置偏移量用于智能姿态计算
            delta_x = self.pose_command_b[env_ids, 0]
            delta_y = self.pose_command_b[env_ids, 1]
            delta_z = self.pose_command_b[env_ids, 2]

            # 俯仰角：基础角度（指向目标） + 随机扰动
            euler_angles[:, 1] = (
                -torch.atan2(delta_z, torch.sqrt(delta_x**2 + delta_y**2)) +
                r.uniform_(*self.cfg.ranges.pitch) * (1 - count) +
                r.uniform_(*self.cfg.ranges_final.pitch) * count
            )

            # 偏航角：基础角度（指向目标） + 随机扰动
            euler_angles[:, 2] = (
                torch.atan2(delta_y, delta_x) +
                r.uniform_(*self.cfg.ranges_init.yaw) * (1 - count) +
                r.uniform_(*self.cfg.ranges_final.yaw) * count
            )


        # B2Z1 测试/部署模式
        elif hasattr(self.cfg, 'is_B2Z1_Play') and self.cfg.is_B2Z1_Play:
            # 固定范围采样，无课程学习
            self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
            
            # z 坐标在世界坐标系采样
            self.pose_command_w_z[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_z)
            self.pose_command_b[env_ids, 2] = (
                self.pose_command_w_z[env_ids, 0] - self.robot.data.root_link_pos_w[env_ids, 2]
            )

            # 智能姿态计算
            delta_x = self.pose_command_b[env_ids, 0]
            delta_y = self.pose_command_b[env_ids, 1]
            delta_z = self.pose_command_b[env_ids, 2]

            euler_angles[:, 0] = r.uniform_(*self.cfg.ranges.roll)
            euler_angles[:, 1] = (
                -torch.atan2(delta_z, torch.sqrt(delta_x**2 + delta_y**2)) +
                r.uniform_(*self.cfg.ranges.pitch)
            )
            euler_angles[:, 2] = (
                torch.atan2(delta_y, delta_x) +
                r.uniform_(*self.cfg.ranges.yaw)
            )

        # 通用模式（完全随机采样）
        else:
            self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
            self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

            euler_angles[:, 0] = r.uniform_(*self.cfg.ranges.roll)
            euler_angles[:, 1] = r.uniform_(*self.cfg.ranges.pitch)
            euler_angles[:, 2] = r.uniform_(*self.cfg.ranges.yaw)

        # 将欧拉角转换为四元数
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        
        # 确保四元数唯一性（实部为正）
        if self.cfg.make_quat_unique:
            quat = quat_unique(quat)    
        self.pose_command_b[env_ids, 3:] = quat



    def _update_command(self) -> None:
        """No continuous update for pose commands."""
        pass

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg
    
    
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

    # ===== 新增 B2Z1 相关参数 =====
    
    is_B2Z1: bool = False
    """B2Z1 标志。
    启用 B2Z1 特有的课程学习和智能姿态计算。
    对应原代码的 is_Go2ARM。
    """
    
    is_B2Z1_Play: bool = False
    """B2Z1 测试模式标志。
    启用固定范围采样，无课程学习。
    对应原代码的 is_Go2ARM_Play。
    """
    
    is_B2Z1_Flat: bool = True
    """B2Z1 平坦地形标志。
    用于区分平坦/粗糙地形的训练配置。
    对应原代码的 is_Go2ARM_Flat。
    """
    
    curriculum_coeff: int = None
    """课程学习系数。
    控制从初始范围到最终范围的过渡速度。
    数值越大，过渡越慢。
    对应原代码的 curriculum_coeff。
    """
    
    ranges_init: Ranges = None
    """初始训练范围。
    训练初期使用的较小范围。
    对应原代码的 ranges_init。
    """
    
    ranges_final: Ranges = None
    """最终训练范围。
    训练后期使用的完整范围。
    对应原代码的 ranges_final。
    """

    def build(self, env: ManagerBasedRlEnv) -> UniformPoseCommand:
        return UniformPoseCommand(self, env)