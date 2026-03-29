from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv

from . import local_manager


class ManagerRLEnv(ManagerBasedRlEnv):
  """Manager-based RL env with split locomotion / arm rewards."""

  def __init__(
    self,
    cfg,
    device: str,
    render_mode: str | None = None,
    **kwargs,
  ) -> None:
    super().__init__(cfg=cfg, device=device, render_mode=render_mode, **kwargs)
    self.leg_reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    self.arm_reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

  def load_managers(self):
    super().load_managers()
    self.reward_manager = local_manager.RewardManager(self.cfg.rewards, self)
    self.observation_manager = local_manager.ObservationManager(self.cfg.observations, self)
    self._configure_gym_env_spaces()

  def step(self, action: torch.Tensor):
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)

    self.episode_length_buf += 1
    self.common_step_counter += 1

    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    self.reward_buf, self.arm_reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.leg_reward_buf = self.reward_buf
    total_rewards = self.leg_reward_buf + self.arm_reward_buf
    self.metrics_manager.compute()

    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()

    self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)

    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)

    self.extras["leg_rewards"] = self.leg_reward_buf.clone()
    self.extras["arm_rewards"] = self.arm_reward_buf.clone()
    self.extras["total_rewards"] = total_rewards.clone()

    log = self.extras.setdefault("log", {})
    recompose_error = torch.abs(
      self.extras["total_rewards"] - (self.extras["leg_rewards"] + self.extras["arm_rewards"])
    )
    log["Metrics/reward_leg_mean"] = torch.mean(self.extras["leg_rewards"])
    log["Metrics/reward_arm_mean"] = torch.mean(self.extras["arm_rewards"])
    log["Metrics/reward_total_mean"] = torch.mean(self.extras["total_rewards"])
    log["Metrics/reward_recompose_error_mean"] = torch.mean(recompose_error)

    return (
      self.obs_buf,
      total_rewards,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )
