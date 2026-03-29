from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class _SplitActionMLP(nn.Module):
  """Shared trunk with separate leg/arm control heads."""

  def __init__(
    self,
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int],
    activation: str,
    leg_action_dim: int,
    arm_action_dim: int,
  ) -> None:
    super().__init__()
    if leg_action_dim <= 0 or arm_action_dim <= 0:
      raise ValueError(
        f"leg_action_dim and arm_action_dim must be > 0, got {leg_action_dim}, {arm_action_dim}."
      )
    if leg_action_dim + arm_action_dim != output_dim:
      raise ValueError(
        f"leg_action_dim + arm_action_dim must equal output_dim, got "
        f"{leg_action_dim} + {arm_action_dim} != {output_dim}."
      )

    hidden_dims_list = list(hidden_dims)
    if len(hidden_dims_list) > 0:
      layers: list[nn.Module] = []
      in_dim = input_dim
      for h in hidden_dims_list:
        layers.append(nn.Linear(in_dim, h))
        layers.append(resolve_nn_activation(activation))
        in_dim = h
      self.backbone = nn.Sequential(*layers)
      head_in_dim = in_dim
    else:
      self.backbone = nn.Identity()
      head_in_dim = input_dim

    self.leg_control_head = nn.Linear(head_in_dim, leg_action_dim)
    self.arm_control_head = nn.Linear(head_in_dim, arm_action_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.backbone(x)
    leg_action = self.leg_control_head(features)
    arm_action = self.arm_control_head(features)
    return torch.cat((leg_action, arm_action), dim=-1)


class DualHeadActorModel(MLPModel):
  """Actor with shared trunk and explicit leg/arm action heads."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    leg_action_dim: int = 12,
    arm_action_dim: int = 6,
    history_obs_set: str = "history",
    privileged_obs_set: str = "privileged",
    encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
    encoder_latent_dim: int = 64,
  ) -> None:
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    self.history_obs_groups, self.history_obs_dim = self._get_optional_obs_dim(
      obs, obs_groups, history_obs_set
    )
    self.privileged_obs_groups, self.privileged_obs_dim = self._get_optional_obs_dim(
      obs, obs_groups, privileged_obs_set
    )
    self.use_privileged_history_regularization = (
      self.history_obs_groups is not None and self.privileged_obs_groups is not None
    )

    self.encoder_latent_dim = encoder_latent_dim
    if self.use_privileged_history_regularization:
      self.history_encoder = self._build_encoder(
        self.history_obs_dim, encoder_latent_dim, encoder_hidden_dims, activation
      )
      self.privileged_encoder = self._build_encoder(
        self.privileged_obs_dim, encoder_latent_dim, encoder_hidden_dims, activation
      )
    else:
      self.history_encoder = None
      self.privileged_encoder = None

    head_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
    actor_input_dim = self._get_latent_dim()
    if self.use_privileged_history_regularization:
      actor_input_dim += encoder_latent_dim

    self.mlp = _SplitActionMLP(
      input_dim=actor_input_dim,
      output_dim=head_output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      leg_action_dim=leg_action_dim,
      arm_action_dim=arm_action_dim,
    )

    self.leg_action_dim = leg_action_dim
    self.arm_action_dim = arm_action_dim

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
    latent = self.get_latent(obs, masks, hidden_state)

    if self.use_privileged_history_regularization:
      history_latent = self.infer_history_latent(obs)
      latent = torch.cat([latent, history_latent], dim=-1)

    mlp_output = self.mlp(latent)
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(mlp_output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(mlp_output)
    return mlp_output

  def infer_history_latent(self, obs: TensorDict) -> torch.Tensor:
    if not self.use_privileged_history_regularization or self.history_encoder is None:
      raise RuntimeError("History encoder is not enabled for this actor.")
    history_obs = self._concat_obs_groups(obs, self.history_obs_groups)
    return self.history_encoder(history_obs)

  def infer_privileged_latent(self, obs: TensorDict) -> torch.Tensor:
    if not self.use_privileged_history_regularization or self.privileged_encoder is None:
      raise RuntimeError("Privileged encoder is not enabled for this actor.")
    privileged_obs = self._concat_obs_groups(obs, self.privileged_obs_groups)
    return self.privileged_encoder(privileged_obs)

  def compute_privileged_history_reg_loss(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if not self.use_privileged_history_regularization:
      return torch.zeros((), device=obs[self.obs_groups[0]].device)

    obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
    history_latent = self.infer_history_latent(obs)
    privileged_latent = self.infer_privileged_latent(obs)
    return F.mse_loss(history_latent, privileged_latent.detach())

  def _concat_obs_groups(self, obs: TensorDict, obs_groups: list[str] | None) -> torch.Tensor:
    if obs_groups is None:
      raise RuntimeError("Observation groups are not initialized.")
    obs_list = [obs[obs_group] for obs_group in obs_groups]
    return torch.cat(obs_list, dim=-1)

  def _build_encoder(
    self,
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int],
    activation: str,
  ) -> nn.Module:
    hidden_dims_list = list(hidden_dims)
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims_list:
      layers.append(nn.Linear(in_dim, h))
      layers.append(resolve_nn_activation(activation))
      in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    layers.append(resolve_nn_activation(activation))
    return nn.Sequential(*layers)

  def _get_optional_obs_dim(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
  ) -> tuple[list[str] | None, int]:
    if obs_set not in obs_groups:
      return None, 0
    active_obs_groups = obs_groups[obs_set]
    obs_dim = 0
    for obs_group in active_obs_groups:
      if len(obs[obs_group].shape) != 2:
        raise ValueError(
          "DualHeadActorModel only supports 1D observations for optional sets, "
          f"got shape {obs[obs_group].shape} for '{obs_group}'."
        )
      obs_dim += obs[obs_group].shape[-1]
    return active_obs_groups, obs_dim


class DualHeadCriticModel(nn.Module):
  """Critic with a shared backbone and explicit per-dimension value heads."""

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
  ) -> None:
    super().__init__()

    if distribution_cfg is not None:
      raise ValueError("DualHeadCriticModel does not support distribution_cfg.")
    if output_dim < 1:
      raise ValueError(f"output_dim must be >= 1, got {output_dim}.")

    self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
    self.obs_normalization = obs_normalization
    if obs_normalization:
      self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
    else:
      self.obs_normalizer = torch.nn.Identity()

    hidden_dims_list = list(hidden_dims)
    if len(hidden_dims_list) > 0:
      layers: list[nn.Module] = []
      in_dim = self.obs_dim
      for h in hidden_dims_list:
        layers.append(nn.Linear(in_dim, h))
        layers.append(resolve_nn_activation(activation))
        in_dim = h
      self.backbone = nn.Sequential(*layers)
      head_in_dim = in_dim
    else:
      self.backbone = nn.Identity()
      head_in_dim = self.obs_dim

    self.output_dim = int(output_dim)
    self.value_heads = nn.ModuleList(
      [nn.Linear(head_in_dim, 1) for _ in range(self.output_dim)]
    )

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    del hidden_state, stochastic_output
    obs = unpad_trajectories(obs, masks) if masks is not None else obs
    latent = self.get_latent(obs)
    features = self.backbone(latent)
    values = [head(features) for head in self.value_heads]
    if len(values) == 1:
      return values[0]
    return torch.cat(values, dim=-1)

  def get_latent(self, obs: TensorDict) -> torch.Tensor:
    obs_list = [obs[obs_group] for obs_group in self.obs_groups]
    latent = torch.cat(obs_list, dim=-1)
    latent = self.obs_normalizer(latent)
    return latent

  def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
    del dones, hidden_state

  def get_hidden_state(self) -> HiddenState:
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    del dones

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      obs_list = [obs[obs_group] for obs_group in self.obs_groups]
      critic_obs = torch.cat(obs_list, dim=-1)
      self.obs_normalizer.update(critic_obs)  # type: ignore[union-attr]

  def _get_obs_dim(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
  ) -> tuple[list[str], int]:
    active_obs_groups = obs_groups[obs_set]
    obs_dim = 0
    for obs_group in active_obs_groups:
      if len(obs[obs_group].shape) != 2:
        raise ValueError(
          "DualHeadCriticModel only supports 1D observations, "
          f"got shape {obs[obs_group].shape} for '{obs_group}'."
        )
      obs_dim += obs[obs_group].shape[-1]
    return active_obs_groups, obs_dim
