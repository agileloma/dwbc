from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage


class MultiRewardRolloutStorage(RolloutStorage):
  """Rollout storage that supports vector rewards/values."""

  def __init__(
    self,
    training_type: str,
    num_envs: int,
    num_transitions_per_env: int,
    obs: TensorDict,
    actions_shape: tuple[int, ...] | list[int],
    reward_dim: int = 1,
    log_prob_dim: int = 1,
    device: str = "cpu",
  ) -> None:
    self.reward_dim = reward_dim
    self.log_prob_dim = log_prob_dim
    super().__init__(
      training_type=training_type,
      num_envs=num_envs,
      num_transitions_per_env=num_transitions_per_env,
      obs=obs,
      actions_shape=actions_shape,
      device=device,
    )

    self.rewards = torch.zeros(
      num_transitions_per_env, num_envs, reward_dim, device=self.device
    )
    if training_type == "rl":
      self.values = torch.zeros(
        num_transitions_per_env, num_envs, reward_dim, device=self.device
      )
      self.actions_log_prob = torch.zeros(
        num_transitions_per_env, num_envs, log_prob_dim, device=self.device
      )
      self.returns = torch.zeros(
        num_transitions_per_env, num_envs, reward_dim, device=self.device
      )
      self.advantages = torch.zeros(
        num_transitions_per_env, num_envs, reward_dim, device=self.device
      )

  def add_transition(self, transition: RolloutStorage.Transition) -> None:
    """Add one transition to the storage at the current step index."""
    if self.step >= self.num_transitions_per_env:
      raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

    self.observations[self.step].copy_(transition.observations)
    self.actions[self.step].copy_(transition.actions)  # type: ignore[arg-type]
    self.rewards[self.step].copy_(transition.rewards.view(-1, self.reward_dim))  # type: ignore[union-attr]
    self.dones[self.step].copy_(transition.dones.view(-1, 1))  # type: ignore[union-attr]

    if self.training_type == "distillation":
      self.privileged_actions[self.step].copy_(transition.privileged_actions)  # type: ignore[arg-type]

    if self.training_type == "rl":
      self.values[self.step].copy_(transition.values)  # type: ignore[arg-type]
      self.actions_log_prob[self.step].copy_(
        transition.actions_log_prob.view(-1, self.log_prob_dim)  # type: ignore[union-attr]
      )
      if self.distribution_params is None:
        self.distribution_params = tuple(
          torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
          for p in transition.distribution_params  # type: ignore[arg-type]
        )
      for i, p in enumerate(transition.distribution_params):  # type: ignore[arg-type]
        self.distribution_params[i][self.step].copy_(p)

    self._save_hidden_states(transition.hidden_states)
    self.step += 1
