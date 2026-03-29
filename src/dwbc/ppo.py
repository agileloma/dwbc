from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from dwbc.storage import MultiRewardRolloutStorage


class MixedPPO(PPO):
  """PPO with Isaac-Lab-style advantage mixing."""

  def __init__(
    self,
    *args,
    mixing_schedule: tuple[float, int, int] | list[float] | None = None,
    reward_dim: int = 1,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.mixing_schedule = tuple(mixing_schedule) if mixing_schedule is not None else None
    self.reward_dim = reward_dim
    self.counter = 0

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    """Record one environment step and keep locomotion/arm rewards separate."""
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)
    if self.rnd:
      self.rnd.update_normalization(obs)

    arm_rewards = extras.get("arm_rewards")
    if arm_rewards is None:
      arm_rewards = torch.zeros_like(rewards)
    else:
      arm_rewards = arm_rewards.to(self.device)

    # Keep scalar training behavior unchanged (total reward), but when reward_dim > 1
    # use explicit locomotion-vs-arm decomposition for two-headed value targets.
    leg_rewards = extras.get("leg_rewards")
    if leg_rewards is None:
      leg_rewards = rewards - arm_rewards
    else:
      leg_rewards = leg_rewards.to(self.device)

    if self.reward_dim > 1:
      reward_tensor = torch.stack([leg_rewards.clone(), arm_rewards.clone()], dim=-1)
    else:
      reward_tensor = rewards.clone().unsqueeze(-1)

    self.transition.rewards = reward_tensor
    self.transition.dones = dones

    if self.rnd:
      self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
      if self.reward_dim > 1:
        self.transition.rewards[..., 0] += self.intrinsic_rewards
      else:
        self.transition.rewards += self.intrinsic_rewards.unsqueeze(-1)

    if "time_outs" in extras:
      self.transition.rewards += self.gamma * (
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device)
      )

    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    self.critic.reset(dones)

  def update(self) -> dict[str, float]:
    """Run optimization epochs over stored batches and return mean losses."""
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_rnd_loss = 0.0 if self.rnd else None
    mean_symmetry_loss = 0.0 if self.symmetry else None
    mixing_ratio = self.get_value_mixing_ratio()

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for batch in generator:
      original_batch_size = batch.observations.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (
            batch.advantages - batch.advantages.mean()
          ) / (batch.advantages.std() + 1e-8)

      if self.symmetry and self.symmetry["use_data_augmentation"]:
        data_augmentation_func = self.symmetry["data_augmentation_func"]
        batch.observations, batch.actions = data_augmentation_func(
          env=self.symmetry["_env"],
          obs=batch.observations,
          actions=batch.actions,
        )
        num_aug = int(batch.observations.batch_size[0] / original_batch_size)
        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
        batch.values = batch.values.repeat(num_aug, 1)
        batch.advantages = batch.advantages.repeat(num_aug, 1)
        batch.returns = batch.returns.repeat(num_aug, 1)

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[1],
      )
      distribution_params = tuple(
        p[:original_batch_size] for p in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params,
            distribution_params,
          )
          kl_mean = torch.mean(kl)

          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size

          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)

          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()

          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      mixed_advantages = self._mix_advantages(batch.advantages, mixing_ratio)
      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      if mixed_advantages.ndim > ratio.ndim:
        ratio = ratio.unsqueeze(-1)
      surrogate = -mixed_advantages * ratio
      surrogate_clipped = -mixed_advantages * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

      if self.symmetry:
        if not self.symmetry["use_data_augmentation"]:
          data_augmentation_func = self.symmetry["data_augmentation_func"]
          batch.observations, _ = data_augmentation_func(
            obs=batch.observations, actions=None, env=self.symmetry["_env"]
          )

        mean_actions = self.actor(batch.observations.detach().clone())
        action_mean_orig = mean_actions[:original_batch_size]
        _, actions_mean_symm = data_augmentation_func(
          obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
        )
        mse_loss = torch.nn.MSELoss()
        symmetry_loss = mse_loss(
          mean_actions[original_batch_size:],
          actions_mean_symm.detach()[original_batch_size:],
        )
        if self.symmetry["use_mirror_loss"]:
          loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
        else:
          symmetry_loss = symmetry_loss.detach()

      if self.rnd:
        with torch.no_grad():
          rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
          rnd_state = self.rnd.state_normalizer(rnd_state)
        predicted_embedding = self.rnd.predictor(rnd_state)
        target_embedding = self.rnd.target(rnd_state).detach()
        mseloss = torch.nn.MSELoss()
        rnd_loss = mseloss(predicted_embedding, target_embedding)

      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.rnd_optimizer:
        self.rnd_optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates

    self.storage.clear()
    self.update_counter()

    loss_dict = {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "mixing_ratio": mixing_ratio,
    }
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss
    return loss_dict

  def save(self) -> dict:
    saved_dict = super().save()
    saved_dict["advantage_mixing_counter"] = self.counter
    return saved_dict

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    self.counter = loaded_dict.get("advantage_mixing_counter", 0)
    return load_iteration

  def update_counter(self) -> None:
    self.counter += 1

  def get_value_mixing_ratio(self) -> float:
    if self.mixing_schedule is None:
      return 0.0

    target_ratio, warmup_start, warmup_duration = self.mixing_schedule
    if warmup_duration <= 0:
      return float(target_ratio if self.counter >= warmup_start else 0.0)

    stage = min(max((self.counter - warmup_start) / warmup_duration, 0.0), 1.0)
    return float(stage * target_ratio)

  def _mix_advantages(
    self,
    advantages: torch.Tensor,
    mixing_ratio: float,
  ) -> torch.Tensor:
    if advantages.shape[-1] < 2 or mixing_ratio == 0.0:
      return advantages

    mixed_advantages = advantages.clone()
    mixed_advantages[..., 0] = advantages[..., 0] + mixing_ratio * advantages[..., 1]
    mixed_advantages[..., 1] = advantages[..., 1] + mixing_ratio * advantages[..., 0]
    return mixed_advantages

  @staticmethod
  def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "MixedPPO":
    """Construct PPO using a two-head critic when advantage mixing is enabled."""
    alg_class: type[MixedPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
    actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
    critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

    default_sets = ["actor", "critic"]
    if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
      default_sets.append("rnd_state")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

    reward_dim = 2 if cfg["algorithm"].get("mixing_schedule") is not None else 1

    actor: MLPModel = actor_class(
      obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
    ).to(device)
    print(f"Actor Model: {actor}")

    if cfg["algorithm"].pop("share_cnn_encoders", None):
      cfg["critic"]["cnns"] = actor.cnns  # type: ignore[attr-defined]

    critic: MLPModel = critic_class(
      obs, cfg["obs_groups"], "critic", reward_dim, **cfg["critic"]
    ).to(device)
    print(f"Critic Model: {critic}")

    storage = MultiRewardRolloutStorage(
      "rl",
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      [env.num_actions],
      reward_dim=reward_dim,
      device=device,
    )

    alg: MixedPPO = alg_class(
      actor,
      critic,
      storage,
      device=device,
      reward_dim=reward_dim,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg["multi_gpu"],
    )
    return alg
