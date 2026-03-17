import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.models import MLPModel

# 保持原来的 StateHistoryEncoder 不变
class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        # 计算卷积输出维度
        if tsteps == 50:
            conv_output_size = channel_size * 3
        elif tsteps == 10:
            conv_output_size = channel_size * 3
        elif tsteps == 20:
            conv_output_size = channel_size * 3

        self.linear_output = nn.Sequential(
                nn.Linear(conv_output_size, output_size), self.activation_fn
                )


    def forward(self, obs):
        # obs shape: [batch, tsteps, input_size]
        batch_size = obs.shape[0]
        T = self.tsteps
        
        # 编码每个时间步
        projection = self.encoder(obs.reshape([batch_size * T, -1]))  # [batch*T, 3*channel_size]
        
        # 重排为 [batch, channels, T] 用于卷积
        output = self.conv_layers(projection.reshape([batch_size, T, -1]).permute((0, 2, 1)))
        
        # 最终线性层
        output = self.linear_output(output)
        return output


class ActorWithEncoders(MLPModel):
    """用于 actor 的模型，包含历史编码器、特权编码器及腿臂分离输出头。"""
    is_recurrent = False  

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,  # 应为 "actor"
        output_dim: int,
        # 原有参数
        num_prop: int,
        num_priv: int,
        num_history: int,
        actor_hidden_dims=[256, 256, 256],
        leg_control_head_hidden_dims=[128, 64],
        arm_control_head_hidden_dims=[128, 64],
        priv_encoder_dims=[64, 18],
        activation="elu",
        activation_out="tanh",
        # MLPModel 需要的参数
        hidden_dims=None,          # 会被 actor_hidden_dims 覆盖
        obs_normalization=False,
        distribution_cfg=None,
        **kwargs,
    ):       
        
        # 调用父类初始化，但会先解析 obs_groups 和 obs_set
        # 注意：父类会创建 self.obs_groups 和 self.obs_dim，但我们需要自己管理 latent 构建
        # 因此可以暂时不依赖父类的 MLP 构建，而是在 __init__ 最后重新定义 self.mlp
        super().__init__(obs, obs_groups, obs_set, output_dim,
                         hidden_dims=actor_hidden_dims,  # 用 actor_hidden_dims 覆盖
                         activation=activation,
                         obs_normalization=obs_normalization,
                         distribution_cfg=distribution_cfg)
        
        
        self.num_leg_actions = kwargs.get('num_leg_actions', output_dim // 2)
        self.num_arm_actions = output_dim - self.num_leg_actions
        self.num_prop = num_prop
        self.num_priv = num_priv
        self.num_history = num_history
        self.activation_out_fn = resolve_nn_activation(activation_out)
        self._encoding_mode = 'priv'


        # 从 obs_groups 中获取需要编码的组
        # 假设配置中已经将 prop 观测组命名为 "proprioceptive"，特权组为 "privileged"
        # 这里我们直接使用传入的 num_prop 等，但更鲁棒的做法是从 obs 中解析维度
        # 为简化，我们沿用你的手动维度定义

        # 构建历史编码器
        activation_fn = resolve_nn_activation(activation)
        self.history_encoder = StateHistoryEncoder(
            activation_fn=activation_fn,
            input_size=num_prop,
            tsteps=num_history,
            output_size=priv_encoder_dims[-1] if priv_encoder_dims else num_priv
        )

        # 构建特权编码器
        if priv_encoder_dims:
            priv_layers = []
            priv_layers.append(nn.Linear(num_priv, priv_encoder_dims[0]))
            priv_layers.append(activation_fn)
            for i in range(len(priv_encoder_dims)-1):
                priv_layers.append(nn.Linear(priv_encoder_dims[i], priv_encoder_dims[i+1]))
                priv_layers.append(activation_fn)
            self.priv_encoder = nn.Sequential(*priv_layers)
            priv_encoded_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoded_dim = num_priv
        self.priv_encoded_dim = priv_encoded_dim

        # 重新计算 MLP 的输入维度：当前 prop + 编码后的特征
        # 当前 prop 维度 = num_prop（因为历史部分已经单独处理）
        mlp_input_dim = num_prop + priv_encoded_dim

        # 如果 distribution_cfg 存在，需要确定 mlp_output_dim
        if self.distribution is not None:
            mlp_output_dim = self.distribution.input_dim
        else:
            mlp_output_dim = output_dim

        # 重新构建 MLP（替换父类自动创建的 self.mlp）
        self.mlp = MLP(mlp_input_dim, mlp_output_dim, actor_hidden_dims, activation)

        # 初始化分布权重
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

        # 构建腿、臂输出头
        backbone_output_dim = actor_hidden_dims[-1] if actor_hidden_dims else mlp_input_dim
        # 腿部头
        leg_layers = []
        prev_dim = backbone_output_dim
        for i, dim in enumerate(leg_control_head_hidden_dims):
            leg_layers.append(nn.Linear(prev_dim, dim))
            if i < len(leg_control_head_hidden_dims)-1:
                leg_layers.append(activation_fn)
            else:
                leg_layers.append(self.activation_out_fn)
            prev_dim = dim
        leg_layers.append(nn.Linear(prev_dim, self.num_leg_actions))  # num_leg_actions 需要定义
        self.actor_leg_head = nn.Sequential(*leg_layers)

        # 臂部头
        arm_layers = []
        prev_dim = backbone_output_dim
        for i, dim in enumerate(arm_control_head_hidden_dims):
            arm_layers.append(nn.Linear(prev_dim, dim))
            if i < len(arm_control_head_hidden_dims)-1:
                arm_layers.append(activation_fn)
            else:
                arm_layers.append(self.activation_out_fn)
            prev_dim = dim
        arm_layers.append(nn.Linear(prev_dim, self.num_arm_actions))
        self.actor_arm_head = nn.Sequential(*arm_layers)

        # 用于历史编码器的隐藏状态（LSTM 会自己维护，这里用变量暂存）
        self._hidden_state = None

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None):
        # prop_obs 是通过拼接 obs_groups 中除 "privileged" 外的所有组得到的，这可能包含其他信息（如命令）。如果这些额外信息的维度不在历史中，需要调整切片逻辑。目前您的设计是假设所有非特权观测都是历史 proprio，这可能过于简化。建议在注释中明确说明此假设，或根据实际观测组动态确定当前 proprio 的位置。
        prop_obs = torch.cat([obs[group] for group in self.obs_groups if group != "privileged"], dim=-1)
        priv_obs = obs.get("privileged", None)

        # 历史编码
        hist_part = prop_obs[:, :self.num_prop * self.num_history]
        hist_part = hist_part.view(-1, self.num_history, self.num_prop)
        hist_feat = self.history_encoder(hist_part)

        # 特权编码（如果存在）
        if priv_obs is not None:
            priv_feat = self.priv_encoder(priv_obs)
        else:
            priv_feat = torch.zeros(prop_obs.shape[0], self.priv_encoded_dim, device=prop_obs.device)

        current_prop = prop_obs[:, -self.num_prop:]


        if self._encoding_mode == 'hist':
            latent = torch.cat([current_prop, hist_feat], dim=-1)
        elif self._encoding_mode == 'priv':
            latent = torch.cat([current_prop, priv_feat], dim=-1)
        else:
            raise ValueError(f"Unknown encoding mode: {self._encoding_mode}")

        if self.obs_normalization:
            latent = self.obs_normalizer(latent)
        return latent

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        # 调用 get_latent 获取输入特征
        latent = self.get_latent(obs, masks, hidden_state)
        # MLP 前向
        mlp_out = self.mlp(latent)
        # 分离头
        leg_out = self.actor_leg_head(mlp_out)
        arm_out = self.actor_arm_head(mlp_out)
        actions = torch.cat([leg_out, arm_out], dim=-1)

        # 如果使用分布，则处理随机输出
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(actions)
                return self.distribution.sample()
            else:
                return self.distribution.deterministic_output(actions)
        return actions

    def infer_priv_latent(self, obs: TensorDict):
            priv_obs = obs.get("privileged")
            if priv_obs is None:
                raise ValueError("Privileged observations not found")
            return self.priv_encoder(priv_obs)

    def infer_hist_latent(self, obs: TensorDict):
        prop_obs = torch.cat([obs[group] for group in self.obs_groups if group != "privileged"], dim=-1)
        hist_part = prop_obs[:, :self.num_prop * self.num_history]
        hist_part = hist_part.view(-1, self.num_history, self.num_prop)
        return self.history_encoder(hist_part)

    def reset(self, dones=None, hidden_state=None):
        # """重置历史编码器的隐藏状态（当环境 done 时）"""
        # if hasattr(self, '_hidden_state') and self._hidden_state is not None and dones is not None:
        #     h, c = self._hidden_state
        #     dones = dones.to(h.dtype).unsqueeze(0)  # [1, batch]
        #     h = h * (1.0 - dones)
        #     c = c * (1.0 - dones)
        #     self._hidden_state = (h, c)
        pass

    def get_hidden_state(self):
        # """返回当前隐藏状态，供存储使用"""
        # return self._hidden_state
        return None

    def detach_hidden_state(self, dones=None):
        # """截断反向传播"""
        # if hasattr(self, '_hidden_state') and self._hidden_state is not None:
        #     h, c = self._hidden_state
        #     self._hidden_state = (h.detach(), c.detach())
        pass

    @property
    def output_mean(self):
        return self.distribution.mean

    @property
    def output_std(self):
        return self.distribution.stddev

    @property
    def output_entropy(self):
        return self.distribution.entropy()  # [batch, num_actions]

    @property
    def output_distribution_params(self):
        return self.distribution.params

    def get_output_log_prob(self, outputs):
        return self.distribution.log_prob(outputs)  # [batch, num_actions]

    def get_kl_divergence(self, old_params, new_params):
        return self.distribution.kl_divergence(old_params, new_params)
    
    def set_encoding_mode(self, mode: str):
        self._encoding_mode = mode   # 'hist' 或 'priv'

class CriticWithEncoders(MLPModel):
    """用于 critic 的模型，可以使用特权信息，输出腿臂综合价值。"""
    is_recurrent = False 

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,  # 应为 "critic"
        output_dim: int,  # 应该为 1
        # 原有参数
        num_prop: int,
        num_priv: int,
        critic_hidden_dims=[256, 256, 256],
        critic_leg_control_head_hidden_dims=[128, 64],
        critic_arm_control_head_hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=False,
        **kwargs,
    ):
        # 调用父类初始化（无分布）
        super().__init__(obs, obs_groups, obs_set, output_dim,
                         hidden_dims=critic_hidden_dims,
                         activation=activation,
                         obs_normalization=obs_normalization,
                         distribution_cfg=None)

        self.num_prop = num_prop
        self.num_priv = num_priv
        activation_fn = resolve_nn_activation(activation)

        # MLP 输入维度：prop + priv（原始特权信息，不编码）
        mlp_input_dim = num_prop + num_priv

        backbone_output_dim = critic_hidden_dims[-1] if critic_hidden_dims else mlp_input_dim
        self.mlp = MLP(mlp_input_dim, backbone_output_dim, critic_hidden_dims, activation)

        # 腿部价值头
        leg_layers = []
        prev_dim = backbone_output_dim
        for i, dim in enumerate(critic_leg_control_head_hidden_dims):
            leg_layers.append(nn.Linear(prev_dim, dim))
            leg_layers.append(activation_fn)
            prev_dim = dim
        leg_layers.append(nn.Linear(prev_dim, 1))
        self.critic_leg_head = nn.Sequential(*leg_layers)

        # 臂部价值头
        arm_layers = []
        prev_dim = backbone_output_dim
        for i, dim in enumerate(critic_arm_control_head_hidden_dims):
            arm_layers.append(nn.Linear(prev_dim, dim))
            arm_layers.append(activation_fn)
            prev_dim = dim
        arm_layers.append(nn.Linear(prev_dim, 1))
        self.critic_arm_head = nn.Sequential(*arm_layers)

    def get_latent(self, obs, masks=None, hidden_state=None):
        """构建 critic 的输入特征：prop + priv"""
        # 从 obs 中提取 prop 和 priv（根据 obs_groups）
        # 假设配置中 prop 组为 "proprioceptive"，priv 组为 "privileged"
        prop_obs = torch.cat([obs[group] for group in self.obs_groups if group != "privileged"], dim=-1)
        priv_obs = obs.get("privileged", torch.zeros(prop_obs.shape[0], self.num_priv, device=prop_obs.device))
        # 取当前 prop（假设最后 num_prop 维）
        current_prop = prop_obs[:, -self.num_prop:]
        latent = torch.cat([current_prop, priv_obs], dim=-1)

        if self.obs_normalization:
            latent = self.obs_normalizer(latent)
        return latent

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_out = self.mlp(latent)
        leg_value = self.critic_leg_head(mlp_out)
        arm_value = self.critic_arm_head(mlp_out)
        # 将腿和臂的价值相加作为总价值（或可设计为平均/加权和）
        return torch.cat([leg_value, arm_value], dim=-1)  # [batch, 2]
        