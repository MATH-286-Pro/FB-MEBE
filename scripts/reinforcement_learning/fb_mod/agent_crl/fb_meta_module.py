import torch
from torch import nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
import numpy as np
import torch.nn.functional as F
import numbers
import math
from typing import Any


##########################
# Helper modules
##########################

class DenseParallel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_parallel: int,
        bias: bool = True,
        device=None,
        dtype=None,
        reset_params=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(DenseParallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        if n_parallel is None or (n_parallel == 1):
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.weight = nn.Parameter(
                torch.empty((n_parallel, in_features, out_features), **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty((n_parallel, 1, out_features), **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
            if self.bias is None:
                raise NotImplementedError
        if reset_params:
            self.reset_parameters()

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            weight_list = [m.weight.T for m in module_list]
            target_weight = torch.stack(weight_list, dim=0)
            self.weight.data.copy_(target_weight.data)
            if self.bias:
                bias_list = [ln.bias.unsqueeze(0) for ln in module_list]
                target_bias = torch.stack(bias_list, dim=0)
                self.bias.data.copy_(target_bias.data)

    # TODO why do these layers have their own reset scheme?
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.n_parallel is None or (self.n_parallel == 1):
            return F.linear(input, self.weight, self.bias)
        else:
            return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, n_parallel={}, bias={}".format(
            self.in_features, self.out_features, self.n_parallel, self.bias is not None
        )

class ParallelLayerNorm(nn.Module):
    def __init__(self, normalized_shape, n_parallel, eps=1e-5, elementwise_affine=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [normalized_shape, ]
        assert len(normalized_shape) == 1
        self.n_parallel = n_parallel
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            if n_parallel is None or (n_parallel == 1):
                self.weight = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
            else:
                self.weight = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            if self.elementwise_affine:
                ln_weights = [ln.weight.unsqueeze(0) for ln in module_list]
                ln_biases = [ln.bias.unsqueeze(0) for ln in module_list]
                target_ln_weights = torch.stack(ln_weights, dim=0)
                target_ln_bias = torch.stack(ln_biases, dim=0)
                self.weight.data.copy_(target_ln_weights.data)
                self.bias.data.copy_(target_ln_bias.data)


    def forward(self, input):
        norm_input = F.layer_norm(
            input, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias
        else:
            return norm_input

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.noise_upper_limit = high - self.loc
        self.noise_lower_limit = low - self.loc

    def _clamp(self, x) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

class Norm(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)

##########################
# Creation utils
##########################
def linear(input_dim, output_dim, num_parallel=1):
    if num_parallel > 1:
        return DenseParallel(input_dim, output_dim, n_parallel=num_parallel)  # 并联 多组神经网络
    return nn.Linear(input_dim, output_dim) 

def layernorm(input_dim, num_parallel=1):
    if num_parallel > 1:
        return ParallelLayerNorm([input_dim], n_parallel=num_parallel)        # 并联 多组神经网络
    return nn.LayerNorm(input_dim)


##########################
# Simple MLP models
##########################

def simple_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq = [linear(input_dim, hidden_dim, num_parallel),     # 神经网络 线性层
           layernorm(hidden_dim, num_parallel), nn.Tanh()]  # 神经网络 归一化层
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]

    seq += [linear(hidden_dim, hidden_dim // 2, num_parallel), nn.ReLU()]    # 输出维度 自动除2对齐 hidden_dim
    return nn.Sequential(*seq)


class ForwardMap(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, archi) -> None:
        super().__init__()

        hidden_dim       = archi.hidden_dim
        hidden_layers    = archi.hidden_layers
        embedding_layers = archi.embedding_layers
        num_parallel     = archi.num_parallel
        output_dim       = None

        self.z_dim = z_dim
        self.num_parallel = num_parallel
        self.hidden_dim = hidden_dim

        self.embed_z  = simple_embedding(obs_dim + z_dim,      hidden_dim, embedding_layers, num_parallel)    #FF0000 这个是什么
        self.embed_sa = simple_embedding(obs_dim + action_dim, hidden_dim, embedding_layers, num_parallel)    #FF0000 这个是什么

        seq = []
        for _ in range(hidden_layers):
            seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
        seq += [linear(hidden_dim, output_dim if output_dim else z_dim, num_parallel)]
        self.Fs = nn.Sequential(*seq)                                                                         #FF0000 这个是什么
    
    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)

        z_embedding  = self.embed_z(torch.cat([obs, z], dim=-1))        # num_parallel x bs x h_dim // 2
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))  # num_parallel x bs x h_dim // 2
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))  # 把 embed_z 和 embed_sa 的输出给到了 Fs
    

class BackwardMap(nn.Module):
    def __init__(self, goal_dim, z_dim, archi) -> None:
        super().__init__()

        hidden_dim       = archi.hidden_dim
        hidden_layers    = archi.hidden_layers
        norm             = archi.norm

        seq = [nn.Linear(goal_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers-1):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [nn.Linear(hidden_dim, z_dim)]
        if norm: 
            seq += [Norm()]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)



class Actor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, archi) -> None:
        super().__init__()

        hidden_dim       = archi.hidden_dim
        hidden_layers    = archi.hidden_layers
        embedding_layers = archi.embedding_layers

        self.embed_z = simple_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = simple_embedding(obs_dim, hidden_dim, embedding_layers)

        seq = []
        for _ in range(hidden_layers):
            seq += [linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [linear(hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs, z, std):
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1)) # bs x h_dim // 2
        s_embedding = self.embed_s(obs) # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist
