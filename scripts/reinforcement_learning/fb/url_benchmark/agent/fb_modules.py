# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb  # pylint: disable=unused-import
import math
import typing as tp

import torch
from torch import nn
import torch.nn.functional as F
from url_benchmark import utils
from copy import deepcopy


class OnlineCov(nn.Module):
    def __init__(self, mom: float, dim: int) -> None:
        super().__init__()
        self.mom = mom  # momentum
        self.count = torch.nn.Parameter(torch.LongTensor([0]), requires_grad=False)
        self.cov: tp.Any = torch.nn.Parameter(torch.zeros((dim, dim), dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.count += 1  # type: ignore
            self.cov.data *= self.mom
            self.cov.data += (1 - self.mom) * torch.matmul(x.T, x) / x.shape[0]
        count = self.count.item()
        cov = self.cov / (1 - self.mom**count)
        return cov


class _L2(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = math.sqrt(self.dim) * F.normalize(x, dim=1)
        return y


def _nl(name: str, dim: int) -> tp.List[nn.Module]:
    """Returns a non-linearity given name and dimension"""
    if name == "irelu":
        return [nn.ReLU(inplace=True)]
    if name == "relu":
        return [nn.ReLU()]
    if name == "ntanh":
        return [nn.LayerNorm(dim), nn.Tanh()]
    if name == "layernorm":
        return [nn.LayerNorm(dim)]
    if name == "tanh":
        return [nn.Tanh()]
    if name == "L2":
        return [_L2(dim)]
    raise ValueError(f"Unknown non-linearity {name}")


def mlp(*layers: tp.Sequence[tp.Union[int, str]]) -> nn.Sequential:
    """Provides a sequence of linear layers and non-linearities
    providing a sequence of dimension for the neurons, or name of
    the non-linearities
    Eg: mlp(10, 12, "relu", 15) returns:
    Sequential(Linear(10, 12), ReLU(), Linear(12, 15))
    """
    assert len(layers) >= 2
    sequence: tp.List[nn.Module] = []
    assert isinstance(layers[0], int), "First input must provide the dimension"
    prev_dim: int = layers[0]
    for layer in layers[1:]:
        if isinstance(layer, str):
            sequence.extend(_nl(layer, prev_dim))
        else:
            assert isinstance(layer, int)
            sequence.append(nn.Linear(prev_dim, layer))
            prev_dim = layer
    return nn.Sequential(*sequence)


class Actor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess

        if self.preprocess:
            self.obs_net = mlp(self.obs_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh",
                             hidden_dim, "irelu",
                             hidden_dim, "irelu")
            feature_dim = hidden_dim

        self.policy = mlp(feature_dim, hidden_dim, "irelu", self.action_dim)
        self.apply(utils.weight_init)
        # initialize the last layer by zero
        # self.policy[-1].weight.data.fill_(0.0)

    def forward(self, obs, z, std):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            obs = self.obs_net(obs)
            h = torch.cat([obs, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class DiagGaussianActor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, hidden_dim, log_std_bounds,
                 preprocess=False) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.log_std_bounds = log_std_bounds
        self.preprocess = preprocess
        feature_dim = obs_dim + z_dim

        self.policy = mlp(feature_dim, hidden_dim, "ntanh", hidden_dim, "relu", 2 * action_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, z):
        assert z.shape[-1] == self.z_dim
        h = torch.cat([obs, z], dim=-1)
        mu, log_std = self.policy(h).chunk(2, dim=-1)
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        dist = utils.SquashedNormal(mu, std)
        return dist


class HighLevelActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim) -> None:
        super().__init__()
        self.policy = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", action_dim, "tanh")
        self.apply(utils.weight_init)

    def forward(self, obs, std=1.):
        mu = self.policy(obs)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


class EnsembleMLP(nn.Module):
    # internal model should only have init and forward meths.

    def __init__(self, f_dict, n_ensemble, device='cuda'):
        super().__init__()
        # needs to be a nn.module list otw we cannot do ensemble.state_dict or optimzie over its params!
        ensemble = nn.ModuleList([ForwardMap(**f_dict).to(device) for _ in range(n_ensemble)])
        # let’s combine the states of the model together by stacking each
        # parameter. For example, ``model[i].fc1.weight`` has shape ``[784, 128]``; we are
        # going to stack the ``.fc1.weight`` of each of the 10 models to produce a big
        # weight of shape ``[10, 784, 128]``.
        # PyTorch offers the ``torch.func.stack_module_state`` convenience function to do
        # this: # --> go from list of dicts to dict of lists
        #  stacked parameters are optimizable (i.e. they are new leaf nodes in the
        # autograd history that are UNRELATED to the original parameters and can be passed
        # directly to an optimizer).
        # buffers accounts for all non_trainable_params, we wont need it
        self.ensemble_params, buffers = torch.func.stack_module_state(ensemble)
        # Construct a "stateless" version of one of the models. It is "stateless" in
        # the sense that the parameters are meta Tensors and do not have storage, we do this by to."meta"
        # we also assign base_model as tuple  to avoid copying the parameters (avoid registration), otw, EnsembleMLP
        # object, will also have self.base_model params, additionally to the self.ensemble_params above.
        
        # TODO: Didnt' we need base_model to be a tuple?
        # self.base_model = (deepcopy(ensemble[0]).to("meta"),)  # used as a fct
        
        base_model = deepcopy(ensemble[0])
        self.base_model = base_model.to('meta')
        # self.device = base_model.device
        # self.to(self.device)

    # # IMPORTANT! model.parameters() of an nn.Module class calls named_parameters we need to override it
    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        # need to override named_parameters st when we pass the parameters to the optimizer,
        #  we will pass all the ensemble_params
        return self.ensemble_params.items()

    # @torch.compile()
    def forward(self, x: tuple):  # x =(obs: torch.tensor,  z: torch.tensor, action: torch.tensor)
        """
        Expects inputs obs, z, and action to have shape (ensemble_size, B, feature_dim),
        where ensemble_size is the number of ensemble members, B is the batch size, and
        feature_dim is the input dimensionality of each component.
        Returns a tuple of outputs (F1, F2) from all ensemble members.
        """
        def fmodel(params, buffers, x):
            return torch.func.functional_call(self.base_model, (params, buffers), (x,))


        # vmap(func) returns a new function that maps func over some dimensions of the inputs.
        # in this case func is fmodel, that has as inputs (params, buffers, x).
        # so we want to map over params (which are each of the ensemble params), buffers is empty, and we don't want to map
        # over x (unless we want different x for different ensemble members) hence:  in_dims = (0,0, None)
        # By using ``None``, we tell ``vmap`` we want the same minibatch to apply for all of
        # the num_ensemble models        
        ensemble_out = torch.vmap(fmodel, in_dims=(0, 0, None))(self.ensemble_params, {}, x)

        return ensemble_out


class ForwardMap(nn.Module):
    """ forward representation class"""

    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess

        if self.preprocess:
            self.obs_action_net = mlp(self.obs_dim + self.action_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim + self.action_dim, hidden_dim, "ntanh",
                             hidden_dim, "irelu",
                             hidden_dim, "irelu")
            feature_dim = hidden_dim

        seq = [feature_dim, hidden_dim, "irelu", self.z_dim]
        self.F1 = mlp(*seq)
        self.F2 = mlp(*seq)

        self.apply(utils.weight_init)

    def forward(self, x: tuple):  #obs, z, action)
        assert isinstance(x, tuple), "x must be a tuple: (obs, z, action)"
        obs, z, action = x
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_action = self.obs_action_net(torch.cat([obs, action], dim=-1))
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            h = torch.cat([obs_action, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z, action], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        F1 = self.F1(h)
        F2 = self.F2(h)
        return F1, F2


class IdentityMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.B = nn.Identity()

    def forward(self, obs):
        return self.B(obs)


class BackwardMap(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(self.obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        B = self.B(obs)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B


class MultinputNet(nn.Module):
    """Network with multiple inputs"""

    def __init__(self, input_dims: tp.Sequence[int], sequence_dims: tp.Sequence[int]) -> None:
        super().__init__()
        input_dims = list(input_dims)
        sequence_dims = list(sequence_dims)
        dim0 = sequence_dims[0]
        self.innets = nn.ModuleList([mlp(indim, dim0, "relu", dim0, "layernorm") for indim in input_dims])  # type: ignore
        sequence: tp.List[tp.Union[str, int]] = [dim0]
        for dim in sequence_dims[1:]:
            sequence.extend(["relu", dim])
        self.outnet = mlp(*sequence)  # type: ignore

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        assert len(tensors) == len(self.innets)
        out = sum(net(x) for net, x in zip(self.innets, tensors)) / len(self.innets)
        return self.outnet(out)  # type : ignore


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.Q = mlp(obs_dim + action_dim + self.z_dim, hidden_dim, "ntanh", hidden_dim, "relu", 1)
        self.apply(utils.weight_init)

    def forward(self, obs, action, z):
        h = torch.cat([obs, action, z], dim=-1)
        Q = self.Q(h)
        return Q


class RNDNN(nn.Module):
    def __init__(self, obs_dim, hidden_dim, out_dim):
        super().__init__()
        self.rnd = mlp(obs_dim, hidden_dim, 'relu', hidden_dim, 'relu', out_dim)
        self.apply(utils.weight_init)

    def forward(self, x):
        x = self.rnd(x)
        return x


class RNDCuriosity:
    def __init__(self, obs_dim, hidden_dim, out_dim, lr_rnd, device):
        self.rnd_pred = RNDNN(obs_dim, hidden_dim, out_dim).to(device)
        self.rnd_target = RNDNN(obs_dim, hidden_dim, out_dim).to(device)
        for param in self.rnd_target.parameters():
            param.requires_grad = False
        self.optimizer = torch.optim.SGD(self.rnd_pred.parameters(), lr=lr_rnd)

    def update_curiosity(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (n_env, n_obs)
        pred = self.rnd_pred(obs)
        target = self.rnd_target(obs)
        rew = torch.norm(pred - target, dim=-1)
        loss = torch.mean((pred - target)**2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return rew.detach(), loss.detach()
