# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import re
import time
import typing as tp

import numpy as np
import torch
import yaml
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from typing import Any, Dict
from pathlib import Path
import importlib

try:
    from typing import Protocol
except ImportError:
    # backward compatible
    from typing_extensions import Protocol  # type: ignore


class Trainable(Protocol):  # cannot from url_benchmark import agent
    @property
    def training(self) -> bool:
        ...

    def train(self, train: bool) -> None:
        ...


class eval_mode:
    def __init__(self, *models: Trainable) -> None:
        self.models = models
        self.prev_states: tp.List[bool] = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args: tp.Any) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


X = tp.TypeVar("X")


def chain(*iterables: tp.Iterable[X]) -> tp.Iterator[X]:  # TODO remove
    for it in iterables:
        yield from it


def soft_update_params(net, target_net, tau) -> None:
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data
                                + (1 - tau) * target_param.data)


# update target network to the MEAN of the ensemble network
def soft_update_params_mean(net, target_net, tau) -> None:
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        soft_mean_target = param.data.mean(0)
        target_param.data.copy_(tau * soft_mean_target
                                + (1 - tau) * target_param.data)


def hard_update_params(net, target_net) -> None:
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(param.data)


def to_torch(xs, device) -> tuple:
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m) -> None:
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            # if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            # if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def grad_norm(params, norm_type: float = 2.0):
    params = [p for p in params if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


def param_norm(params, norm_type: float = 2.0):
    total_norm = torch.norm(
        torch.stack([torch.norm(p.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


def _repr(obj: tp.Any) -> str:
    items = {x: y for x, y in obj.__dict__.items() if not x.startswith("_")}
    params = ", ".join(f"{x}={y!r}" for x, y in sorted(items.items()))
    return f"{obj.__class__.__name__}({params})"


class Until:
    def __init__(self, until: tp.Optional[int], action_repeat: int = 1) -> None:
        self.until = until
        self.action_repeat = action_repeat

    def __call__(self, step: int) -> bool:
        if self.until is None:
            return True
        until = self.until // self.action_repeat
        return step < until

    def __repr__(self) -> str:
        return _repr(self)


class Every:
    def __init__(self, every: tp.Optional[int], action_repeat: int = 1) -> None:
        self.every = every
        self.action_repeat = action_repeat

    def __call__(self, step: int) -> bool:
        if self.every is None:
            return False
        every = self.every // self.action_repeat
        if step % every == 0:
            return True
        return False

    def __repr__(self) -> str:
        return _repr(self)


class Timer:
    def __init__(self) -> None:
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self) -> tp.Tuple[float, float]:
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self) -> float:
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1) -> None:
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x) -> torch.Tensor:
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x) -> torch.Tensor:
        return x.tanh()

    def _inverse(self, y) -> torch.Tensor:
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y) -> torch.Tensor:
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale) -> None:
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


def schedule(schdl, step) -> float:
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class FloatStats:

    def __init__(self) -> None:
        self.min = np.inf
        self.max = -np.inf
        self.mean = 0.0
        self.count = 0

    def add(self, value: float) -> "FloatStats":
        self.min = min(value, self.min)
        self.max = max(value, self.max)
        self.count += 1
        self.mean = (self.count - 1) / self.count * self.mean + 1 / self.count * value
        return self


import os
from isaaclab.utils.string import callable_to_string


def class_to_dict(obj: object, ignore: list = []) -> dict[str, Any]:
    """Convert an object into dictionary recursively.

    Note:
        Ignores all names starting with "__" (i.e. built-in methods).

    Args:
        obj: An instance of a class to convert.

    Raises:
        ValueError: When input argument is not an object.

    Returns:
        Converted dictionary mapping.
    """
    # check that input data is class instance
    if not hasattr(obj, "__class__"):
        raise ValueError(f"Expected a class instance. Received: {type(obj)}.")
    # convert object to dictionary
    if isinstance(obj, dict):
        obj_dict = obj
    elif isinstance(obj, torch.Tensor):
        # We have to treat torch tensors specially because `torch.tensor.__dict__` returns an empty
        # dict, which would mean that a torch.tensor would be stored as an empty dict. Instead we
        # want to store it directly as the tensor.
        return obj
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    else:
        return obj

    # convert to dictionary
    data = dict()
    for key, value in obj_dict.items():
        # disregard builtin attributes
        if key.startswith("__"):
            continue
        if key in ignore:
            continue
        # check if attribute is callable -- function
        if callable(value):
            data[key] = callable_to_string(value)
        # check if attribute is a dictionary
        elif hasattr(value, "__dict__") or isinstance(value, dict):
            data[key] = class_to_dict(value)
        # check if attribute is a list or tuple
        elif isinstance(value, (list, tuple)):
            data[key] = type(value)([class_to_dict(v) for v in value])
        else:
            data[key] = value
    return data


def load_function(path: str):
    print(
        'in load function'
    )
    """Safely load a function from a 'module.submodule:func' string."""
    if ":" not in path:
        raise ValueError(f"Invalid function path: {path}")
    module_path, func_name = path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def update_class_from_dict(obj, data: Dict[str, Any]):
    for key, val in data.items():
        if isinstance(obj, dict):
            attr = obj.get(key, None)
        else:
            attr = getattr(obj, key, None)

        # Automatically load function if it's a path string and key is 'func'
        if key == "func" or "class_type" in key and isinstance(val, str) and ":" in val:
            func = load_function(val)
            setattr(obj, key, func)
            continue

        # Recursively update nested objects
        if isinstance(val, dict):
            update_class_from_dict(attr, val)
        else:
            if hasattr(obj, key):
                print('setting', key, val)
                setattr(obj, key, val)  # set entire dict if no sub-structure exists


def load_config(play_path, cfg, env_cfg):
    cwd = Path(__file__).parent.resolve().absolute()
    # This loads the play config:
    with open(os.path.join(cwd, 'configs', play_path + '.yaml',), 'r') as f:
        play_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # This is the path to the model to load config
    load_cfg = play_cfg['load_model'].split('models')[0] + 'config.yaml'

    def tuple_constructor(loader, node):
        # Load the sequence of values from the YAML node
        values = loader.construct_sequence(node)
        # Return a tuple constructed from the sequence
        return tuple(values)

    def safe_update(cfg, update_dict):
        # Get only keys that exist in the config
        filtered_update = {k: v for k, v in update_dict.items() if k in cfg}
        cfg.merge_with(filtered_update)

    # Deal with !!python:tuple in config
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)
    # Load the YAML file
    with open(load_cfg, 'r') as f:
        new_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # Update config with the config of the desired loaded model
    safe_update(cfg, new_cfg)
    update_class_from_dict(env_cfg, new_cfg)
    # Rewrite the load config with the play config
    safe_update(cfg, play_cfg)


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    """Saves data into a YAML file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save either a dictionary or class object.
        sort_keys: Whether to sort the keys in the output file. Defaults to False.
    """
    # check ending
    if not filename.endswith("yaml"):
        filename += ".yaml"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # convert data into dictionary
    if not isinstance(data, dict):
        data = class_to_dict(data)
    # save data
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)


from collections import defaultdict


def update_merged_dict(merged: defaultdict, new_output: dict) -> defaultdict:
    """Updates merged dictionary by concatenating new tensors iteratively."""
    for key, value in new_output.items():
        merged[key] = torch.cat((merged[key], value), dim=0) if key in merged else value.clone()
    return merged
