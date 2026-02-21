# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import math
import dataclasses
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy
from pathlib import Path
from safetensors.torch import save_model as safetensors_save_model
import json

from ..nn_models import build_backward, build_forward, build_actor, eval_mode, build_critic
from .. import config_from_dict, load_model

########### user define config start ############
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from toolbox.dataclass_pylance import AGENT_CFG, MODEL_CFG, TRAIN_CFG
########### user define config end   ############

class FBModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = config_from_dict(kwargs, MODEL_CFG)

        policy_dim = self.cfg.policy_dim
        obs_dim    = self.cfg.obs_dim
        goal_dim   = self.cfg.goal_dim
        action_dim = self.cfg.action_dim
        critic_dim = self.cfg.critic_dim
        arch = self.cfg.archi

        # create networks
        self._forward_map  = build_forward(obs_dim, arch.z_dim, action_dim, arch.f)
        self._backward_map = build_backward(goal_dim, arch.z_dim, arch.b)
        self._actor        = build_actor(policy_dim, arch.z_dim, action_dim, arch.actor)

        self._policy_normalizer = nn.BatchNorm1d(policy_dim, affine=False, momentum=self.cfg.momentum) if self.cfg.norm_obs else nn.Identity()
        self._F_normalizer      = nn.BatchNorm1d(obs_dim,    affine=False, momentum=self.cfg.momentum) if self.cfg.norm_obs else nn.Identity()
        self._B_normalizer      = nn.BatchNorm1d(goal_dim,   affine=False, momentum=self.cfg.momentum) if self.cfg.norm_obs else nn.Identity()
        self._critic_normalizer = nn.BatchNorm1d(critic_dim, affine=False, momentum=self.cfg.momentum) if self.cfg.norm_obs else nn.Identity()

        if self.cfg.archi.critic.enable:
            self._critic        = build_critic(critic_dim, action_dim, arch.critic, output_dim=1)

    def _prepare_for_train(self) -> None:
        # create TARGET networks
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)
        if self.cfg.archi.critic.enable:
            self._target_critic = copy.deepcopy(self._critic)

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.cfg.device = device.type  # type: ignore
        return super().to(*args, **kwargs)

    @classmethod
    def load(cls, path: str, device: str | None = None):
        return load_model(path, device, cls=cls)

    def save(self, output_folder: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        safetensors_save_model(self, output_folder / "model.safetensors")
        with (output_folder / "config.json").open("w+") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)

    ######################## convinient tools ####################
    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z
    
    ######################## Network ####################
    def _policy_normalize(self, policy: torch.Tensor):
        with torch.no_grad(), eval_mode(self._policy_normalizer):
            return self._policy_normalizer(policy)

    def _F_normalize(self, obs: torch.Tensor):
        with torch.no_grad(), eval_mode(self._F_normalizer):
            return self._F_normalizer(obs)
    
    def _B_normalize(self, goal: torch.Tensor):
        with torch.no_grad(), eval_mode(self._B_normalizer):
            return self._B_normalizer(goal)
    
    def _critic_normalize(self, critic: torch.Tensor):
        with torch.no_grad(), eval_mode(self._critic_normalizer):
            return self._critic_normalizer(critic)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor):
        return self._backward_map(self._B_normalize(obs))

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        return self._forward_map(self._F_normalize(obs), z, action)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, std: float):
        return self._actor(self._policy_normalize(obs), z, std)

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean
        return dist.sample()
    
    ######################## Inference ####################
    def reward_inference(self, next_obs: torch.Tensor, reward: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        num_batches = int(np.ceil(next_obs.shape[0] / self.cfg.inference_batch_size))
        z = 0
        wr = reward if weight is None else reward * weight
        for i in range(num_batches):
            start_idx, end_idx = i * self.cfg.inference_batch_size, (i + 1) * self.cfg.inference_batch_size
            B = self.backward_map(next_obs[start_idx:end_idx].to(self.cfg.device))
            z += torch.matmul(wr[start_idx:end_idx].to(self.cfg.device).T, B)
        return self.project_z(z)

    def reward_wr_inference(self, next_obs: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        return self.reward_inference(next_obs, reward, F.softmax(10 * reward, dim=0))

    def goal_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
        z = self.backward_map(next_obs)
        return self.project_z(z)

    def tracking_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
        z = self.backward_map(next_obs)
        for step in range(z.shape[0]):
            end_idx = min(step + self.cfg.seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return self.project_z(z)
