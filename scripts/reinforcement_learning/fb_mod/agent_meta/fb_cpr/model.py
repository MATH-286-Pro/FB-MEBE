# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from ..fb.model import FBModel
from ..fb.model import Config as FBConfig
from ..fb.model import ArchiConfig as FBArchiConfig
from ..nn_models import build_forward, build_discriminator
from .. import config_from_dict
import torch
import copy


@dataclasses.dataclass
class CriticArchiConfig:
    hidden_dim: int = 1024
    model: str = "simple"  # {'simple', 'residual'}
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    ensemble_mode: str = "batch"  # {'batch', 'seq', 'vmap'}


@dataclasses.dataclass
class DiscriminatorArchiConfig:
    hidden_dim: int = 1024
    hidden_layers: int = 2


@dataclasses.dataclass
class ArchiConfig(FBArchiConfig):
    critic: CriticArchiConfig = dataclasses.field(default_factory=CriticArchiConfig)
    discriminator: DiscriminatorArchiConfig = dataclasses.field(default_factory=DiscriminatorArchiConfig)


@dataclasses.dataclass
class Config(FBConfig):
    archi: ArchiConfig = dataclasses.field(default_factory=ArchiConfig)


class FBcprModel(FBModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = config_from_dict(kwargs, Config)
        self._discriminator = build_discriminator(self.cfg.obs_dim, self.cfg.archi.z_dim, self.cfg.archi.discriminator)
        self._critic = build_forward(self.cfg.obs_dim, self.cfg.archi.z_dim, self.cfg.action_dim, self.cfg.archi.critic, output_dim=1)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.cfg.device)

    def _prepare_for_train(self) -> None:
        super()._prepare_for_train()
        self._target_critic = copy.deepcopy(self._critic)

    @torch.no_grad()
    def critic(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        return self._critic(self._normalize(obs), z, action)

    @torch.no_grad()
    def discriminator(self, obs: torch.Tensor, z: torch.Tensor):
        return self._discriminator(self._normalize(obs), z)
