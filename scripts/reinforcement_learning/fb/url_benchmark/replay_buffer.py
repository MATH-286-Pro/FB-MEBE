# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import io
import random
import traceback
import typing as tp
from pathlib import Path
from collections import defaultdict
import dataclasses

import numpy as np
import torch
from torch.utils.data import IterableDataset
from dm_env import specs, TimeStep


EpisodeTuple = tp.Tuple[np.ndarray, ...]
Episode = tp.Dict[str, np.ndarray]
T = tp.TypeVar("T", np.ndarray, torch.Tensor)
B = tp.TypeVar("B", bound="EpisodeBatch")


@dataclasses.dataclass
class EpisodeBatch(tp.Generic[T]):
    """For later use
    A container for batchable replayed episodes
    """
    obs: T
    action: T
    reward: T
    next_obs: T
    discount: T
    meta: tp.Dict[str, T] = dataclasses.field(default_factory=dict)
    _physics: tp.Optional[T] = None
    goal: tp.Optional[T] = None
    next_goal: tp.Optional[T] = None
    future_obs: tp.Optional[T] = None
    future_goal: tp.Optional[T] = None

    def __post_init__(self) -> None:
        # some security to be removed later
        assert isinstance(self.reward, (np.ndarray, torch.Tensor))
        assert isinstance(self.discount, (np.ndarray, torch.Tensor))
        assert isinstance(self.meta, dict)

    def to(self, device: str) -> "EpisodeBatch[torch.Tensor]":
        """Creates a new instance on the appropriate device"""
        out: tp.Dict[str, tp.Any] = {}
        for field in dataclasses.fields(self):
            data = getattr(self, field.name)
            if field.name == "meta":
                out[field.name] = {x: torch.as_tensor(y, device=device) for x, y in data.items()}  # type: ignore
            elif isinstance(data, (torch.Tensor, np.ndarray)):
                out[field.name] = torch.as_tensor(data, device=device)  # type: ignore
            elif data is None:
                out[field.name] = data
            else:
                raise RuntimeError(f"Not sure what to do with {field.name}: {data}")
        return EpisodeBatch(**out)

    @classmethod
    def collate_fn(cls, batches: tp.List["EpisodeBatch[T]"]) -> "EpisodeBatch[torch.Tensor]":
        """Creates a new instance from several by stacking in a new first dimension
        for all attributes
        """
        out: tp.Dict[str, tp.Any] = {}
        if isinstance(batches[0].obs, np.ndarray):  # move everything to pytorch if first one is numpy
            batches = [b.to("cpu") for b in batches]  # type: ignore
        for field in dataclasses.fields(cls):
            data = [getattr(mf, field.name) for mf in batches]
            # skip fields with None data
            if data[0] is None:
                if any(x is not None for x in data):
                    raise RuntimeError("Found a non-None value mixed with Nones")
                out[field.name] = None
                continue
            # reward and discount can be float which should be converted to
            # tensors for stacking
            if field.name == "meta":
                meta = {k: torch.stack([d[k] for d in data]) for k in data[0]}
                out[field.name] = meta
            elif isinstance(data[0], torch.Tensor):
                out[field.name] = torch.stack(data)
            else:
                raise RuntimeError(f"Not sure what to do with {field.name}: {data}")
                # out[field.name] = [x for y in data for x in y]
        return EpisodeBatch(**out)

    def unpack(self) -> tp.Tuple[T, T, T, T, T]:
        """Unpacks the structure into the legacy unnamed tuple.
        Try to avoid it if possible, this is more likely to be wrong than using names
        """
        return (self.obs, self.action, self.reward, self.discount, self.next_obs)
        # return (self.obs, self.action, self.reward, self.discount, self.next_obs, *self.meta)

    def with_no_reward(self: B) -> B:
        reward = self.reward
        reward = torch.zeros_like(reward) if isinstance(reward, torch.Tensor) else 0 * reward
        return dataclasses.replace(self, reward=reward)
