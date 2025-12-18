# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils
from rsl_rl.modules import EmpiricalNormalization
from typing import Tuple, Dict

from ..env_default_abs.go2_cfg_rnd_full import Go2FlatEnvNormCfg
from ..env_default_abs.go2_env import Go2NormEnv

# 继承自 Go2NormEnv，实现增量式动作空间
# Successor of Go2NormEnv, implementing incremental action space
class Go2_Incremental_Env(Go2NormEnv):
    
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.joint_pos #0000ff
        # self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos   # original one

    #00ff00 -0.1
    # In incremental control, action rate is just the action
    # Since action rate the difference between current action and previous action
    def action_rate_l2(self) -> torch.Tensor:
        return torch.sum(torch.square(self._actions), dim=1)