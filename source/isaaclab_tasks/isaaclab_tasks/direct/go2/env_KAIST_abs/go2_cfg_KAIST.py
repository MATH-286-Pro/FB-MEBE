# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from ..env_default_abs.go2_cfg_rnd_full import Go2FlatEnvNormCfg


@configclass
class Go2FlatEnvNormKAISTCfg(Go2FlatEnvNormCfg):
    policy_space      = 33+0+12+2      # actor
    observation_space = 33+1           # F
    goal_space        = 10             # B
    critic_space      = 33+1+12+2+8    # critic