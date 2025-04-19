# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Direct-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env:Go2FlatEnvCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Direct-Norm-v0",
    entry_point=f"{__name__}.go2_env_norm:Go2NormEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_norm:Go2FlatEnvNormCfg",
    },
)