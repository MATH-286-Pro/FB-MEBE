# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym
from . import agents


gym.register(
    id="Isaac-Flat-Unitree-Go2-Rnd-Full-FB-ABS-v0",
    entry_point=f"{__name__}.env_default_abs.go2_env:Go2NormEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_default_abs.go2_cfg_rnd_full:Go2FlatEnvNormCfg",          # Env Config
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Flat-Unitree-Go2-Rnd-Full-FB-INC-v0",
    entry_point=f"{__name__}.env_default_inc.go2_env_incremental:Go2_Incremental_Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_default_inc.go2_cfg_rnd_full_incremental:Go2FlatEnvNormCfg",          # Env Config
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Flat-Unitree-Go2-Rnd-Full-FB-ABS-KAIST-v0",
    entry_point=f"{__name__}.env_KAIST_abs.go2_env_KAIST:Go2_KAIST_Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_KAIST_abs.go2_cfg_KAIST:Go2FlatEnvNormKAISTCfg",          # Env Config
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Flat-Unitree-Go2-FB-v0",
    entry_point=f"{__name__}.go2_env:Go2NormEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_cfg_fix_f:Go2FlatEnvNormCfg",          # Env Config
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)