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
from .go2_cfg_base import Go2_Base_Cfg

#################
FF_TORQUE = True
GRAVITY_COMPENSATION = True  # only for swing legs

BASE_ACTION = True
BASE_ACTION_SCALE = 5.0

JOINT_ACTION = True
JOINT_ACTION_SCALE = 0.15

SAMPLE_VEL_COMMANDS = True
SAMPLE_POS_COMMANDS = True
SAMPLE_FORCE_COMMANDS = True

POS_LIMIT_MARGIN = 0.1  # rad
TORQUE_LIMIT_SCALE = 0.9

#################
ALPHA = 0.5
POS_ALPHA = ALPHA
VEL_ALPHA = ALPHA
TOR_ALPHA = ALPHA

ADD_LINK_DR = True

ACTUATOR_DELAY = True
ACTUATOR_DELAY_STEPS = 10
################

VEL_X = [-0.5, 0.5]
VEL_Y = [-0.5, 0.5]
AVEL_Z = [-0.5, 0.5]

# set to zero when using standing gait
# VEL_X = [-0.0, 0.0]
# VEL_Y = [-0.0, 0.0]
# AVEL_Z = [-0.0, 0.0]

FL_POS_X = [0.1934, 0.5]
FL_POS_Y = [0, 0.2]
FL_POS_Z = [0.0, 0.4]

FL_FORCE_X = [-30.0, 30.0]
FL_FORCE_Y = [-30.0, 30.0]
FL_FORCE_Z = [-30.0, 30.0]


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

    add_base_com_pos = EventTerm(
        func=mdp.randomize_rigid_body_com_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_pos_distribution_params": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )


    # Random reset base pos + yaw
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), 
                           "y": (-0.5, 0.5), 
                           "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, +0.3),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class Go2FlatEnvNormCfg(Go2_Base_Cfg):

    # events
    events: EventCfg = EventCfg()

    # Domain Randomization
    add_noise = True