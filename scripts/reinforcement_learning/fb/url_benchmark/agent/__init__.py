# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# register agents for hydra
from .ddpg import DDPGAgent as DDPGAgent
from .ddpg import DDPGAgentConfig as DDPGAgentConfig
from .fb_ddpg import FBDDPGAgent as FBDDPGAgent
from .ddpg import MetaDict as MetaDict
