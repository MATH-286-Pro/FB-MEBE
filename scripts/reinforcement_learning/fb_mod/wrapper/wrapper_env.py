# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch
import numpy as np
import copy

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.direct.go2.env_default_abs.go2_env import Go2NormEnv

import gc

class FB_VecEnvWrapper(VecEnv):

    def __init__(self, env: DirectRLEnv):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim # type: ignore
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0] # type: ignore
        else:
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])
        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim # type: ignore
        ):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0] # type: ignore
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_privileged_obs = 0
        
        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @property
    def observation_spec(self):
        """Returns the observation specification of the environment."""
        return self.env.unwrapped.single_observation_space['policy'] # type: ignore

    @property
    def goal_spec(self):
        """Returns the goal specification of the environment."""
        return self.env.unwrapped.single_observation_space['goal'] # type: ignore

    @property
    def action_spec(self):
        """Returns the action specification of the environment."""
        return self.env.unwrapped.single_action_space # type: ignore

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> Go2NormEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped # type: ignore

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute() # type: ignore
        else:
            obs_dict = self.unwrapped._get_observations() # type: ignore
        return obs_dict["policy"], {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    ####################################################################################################################
    # 兼容 video wrapper 的接口       
    def stop_recording(self, name_prefix: str, step: int):
        try:
            self.env.stop_recording(name_prefix, step)
        except AttributeError:
            pass

    ####################################################################################################################
    def eval_task(self, command_xyw: list):
        self.env.unwrapped.termination_type = "none"                                               # type: ignore
        self.env.unwrapped._commands[:] = torch.tensor(command_xyw, device = self.device)          # type: ignore
        self.env.unwrapped._is_eval_mode = True                                                    # type: ignore

    def train_task(self):
        self.env.unwrapped.termination_type = "contact"   # type: ignore
        self.env.unwrapped._is_eval_mode = False          # type: ignore


    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self):

        obs_dict, _ = self.env.reset()

        index_qpos = list(range(9,  21))
        index_qvel = list(range(21, 33))
        td = {
            'obs':         copy.deepcopy(obs_dict),
            'time':        self.env.unwrapped.episode_length_buf.unsqueeze(-1).clone(), # type: ignore
            'done':        torch.zeros(obs_dict['policy'].shape[0], device=self.device, requires_grad=False),
            'reward_task': torch.zeros(obs_dict['policy'].shape[0], device=self.device, requires_grad=False),
            'reward_reg':  torch.zeros(obs_dict['policy'].shape[0], device=self.device, requires_grad=False)
        }

        info = {
            # 'qpos': obs_dict['raw'][:, index_qpos].clone(),
            # 'qvel': obs_dict['raw'][:, index_qvel].clone()
        }
        
        return td, info

    def step(self, actions: torch.Tensor):

        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, device=self.device)

        obs_dict, reward, terminated, truncated, extras = self.env.step(actions)  # 这里的 obs 是从 _get_observations() 获取的，所以已经 noise + normalize 过了
    
        index_qpos = list(range(9,  21))
        index_qvel = list(range(21, 33))

        td = {
            'obs':  copy.deepcopy(obs_dict),
            'time': self.env.unwrapped.episode_length_buf.unsqueeze(-1).clone(), # type: ignore
            'done': (terminated | truncated).clone(),
            'reward_task': reward.clone(),
            'reward_reg':  self.env.unwrapped.get_reg_reward().clone()      # type: ignore
        }

        info = {
            # 'qpos':     obs_dict['raw'][:, index_qpos].clone(),
            # 'qvel':     obs_dict['raw'][:, index_qvel].clone(),
            'rew_dict': copy.deepcopy(extras['rew_dict']),
            'reg_dict': copy.deepcopy(self.env.unwrapped.get_env_metrics()),
        }

        return td, reward.clone(), terminated.clone(), truncated.clone(), info

    def close(self):
        self.env.close()