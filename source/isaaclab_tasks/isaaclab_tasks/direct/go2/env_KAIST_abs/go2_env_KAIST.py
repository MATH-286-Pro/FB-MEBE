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

from .go2_cfg_KAIST import Go2FlatEnvNormKAISTCfg
from ..env_default_abs.go2_env import Go2NormEnv

# 继承自 Go2NormEnv，实现增量式动作空间
# Successor of Go2NormEnv, implementing incremental action space
class Go2_KAIST_Env(Go2NormEnv):
    cfg: Go2FlatEnvNormKAISTCfg
    
    def __init__(self, cfg: Go2FlatEnvNormKAISTCfg, render_mode: str | None = None, **kwargs):
        self._T = 0.4        
        super().__init__(cfg, render_mode, **kwargs)


    def user_return_policy_obs(self, add_noise):
        return torch.cat(
            [
                self._robot.data.root_lin_vel_b      + (add_noise * torch.rand_like(self._robot.data.root_lin_vel_b) * 0.2 - 0.1),
                self._robot.data.root_ang_vel_b      + (add_noise * torch.rand_like(self._robot.data.root_ang_vel_b) * 0.4 - 0.2),
                self._robot.data.projected_gravity_b + (add_noise * torch.rand_like(self._robot.data.projected_gravity_b) * 0.1 - 0.05),
                # no height
                self._robot.data.joint_pos - self._robot.data.default_joint_pos + (add_noise * torch.rand_like(self._robot.data.joint_pos) * 0.02 - 0.01),
                self._robot.data.joint_vel + (add_noise * torch.rand_like(self._robot.data.joint_vel) * 0.3 - 0.15),
                self._previous_actions,   # 12 dim
                torch.sin(2*torch.pi * self.episode_length_buf / (self._T / self.step_dt)).unsqueeze(-1), # 时间信息
                torch.cos(2*torch.pi * self.episode_length_buf / (self._T / self.step_dt)).unsqueeze(-1),
            ],
            dim=-1,
        )
    
    def user_return_critic_obs(self):
        return torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.root_pos_w[:, 2:],
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                self._previous_actions,
                torch.sin(2*torch.pi * self.episode_length_buf / (self._T / self.step_dt)).unsqueeze(-1), # 时间信息
                torch.cos(2*torch.pi * self.episode_length_buf / (self._T / self.step_dt)).unsqueeze(-1),
                self._feet_contact_force,
                self._feet_height_z,
            ],
            dim=-1,
        )
    
    def user_return_reguarization_reward(self):
    
        # Regularization
        self.action_rate = torch.norm(self._actions - self._previous_actions, dim=1)
        self.action_rate_rew = torch.exp(-torch.square(self.action_rate / 4.0))  # 4.0

        self.joint_torques = torch.norm(self._robot.data.applied_torque, dim=1)
        self.joint_torques_rew = torch.exp(-torch.square(self.joint_torques / 20))

        #0000ff 2025.10.16 new added
        self.air_time       = self.compute_feet_air_time(threshold=0.5)
        self.foot_clearance = self.compute_feet_clearance(target_height=0.10, std=0.005, tanh_mult=2.0)
        self.slip_penalty   = self.compute_feet_slip_penalty(threshold=1.0)

        #00ff00 unitree_isaaclab
        self.penalty_joint_vel         = self.joint_vel_l2()
        self.penalty_joint_acc         = self.joint_acc_l2()
        self.penalty_joint_torques     = self.joint_torque_l2()
        self.penalty_action_rate       = self.action_rate_l2()
        self.penalty_energy            = self.energy()

        self.reward_feet_air_time      = self.feet_air_time(threshold=0.5)
        self.penalty_air_time_variance = self.air_time_variance()
        self.penalty_feet_slide        = self.feet_slide()

        self.penalty_undesired_contact = self.undesired_contacts(threshold=1.0)

        regularization_rewards = {
            'joint_acc':         -2.5e-7 * self.penalty_joint_acc,
            'action_rate':       -0.1 *    self.penalty_action_rate,
            'feet_slide':        -0.5 * self.feet_slide(),
            'barrier':           +1.0 * self._get_barrier_restriction_rewards(),
            # 'joint_vel':         -0.001 *  self.penalty_joint_vel,
            # 'joint_torques':     -2e-4 *   self.penalty_joint_torques,
            # 'energy':            -2e-5 *   self.penalty_energy,

            # "air_time":          +0.1 * self.reward_feet_air_time,
            # 'air_time_variance': -1.0 * self.penalty_air_time_variance,
            # 'feet_slide':        -0.1 * self.penalty_feet_slide,

            # 'undesired_contact': -1.0 * self.penalty_undesired_contact,
        }
        return regularization_rewards
    

    # Paper 2025 KAIST: 
    #  A Learning Framework for diverse Legged Robot Locomotion Using Barrier-Based Style Rewards
    #00fff0 Soft Barrier Function
    def Barrier(self, z: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
        delta_tensor = torch.tensor(delta, device=z.device, dtype=z.dtype)
        return torch.where(z > delta_tensor, 
                        torch.log(z),
                        torch.log(delta_tensor) - 0.5 * (((z - 2*delta_tensor)/delta_tensor)**2 - 1))

    #00fff0 KAIST feet reward
    def _get_barrier_restriction_rewards(self)-> torch.Tensor:

        # 机器人各个状态
        env_step = self.episode_length_buf
        dt = self.step_dt
        feet_z_pos = self._robot.data.body_pos_w[:, self._feet_ids, 2]       # [1024, 4]
        contact_bool = feet_z_pos <= 0.024
        contact_coeff = (contact_bool.float() - 0.5) * 2.0                   # -1 or +1

        # 不同约束奖励系数
        d_gait_lower = -0.6
        d_feet_clearance_lower = -0.08

        delta_gait = 0.1                  # 0.1
        delta_feet_clearance = 0.01        # 0.01


        # 使用的约束
        # 1. 步态奖励
        #      dt=0.02 周期为 1s, 50 步为一个周期
        #      g(step) = sin(2π * (step / 50 + phi_i))  phi_i 为各脚相位差
        def gait_sine(step, dt):
            T = self._T 
            T_step = T / dt
            phase_offset = [0.0, 0.5, 0.5, 0.0]  # trot
            g = torch.stack([torch.sin(2*torch.pi * (step / T_step + phase_offset[0])),    # FL 左前
                             torch.sin(2*torch.pi * (step / T_step + phase_offset[1])),    # FR 右前
                             torch.sin(2*torch.pi * (step / T_step + phase_offset[2])),    # RL 左后
                             torch.sin(2*torch.pi * (step / T_step + phase_offset[3]))     # RR 右后
                             ], dim=1)  # [1024, 4]
            return g

        g = gait_sine(env_step, dt)   # [1024, 4]
        f = contact_coeff * g     # [1024, 4]
        R_gait = torch.sum(self.Barrier(-d_gait_lower + f, delta_gait), dim = 1) 


        # 2. 足端高度奖励
        feet_diff = torch.clip(feet_z_pos - 0.07, max=0.0)   # 足端高度超过 0.10m 部分不计算奖励
        l = torch.where(g <= d_gait_lower, feet_diff, 0)
        R_feet_clearance = torch.sum(self.Barrier(-d_feet_clearance_lower + l, delta_feet_clearance), dim = 1)


        # 打包奖励
        reward_dict = {
            "gait": R_gait,
            "feet_clearance": R_feet_clearance
        }

        alpha = {"gait":           torch.tensor(0.1, device=self.device),
                 "feet_clearance": torch.tensor(0.1, device=self.device),}

        # 累加奖励
        regularization_reward = torch.zeros_like(self.episode_length_buf, dtype=torch.float32)

        for key, value in reward_dict.items():
            regularization_reward += alpha[key] * value
        
        return regularization_reward