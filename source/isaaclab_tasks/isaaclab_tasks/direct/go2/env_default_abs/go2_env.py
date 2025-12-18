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

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG

from .go2_cfg_fix_f import Go2FlatEnvNormCfg


class Go2NormEnv(DirectRLEnv):
    cfg: Go2FlatEnvNormCfg

    def __init__(self, cfg: Go2FlatEnvNormCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # reward selection
        self.reward_type_list = ["default", "trot", "pace"]  # use this to choose reward
        self.reward_type = self.reward_type_list[1]

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands + base_tilt orientation
        self._commands         = torch.zeros(self.num_envs, 3, device=self.device)
        self._commands_range = {
            "vx": [-1.0, +1.0],
            "vy": [-0.5, +0.5],
            "wz": [-1.0, +1.0],
        }
        # Sample initial commands from _commands_range
        self._commands[:, 0] = torch.rand(self.num_envs, device=self.device) * (self._commands_range["vx"][1] - self._commands_range["vx"][0]) + self._commands_range["vx"][0]
        self._commands[:, 1] = torch.rand(self.num_envs, device=self.device) * (self._commands_range["vy"][1] - self._commands_range["vy"][0]) + self._commands_range["vy"][0]
        self._commands[:, 2] = torch.rand(self.num_envs, device=self.device) * (self._commands_range["wz"][1] - self._commands_range["wz"][0]) + self._commands_range["wz"][0]
        self.desired_base_tilt = torch.tensor([0, 0, -1],      device=self.device)
        self.task_reward = "locomotion"  # make this default for training with regularizer
        self._is_eval_mode = False       # flag to prevent command randomization during eval

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [

                "lin_vel_xy_rew",
                "ang_vel_z_rew",
                "lin_z_rew",
                "ang_xy_rew",
                "action_rate_rew",
                "action_rate",
                "joint_torques_rew",
                "joint_torques",
                "foot_clearance",
                "air_time",
                "gait_symmetry",
                "reg_reward",
                "base_height",
                "feet_slip_penalty",
            ]
        }

        # Get specific body indices
        #ff0000 2025.9.25 14:52 These index are different from the ones in self._robot
        self._base_contact_id, _   = self._contact_sensor.find_bodies("base")       
        self._other_contact_id, _  = self._contact_sensor.find_bodies("Head_.*")
        self._feet_contact_ids, _  = self._contact_sensor.find_bodies(".*foot")
        self._thigh_contact_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        self._calf_contact_ids, _  = self._contact_sensor.find_bodies(".*calf")
        self._hip_contact_ids, _   = self._contact_sensor.find_bodies(".*hip")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*thigh")

        self._base_id, _   = self._robot.find_bodies("base")
        self._other_id, _  = self._robot.find_bodies("Head_.*")
        self._feet_ids, _  = self._robot.find_bodies(".*foot")
        self._thigh_ids, _ = self._robot.find_bodies(".*thigh")
        self._calf_ids, _  = self._robot.find_bodies(".*calf")

        # Set up the environment
        self.termination_type = "contact"

        # User defined variable
        self._feet_contact_state = torch.zeros((self.num_envs, len(self._feet_contact_ids)), dtype=torch.bool, device=self.device)
        
        # Robot Die and Reset Delay 延迟重置相关变量
        self.delay_reset_frames = 10  # 延迟重置帧数
        self.contact_delay_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)  # 每个环境的延迟计数器
        self.has_undesired_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 是否有不期望的接触
        
        obs_test = self._get_observations()
        # assert obs_test['policy'].shape[1] == self.single_observation_space['policy'].shape[0], \
        #     f"Observation space mismatch: {obs_test['policy'].shape[1]} != {self.single_observation_space['policy'].shape[0]}"
        # if 'goal' in self.single_observation_space:
        #     assert obs_test['goal'].shape[1] == self.single_observation_space['goal'].shape[0], \
        #         f"Goal space mismatch: {obs_test['goal'].shape[1]} != {self.single_observation_space['goal'].shape[0]}"

        # Align space dimensions with cfg
        if 'policy' in self.single_observation_space:
            assert obs_test['policy'].shape[1] == self.cfg.observation_space, \
            f"Policy observation vs cfg mismatch: {obs_test['policy'].shape[1]} != {self.cfg.observation_space}"
        
        if 'goal' in self.single_observation_space:
            assert obs_test['goal'].shape[1] == self.single_observation_space['goal'].shape[0], \
            f"Goal space mismatch: {obs_test['goal'].shape[1]} != {self.single_observation_space['goal'].shape[0]}"
            assert obs_test['goal'].shape[1] == self.cfg.goal_space, \
            f"Goal observation vs cfg mismatch: {obs_test['goal'].shape[1]} != {self.cfg.goal_space}"
        
        if 'critic' in obs_test:
            assert obs_test['critic'].shape[1] == self.cfg.critic_space, \
            f"Critic observation vs cfg mismatch: {obs_test['critic'].shape[1]} != {self.cfg.critic_space}"

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._actions = torch.clamp(self._actions, -1.0, 1.0)
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:

        # Data explanation:
        #   self._contact_sensor.data.net_forces_w: shape (num_envs, num_bodies, 3)
        #   self._contact_sensor.data.net_forces_w_history: shape (num_envs, history_length, num_bodies, 3)

        # observation update
        self._feet_contact_force = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids, :], dim=-1)
        self._feet_contact_state = self._feet_contact_force > 1.0   # Contact threshold of 1N
        self._feet_height_z      = self._robot.data.body_pos_w[:, self._feet_ids, 2]
        self._feet_xy_vel_norm   = torch.norm(self._robot.data.body_lin_vel_w[:, self._feet_ids, :2], dim=-1)


        # 带噪声的观测
        add_noise = self.cfg.add_noise
        obs_noise = torch.cat(
            [
                self._robot.data.root_lin_vel_b      + (add_noise * torch.rand_like(self._robot.data.root_lin_vel_b) * 0.2 - 0.1),
                self._robot.data.root_ang_vel_b      + (add_noise * torch.rand_like(self._robot.data.root_ang_vel_b) * 0.4 - 0.2),
                self._robot.data.projected_gravity_b + (add_noise * torch.rand_like(self._robot.data.projected_gravity_b) * 0.1 - 0.05),
                # no height
                self._robot.data.joint_pos - self._robot.data.default_joint_pos + (add_noise * torch.rand_like(self._robot.data.joint_pos) * 0.02 - 0.01),
                self._robot.data.joint_vel + (add_noise * torch.rand_like(self._robot.data.joint_vel) * 0.3 - 0.15),
                self._previous_actions,   # 12 dim
            ],
            dim=-1,
        )

        # 不带噪声的观测
        obs_raw = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.root_pos_w[:, 2:],
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                self._previous_actions,
                self._feet_contact_force,
                self._feet_height_z,
            ],
            dim=-1,
        )

        # 用于 inference 奖励函数计算
        obs_raw_dict = {
            "lin_vel":                self._robot.data.root_lin_vel_b,
            "ang_vel":                self._robot.data.root_ang_vel_b,
            "gravity":                self._robot.data.projected_gravity_b,

            "base_height":            self._robot.data.root_pos_w[:, 2:],

            "joint_pos":              self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            "joint_vel":              self._robot.data.joint_vel,
            "joint_acc":              self._robot.data.joint_acc,
            "joint_torque":           self._robot.data.applied_torque,

            "actions":                self._actions,
            "last_actions":           self._previous_actions,

            "feet_contact_state":     self._feet_contact_state,
            "feet_height":            self._feet_height_z,
            "feet_xy_vel_norm":       self._feet_xy_vel_norm,
            "feet_last_air_time":     self._contact_sensor.data.last_air_time[:, self._feet_contact_ids],
            "feet_last_contact_time": self._contact_sensor.data.last_contact_time[:, self._feet_contact_ids],
            "feet_first_contact":     self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_ids],
        }

        observations = {"policy": obs_noise[:, :self.cfg.policy_space],        # (for actor)
                        "obs"   : obs_raw[:, :self.cfg.observation_space],     # (for F)
                        "goal"  : obs_raw[:, :self.cfg.goal_space],            # (for B)
                        "critic": obs_raw[:, :self.cfg.critic_space],          # (for critic)
                        "raw":    obs_raw_dict                                 # (for inference)
                        }
        
        self._previous_actions = self._actions.detach().clone()

        return observations
    

    def _get_rewards(self) -> tuple[torch.Tensor, dict]:

        task_rewards, regularization_rewards = self._get_locomotion_rewards()
        # self.reg_rew = torch.prod(torch.stack(list(regularization_rewards.values())), dim=0) #0000ff
        self.reg_rew = torch.sum(torch.stack(list(regularization_rewards.values())), dim=0) #0000ff 改为累加
        regularization_metrics = self.get_env_metrics()  #0000ff

        # all_rewards = task_rewards | regularization_metrics

        if self.task_reward == 'locomotion':
            reward = 1.0 * torch.prod(torch.stack(list(task_rewards.values())), dim=0)

        elif self.task_reward == 'reg_locomotion':
            reward = torch.prod(torch.stack(list(regularization_rewards.values())), dim=0)

        else:
            raise ValueError(f"Unknown reward type: {self.task_reward}")

        # Logging
        metrics_dict = dict()
        for key, value in task_rewards.items():
            self._episode_sums[key] += value
            metrics_dict[key] = value.mean().item()
        
        metrics_dict.update({'reward_task': reward.mean().item()})
        metrics_dict.update(regularization_metrics)

        return reward, metrics_dict

    def _get_locomotion_rewards(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # linear velocity tracking
        # task
        lin_vel_xy_error = torch.norm(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2], dim=1)
        lin_vel_xy_rew = torch.exp(-torch.square(lin_vel_xy_error / 0.3))
        # yaw rate tracking
        ang_vel_z_error = torch.abs(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        ang_vel_z_rew = torch.exp(-torch.square(ang_vel_z_error / 0.2))

        # pose (base height)
        lin_z_error = torch.abs(self._robot.data.root_pos_w[:, 2] - 0.28)
        lin_z_rew = torch.exp(-torch.square(lin_z_error / 0.1))

        target_gravity_b = torch.zeros_like(self._robot.data.projected_gravity_b)
        target_gravity_b[:, 2] = -1.0
        ang_xy_error = torch.norm(self._robot.data.projected_gravity_b - target_gravity_b, dim=1)
        ang_xy_rew = torch.exp(-torch.square(ang_xy_error / 0.1))

        # lin_vel_z_error = torch.abs(self._robot.data.root_lin_vel_b[:, 2])
        # lin_vel_z_rew = torch.exp(-torch.square(lin_vel_z_error / 0.2))

        # ang_vel_xy_error = torch.norm(self._robot.data.root_ang_vel_b[:, :2], dim=1)
        # ang_vel_xy_rew = torch.exp(-torch.square(ang_vel_xy_error / 2.0))

        # Regularization
        self.action_rate = torch.norm(self._actions - self._previous_actions, dim=1)
        self.action_rate_rew = torch.exp(-torch.square(self.action_rate / 4.0))  # 4.0

        self.joint_torques = torch.norm(self._robot.data.applied_torque, dim=1)
        self.joint_torques_rew = torch.exp(-torch.square(self.joint_torques / 20))

        #0000ff 2025.10.16 new added
        self.air_time       = self.compute_feet_air_time(threshold=0.5)
        self.foot_clearance = self.compute_feet_clearance(target_height=0.10, std=0.005, tanh_mult=2.0)
        self.slip_penalty   = self.compute_feet_slip_penalty(threshold=1.0)


        task_rewards = {
            "lin_vel_xy_rew": lin_vel_xy_rew,
            "ang_vel_z_rew": ang_vel_z_rew,
            # "lin_z_rew": lin_z_rew,
            "ang_xy_rew": ang_xy_rew,
        }

        regularization_rewards = {
            # "action_rate_rew":    self.action_rate_rew,
            # "joint_torques_rew": self.joint_torques_rew,
            # "foot_clearance":    self.foot_clearance,
            "air_time":          +5.0 * self.air_time,
            # 'gait_symmetry':     gait_symmetry,
            'feet_slip_penalty': -0.5 * self.slip_penalty,
        }

        return task_rewards, regularization_rewards


    #0000ff
    def get_reg_reward(self):
        return self.reg_rew
    
    #0000ff
    def get_env_metrics(self) -> dict:
        
        # 计算其他可能需要的指标
        base_height = self._robot.data.root_pos_w[:, 2]
        
        return {
            'reward_reg':         self.reg_rew.mean().item(),
            'action_rate':        self.action_rate.mean().item(),
            'foot_clearance':     self.foot_clearance.mean().item(),
            'joint_torques':      self.joint_torques.mean().item(),
            'base_height':        base_height.mean().item(),
            'air_time':           self.air_time.mean().item(),
            'feet_slip_penalty':  self.slip_penalty.mean().item(),
            'reset_percent':      torch.sum(self.reset_buf).item() / self.num_envs * 100,
        }

    # termination condition #0000ff
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        died = torch.zeros_like(time_out)
        
        if "contact" in self.termination_type:

            if self.termination_type == "contact":
                self.delay_reset_frames = 1
            
            elif self.termination_type == "contact_delay":
                self.delay_reset_frames = 10

            net_contact_forces = self._contact_sensor.data.net_forces_w_history  # type: ignore

            # Force any where else except the feet larger than 1N will reset the robot
            # 机器人除了脚其他任一位置接触到的力大于1N将重置机器人
            current_undesired_contact = (
                torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_contact_id], dim=-1), dim=1)[0] > 1.0, dim=1) |
                torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._other_contact_id], dim=-1), dim=1)[0] > 1.0, dim=1) |
                torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._thigh_contact_ids], dim=-1), dim=1)[0] > 1.0, dim=1) |
                torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._calf_contact_ids], dim=-1), dim=1)[0] > 1.0, dim=1) |
                torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._hip_contact_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
            )
            
            # 延迟重置逻辑
            # 如果当前有不期望的接触，但之前没有，开始计数
            new_contact = current_undesired_contact & ~self.has_undesired_contact
            self.contact_delay_counter[new_contact] = 0
            self.has_undesired_contact[new_contact] = True
            
            # 如果当前没有不期望的接触，重置状态
            no_contact = ~current_undesired_contact
            self.has_undesired_contact[no_contact] = False
            self.contact_delay_counter[no_contact] = 0
            
            # 对于有不期望接触的环境，增加计数器
            self.contact_delay_counter[self.has_undesired_contact] += 1
            
            # 当计数器达到延迟帧数时，才真正重置
            died = self.contact_delay_counter >= self.delay_reset_frames

        elif self.termination_type == "roll":
            # Add roll angle termination condition
            # Extract roll angle from root orientation quaternion
            root_quat = self._robot.data.root_quat_w  # [num_envs, 4] (w, x, y, z)
            roll, _, _ = math_utils.euler_xyz_from_quat(root_quat)  # Extract roll, pitch, yaw
            roll_degrees = torch.rad2deg(roll)  # Convert to degrees
            roll_degrees = ((roll_degrees + 180) % 360) - 180
            died_roll = (170 < roll_degrees) | (roll_degrees < -170.0)

            died = died | died_roll

        elif self.termination_type == "none":
            pass

        else:
            raise ValueError(f"Unknown termination condition: {self.termination_type}")

        return died, time_out
        # died 或者 time_out 都会重置环境
        # self.reset_buf = self.reset_terminated | self.reset_time_outs (in direct_rl_env.py)

    # reset robot in env_ids
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # 检查重置后的脚高度，如果最小脚高度 < 0.02，抬高机器人基座
        # 需要写入数据后才能读取更新后的位置
        self._robot.write_data_to_sim()
        self.scene.update(dt=0.0)  # 更新场景以获取最新的body位置
        
        # 获取重置环境的脚高度
        feet_height_reset = self._robot.data.body_pos_w[env_ids][:, self._feet_ids, 2]  # [len(env_ids), 4]
        min_feet_height = torch.min(feet_height_reset, dim=1)[0]  # [len(env_ids)]
        
        # 找到需要抬高的环境（最小脚高度 < 0.02）
        need_lift = min_feet_height < 0.02
        lift_env_ids = env_ids[need_lift]
        
        if len(lift_env_ids) > 0:
            # 计算需要抬高的高度（目标是最小脚高度达到 0.02，再加一点余量 0.01）
            lift_height = 0.03 - min_feet_height[need_lift]
            
            # 获取当前 base 位置并抬高
            current_pos = self._robot.data.root_pos_w[lift_env_ids].clone()
            current_pos[:, 2] += lift_height
            
            # 设置新的 base 位置
            self._robot.write_root_pose_to_sim(
                root_pose=torch.cat([current_pos, self._robot.data.root_quat_w[lift_env_ids]], dim=-1),
                env_ids=lift_env_ids
            )

        # 到达 episode length 的环境会根据 episode_length_buf 重置
        self.episode_length_buf[env_ids] = 0
        self._actions[env_ids] = torch.zeros_like(self._actions[env_ids])
        self._previous_actions = self._previous_actions.clone()
        self._previous_actions[env_ids] = torch.zeros_like(self._actions[env_ids])
        
        # 重置延迟重置相关的变量
        self.contact_delay_counter[env_ids] = 0
        self.has_undesired_contact[env_ids] = False
        
        # Sample new commands from _commands_range (only in training mode)
        if not self._is_eval_mode:
            self._commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (self._commands_range["vx"][1] - self._commands_range["vx"][0]) + self._commands_range["vx"][0]
            self._commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (self._commands_range["vy"][1] - self._commands_range["vy"][0]) + self._commands_range["vy"][0]
            self._commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (self._commands_range["wz"][1] - self._commands_range["wz"][0]) + self._commands_range["wz"][0]



    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or toggle visibility of the velocity arrows used for debugging.

        When enabled, two instanced arrow markers are created: one for the desired velocity
        (green) and one for the current velocity (blue). The markers are updated in the
        `_debug_vis_callback` which is subscribed by the parent class when debug vis is enabled.
        """
        # create markers if necessary
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                # Use the pre-defined arrow USD files and place them under /Visuals/Command

                goal_vel_visualizer_cfg    = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
                current_vel_visualizer_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current")

                goal_vel_visualizer_cfg.markers["arrow"].scale    = (0.5, 0.5, 0.5)
                current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

                self.goal_vel_visualizer = VisualizationMarkers(
                    goal_vel_visualizer_cfg
                )
                self.current_vel_visualizer = VisualizationMarkers(
                    current_vel_visualizer_cfg
                )
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Post-update callback that updates arrow poses/scales each frame."""
        # ensure robot is initialized
        if not self._robot.is_initialized:
            return

        # base position for arrows (slightly above robot)
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # desired velocity (commands) and actual velocity (in base frame -> convert to arrow)
        # Note: self._commands is in base-frame (x,y) matching robot.data.root_lin_vel_b
        desired_xy = self._commands[:, :2]
        actual_xy  = self._robot.data.root_lin_vel_b[:, :2]

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(desired_xy, use_goal=True)
        vel_arrow_scale, vel_arrow_quat         = self._resolve_xy_velocity_to_arrow(actual_xy, use_goal=False)

        # visualize
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor, use_goal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert XY velocity vectors to marker scales and quaternions for arrow visualization.

        Returns (scales, orientations) where orientations are quaternions in (w,x,y,z) and
        scales are (num_envs, 3).
        """
        # pick the appropriate visualizer to read default scale
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale

        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        # direction -> yaw angle
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        # convert from base -> world by rotating with base quaternion
        base_quat_w = self._robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


    #0000ff weight = +5.0 SPOT
    def compute_feet_air_time(self, threshold=0.5):
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_contact_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
        return reward
    
    #0000ff weight = +0.5 SPOT
    def compute_feet_clearance(self, target_height=0.10, std = 0.05, tanh_mult = 2.0):
        foot_error = torch.square(self._robot.data.body_pos_w[:, self._feet_ids, 2] - target_height)
        foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(self._robot.data.body_lin_vel_w[:, self._feet_ids, :2], dim=2))
        reward = foot_error * foot_velocity_tanh
        return torch.exp(-torch.sum(reward, dim=1) / std)

    #0000ff weight = -0.5 SPOT
    def compute_feet_slip_penalty(self, threshold=1.0):

        net_contact_forces = self._contact_sensor.data.net_forces_w_history

        is_contact = torch.max(torch.norm(net_contact_forces[:,:, self._feet_contact_ids], dim=-1), dim=1)[0] > threshold
        foot_planner_velocity = torch.linalg.norm(self._robot.data.body_lin_vel_w[:, self._feet_ids, :2], dim=2)

        reward = is_contact * foot_planner_velocity
        return torch.sum(reward, dim=1)