# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

from .go2_cfg import Go2FlatEnvCfg


class Go2Env(DirectRLEnv):
    cfg: Go2FlatEnvCfg

    def __init__(self, cfg: Go2FlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # reward selection
        self.reward_type_list = ["default", "trot", "pace"]  # use this to choose reward
        self.reward_type = self.reward_type_list[0]

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.desired_velocity = torch.tensor([0.5, 0, 0], device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "base_height_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "flat_orientation_l2",
                "gait_symmetry"
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        self.use_termination = True

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
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        add_noise = self.cfg.add_noise
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b + (
                        add_noise * torch.rand_like(self._robot.data.root_lin_vel_b) * 0.2 - 0.1),
                self._robot.data.root_ang_vel_b + (
                        add_noise * torch.rand_like(self._robot.data.root_ang_vel_b) * 0.4 - 0.2),
                self._robot.data.projected_gravity_b + (
                        add_noise * torch.rand_like(self._robot.data.projected_gravity_b) * 0.1 - 0.05),
                # self._commands,
                self._robot.data.joint_pos - self._robot.data.default_joint_pos + (
                        add_noise * torch.rand_like(self._robot.data.joint_pos) * 0.02 - 0.01),
                self._robot.data.joint_vel + (
                        add_noise * torch.rand_like(self._robot.data.joint_vel) * 0.3 - 0.15),
                self._actions,
            ],
            dim=-1,
        )
        goal = torch.cat([
            self._robot.data.root_lin_vel_b + (
                    add_noise * torch.rand_like(self._robot.data.root_lin_vel_b) * 0.2 - 0.1),
            self._robot.data.root_ang_vel_b + (
                    add_noise * torch.rand_like(self._robot.data.root_ang_vel_b) * 0.4 - 0.2),
            self._robot.data.projected_gravity_b + (
                    add_noise * torch.rand_like(self._robot.data.projected_gravity_b) * 0.1 - 0.05),
        ], dim=-1)
        observations = {"policy": obs}
        observations["goal"] = goal
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)

        # base height
        des_height = 0.4
        if self.reward_type == "default":
            des_height = 0.3
        base_height_error = torch.square(self._robot.data.root_pos_w[:, 2] - des_height)

        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
                torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # Gait-specific rewards
        if self.reward_type == "trot":
            gait_symmetry = (0.5 * torch.logical_not(
                torch.logical_xor(first_contact[:, 0], first_contact[:, 3])).double()
                             + 0.5 * torch.logical_not(
                        torch.logical_xor(first_contact[:, 1], first_contact[:, 2])).double())
            gait_symmetry2 = (0.5 * torch.logical_xor(first_contact[:, 0], first_contact[:, 1]).double() +
                              0.5 * torch.logical_xor(
                        first_contact[:, 2], first_contact[:, 3]).double())
            gait_symmetry = 0.5 * gait_symmetry + 0.5 * gait_symmetry2
        elif self.reward_type == "pace":
            gait_symmetry = (0.5 * torch.logical_not(
                torch.logical_xor(first_contact[:, 0], first_contact[:, 2])).double()
                             + 0.5 * torch.logical_not(
                        torch.logical_xor(first_contact[:, 1], first_contact[:, 3])).double())
            gait_symmetry2 = (0.5 * torch.logical_xor(first_contact[:, 0], first_contact[:, 1]).double() +
                              0.5 * torch.logical_xor(
                        first_contact[:, 2], first_contact[:, 3]).double())
            gait_symmetry = 0.5 * gait_symmetry + 0.5 * gait_symmetry2
        elif self.reward_type == "default":
            gait_symmetry = torch.zeros_like(lin_vel_error_mapped)
        else:
            raise ValueError(f"Unknown rewrad type: {self.reward_type}")

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "base_height_l2": base_height_error * self.cfg.base_height_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "gait_symmetry": gait_symmetry * self.cfg.gait_symmetry_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out)
        if self.use_termination:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0,
                             dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        # if len(env_ids) == self.num_envs:
        # Spread out the resets to avoid spikes in training when many environments reset at a similar time
        # self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self.episode_length_buf[env_ids] = 0
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        self._commands[env_ids] = torch.ones_like(self._commands[env_ids]) * self.desired_velocity
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 0] += torch.rand_like(default_root_state[:, 0]) * 1.0 - 0.5
        default_root_state[:, 1] += torch.rand_like(default_root_state[:, 2]) * 1.0 - 0.5

        orientations_delta = math_utils.quat_from_euler_xyz(torch.zeros_like(default_root_state[:, 0]),
                                                            torch.zeros_like(default_root_state[:, 0]),
                                                            torch.rand_like(default_root_state[:, 0]) * 2 * 3.14 - 3.14)
        default_root_state[:, 3:7] = math_utils.quat_mul(default_root_state[:, 3:7], orientations_delta)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
