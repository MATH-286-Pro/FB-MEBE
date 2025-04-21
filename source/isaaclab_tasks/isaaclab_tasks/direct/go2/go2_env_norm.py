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

from .go2_norm_cfg import Go2FlatEnvNormCfg


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
        self._commands = torch.zeros(self.num_envs, 6, device=self.device)
        self.desired_velocity = torch.tensor([0.5, 0, 0], device=self.device)
        self.desired_base_tilt = torch.tensor([0, 0, -1], device=self.device)
        self.goal_space_type = "basic"
        self.task_reward = "reg_locomotion"  # make this default for training with regularizer

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [

                "track_lin_vel_xy",
                "track_ang_vel_z",
                "lin_z_rew",
                "ang_xy_rew",
                "action_rate_rew",
                "joint_torques_rew",
                "air_time",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._other_id, _ = self._contact_sensor.find_bodies("Head_.*")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        self._calf_ids, _ = self._contact_sensor.find_bodies(".*calf")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*thigh")

        # Set up the environment
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
        goal = self._get_goal(obs)

        observations = {"policy": obs}
        observations["goal"] = goal
        return observations

    def _get_goal(self, obs) -> torch.Tensor:
        if self.goal_space_type == "basic":
            goal_dim = self._robot.data.root_lin_vel_b.shape[1] + self._robot.data.root_ang_vel_b.shape[1] +\
                self._robot.data.projected_gravity_b.shape[1]
            goal = obs[:, :goal_dim]  # TODO a bit hard coded!
        else:
            return NotImplementedError
        return goal

    def _get_rewards(self) -> tuple[torch.Tensor, dict]:

        task_rewards, regularization_rewards = self._get_locomotion_rewards()
        all_rewards = task_rewards | regularization_rewards
        if self.task_reward == 'locomotion':
            reward = 1.0 * torch.prod(torch.stack(list(task_rewards.values())), dim=0) + \
                0.5 * torch.prod(torch.stack(list(regularization_rewards.values())), dim=0)

        elif self.task_reward == 'reg_locomotion':
            reward = torch.prod(torch.stack(list(regularization_rewards.values())), dim=0)
        
        else:
            raise ValueError(f"Unknown reward type: {self.task_reward}")

        rew_dict = dict()
        # Logging
        for key, value in all_rewards.items():
            self._episode_sums[key] += value
            rew_dict["Step_Reward/" + key] = torch.mean(value)
        return reward, rew_dict

    def _get_locomotion_rewards(self) -> (dict[str, torch.Tensor], dict[str, torch.Tensor]):
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

        # regularization
        action_rate = torch.norm(self._actions - self._previous_actions, dim=1)
        action_rate_rew = torch.exp(-torch.square(action_rate / 4.0))

        joint_torques = torch.norm(self._robot.data.applied_torque, dim=1)
        joint_torques_rew = torch.exp(-torch.square(joint_torques / 20))

        zero_command = torch.norm(self._commands[:, :3], dim=1) < 0.1
        # foot_pos_z = self._robot.data.body_pos_w[:, self._feet_ids, 2]
        # foot_height_error = torch.square(foot_pos_z - 0.10)
        # foot_vel_xy = self._robot.data.body_vel_w[:, self._feet_ids, :2]
        # foot_vel_xy_square = torch.square(torch.norm(foot_vel_xy, dim=-1))
        # foot_clearance_error = torch.sum(foot_height_error * foot_vel_xy_square, dim=1)
        # foot_clearance = 1.0 * zero_command + torch.exp(-torch.square(foot_clearance_error / 0.01)) * ~zero_command

        # if currently in contact + contact happen less than (dt) seconds ago
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        # time spent in the air, before the last contact (if currently in the air it will still account for the prev )
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time) * first_contact, dim=1)
        air_time_rew = torch.sigmoid(10 * air_time) * ~zero_command + 1.0 * zero_command

        task_rewards = {
            "track_lin_vel_xy": lin_vel_xy_rew,
            "track_ang_vel_z": ang_vel_z_rew,
            "lin_z_rew": lin_z_rew,
            "ang_xy_rew": ang_xy_rew,
        }

        regularization_rewards = {
            "action_rate_rew": action_rate_rew,
            "joint_torques_rew": joint_torques_rew,
            # "foot_clearance": foot_clearance,
            "air_time": air_time_rew,
        }

        rewards = {
            "track_lin_vel_xy": lin_vel_xy_rew ,
            "track_ang_vel_z": ang_vel_z_rew ,
            "lin_z_rew": lin_z_rew ,
            "ang_xy_rew": ang_xy_rew,
            # "lin_vel_z_rew": lin_vel_z_rew * self.cfg.reward_scale["lin_vel_z_rew"],
            # "ang_vel_xy_rew": ang_vel_xy_rew * self.cfg.reward_scale["ang_vel_xy_rew"],
            "action_rate_rew": action_rate_rew,
            "joint_torques_rew": joint_torques_rew,
            # "foot_clearance": foot_clearance * self.cfg.reward_scale["foot_clearance"],
            "air_time": air_time_rew,
        }

        return task_rewards, regularization_rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out)
        if self.use_termination:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history  # type: ignore
            died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
            died2 = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._other_id], dim=-1), dim=1)[0] > 1.0, dim=1)
            died3 = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._thigh_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
            died4 = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._calf_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
            died = died | died2
            died = died | died3
            died = died | died4
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
        self._commands[env_ids] = torch.ones_like(self._commands[env_ids]) * torch.cat((self.desired_velocity, self.desired_base_tilt), dim=0)
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
            extras["Episode_Reward/" + key] = episodic_sum_avg
            self._episode_sums[key][env_ids] = 0.0
        extras["Num_env_resets"] = len(env_ids)
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
