import torch
from functools import wraps


def reward_fn(obs: dict[str, torch.Tensor], 
              command: dict[str, torch.Tensor]) -> torch.Tensor:

    current_lin_vel = obs["lin_vel"]
    current_ang_vel = obs["ang_vel"][:,2]
    current_gravity = obs["gravity"]
    current_height  = obs["base_height"]

    target_lin_vel  = command["lin_vel"]
    target_ang_vel  = command["ang_vel"][2]
    target_gravity  = command["gravity"]
    target_height   = command["height"]

    error_lin_vel = torch.norm(current_lin_vel - target_lin_vel, dim=-1)
    error_ang_vel = torch.abs(current_ang_vel - target_ang_vel)
    error_gravity = torch.norm(current_gravity - target_gravity, dim=-1)
    error_height  = torch.abs(current_height - target_height).squeeze(-1)

    lin_vel_reward = torch.exp(-torch.square(error_lin_vel / 0.3))
    ang_vel_reward = torch.exp(-torch.square(error_ang_vel / 0.2))
    gravity_reward = torch.exp(-torch.square(error_gravity / 0.1))
    height_reward  = torch.exp(-torch.square(error_height / 0.05))

    task_rewards = {
        "lin_vel_reward": lin_vel_reward,
        "ang_vel_reward": ang_vel_reward,
        "gravity_reward": gravity_reward,
        "height_reward":  height_reward,
    }

    task_rewards = torch.prod(torch.stack(list(task_rewards.values())), dim=0)

    return task_rewards