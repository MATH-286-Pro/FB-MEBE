import torch
from functools import wraps


def reward_fn(obs: dict[str, torch.Tensor], 
              cmd: dict[str, torch.Tensor]) -> torch.Tensor:

    current_lin_vel = torch.cat([obs["vx"], obs["vy"], obs["vz"]], dim=-1)
    current_wz      = obs["wz"]
    current_gravity = torch.cat([obs["gx"], obs["gy"], obs["gz"]], dim=-1)
    current_height  = obs["base_height"]

    target_lin_vel  = torch.cat([cmd["vx"], cmd["vy"], cmd["vz"]], dim=-1)
    target_wz       = cmd["wz"]
    target_gravity  = torch.cat([cmd["gx"], cmd["gy"], cmd["gz"]], dim=-1)
    target_height   = cmd["base_height"]

    error_lin_vel = torch.norm(current_lin_vel - target_lin_vel[0], dim=-1)
    error_wz      = torch.abs(current_wz - target_wz[0]).squeeze(-1)
    error_gravity = torch.norm(current_gravity - target_gravity[0], dim=-1)
    error_height  = torch.abs(current_height - target_height[0]).squeeze(-1)

    reward_lin_vel = torch.exp(-torch.square(error_lin_vel / 0.3))
    reward_wz      = torch.exp(-torch.square(error_wz / 0.2))
    reward_gravity = torch.exp(-torch.square(error_gravity / 0.1))
    reward_height  = torch.exp(-torch.square(error_height / 0.05))

    task_rewards = {
        "reward_lin_vel": reward_lin_vel,
        "reward_wz":      reward_wz,
        "reward_gravity": reward_gravity,
        "reward_height":  reward_height,
    }

    task_rewards = torch.prod(torch.stack(list(task_rewards.values())), dim=0)

    return task_rewards