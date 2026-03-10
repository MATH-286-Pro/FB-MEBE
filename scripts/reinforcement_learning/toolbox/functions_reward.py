# 架构思考
# CMD Sampler 需要和 Reward Function 保持一致
# 1.使用 Dict 还是 Tensor 的 RAW 数据
# 2.goal 中使用 pitch, roll 还是 gx, gy, gz

# 架构思考
# 1.必须使用 Dict 保证后续 Joint Pos Reach 的扩展性
# 2.为了与大部分 四足 RL 保持一致，还是使用 gx gy gz


import torch
import typing as tp
from collections import defaultdict
from .config_task import TASK_CFG, DEFAULT_COMMAND

def _format_value(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if "." in text else f"{text}.0"


class CMDSampler:
    
    def __init__(self, device: str = "cpu"):
        self.device = device

    def _build_default_command(self, num_sample: int) -> dict[str, torch.Tensor]:
        command = {}
        for key, value in DEFAULT_COMMAND.items():
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            command[key] = (
                tensor.repeat(num_sample).unsqueeze(-1)
                if tensor.ndim == 0
                else tensor.unsqueeze(0).expand(num_sample, -1).clone()
            )
        return command

    def _build_log_index(
        self,
        mode: str,
        num_sample: int,
        sample_keys: list[str],
        sampled_values: torch.Tensor,
    ) -> dict[str, list[int]]:
        
        if mode == "RANDOM":
            return {"random": list(range(num_sample))}

        if mode == "LIST":
            log_index: dict[str, list[int]] = defaultdict(list)
            for env_idx, row in enumerate(sampled_values.detach().cpu().tolist()):
                name = "[" + ", ".join(
                    f"{key}={_format_value(float(row[i]))}"
                    for i, key in enumerate(sample_keys)
                ) + "]"
                log_index[name].append(env_idx)
            return dict(log_index)

    def sample(
        self,
        num_sample: int,
        task: str = "locomotion",
        mode: str = "list",
    ) -> dict[str, tp.Any]:
        
        task_cfg = getattr(TASK_CFG(), task)
        mode = mode.upper()

        match mode:
            
            # sample_keys   = ["vx", "vy", "wz"]
            # sample_values = torch.Tensor([[0.0, 0.0, 0.0],
            #                               [0.5, 0.0, 0.0],
            #                               [1.0, 0.0, 0.0],])

            case "LIST":
                sample_keys = task_cfg.LIST[0]
                candidates  = torch.tensor(task_cfg.LIST[1], dtype=torch.float32, device=self.device)                   # partial command
                sampled_values = candidates[torch.randint(0, candidates.shape[0], (num_sample,), device=self.device)]   # sampled partial command
            
            case "RANDOM":
                sample_keys = task_cfg.RANDOM[0]
                value_range = torch.tensor(task_cfg.RANDOM[1], dtype=torch.float32, device=self.device)
                low  = torch.minimum(value_range[0], value_range[1]) 
                high = torch.maximum(value_range[0], value_range[1])
                sampled_values = (torch.rand((num_sample, len(sample_keys)), device=self.device) * (high - low) + low)

        command_partial = {
            key: sampled_values[:, i].unsqueeze(-1)
            for i, key in enumerate(sample_keys)
        }

        command: dict[str, tp.Any] = self._build_default_command(num_sample)
        command.update(command_partial)
        command["log_index"] = self._build_log_index(mode, num_sample, sample_keys, sampled_values)

        return command

class RewardFunction:
    def __init__(self, inference_chunk_size: int = 256):
        self.inference_chunk_size = inference_chunk_size

    # 需要广播
    @torch.no_grad()
    def inference(self, obs: dict[str, torch.Tensor], cmd: dict[str, torch.Tensor], task: str) -> torch.Tensor:
        return reward_fn(obs, cmd, task, broadcast=True, chunk_size=self.inference_chunk_size)

    # 不用广播
    @torch.no_grad()
    def eval(self, obs: dict[str, torch.Tensor], cmd: dict[str, torch.Tensor], task: str) -> torch.Tensor:
        return reward_fn(obs, cmd, task, broadcast=False)


def reward_fn(obs: dict[str, torch.Tensor], 
              cmd: dict[str, torch.Tensor],
              task: str = "locomotion",
              broadcast: bool = True,
              chunk_size: int = 256) -> torch.Tensor:
    """
    obs: 多观测
    cmd: 多命令
    task: 任务类型
    broadcast: 是否需要广播
        如果为推理 = 需要广播   (有chunk计算) => 返回 [num_inference_sample, num_envs]
        如果为评估 = 不需要广播 (无chunk计算) => 返回 [num_envs]
    """

    def _to_1d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            return x.unsqueeze(0)
        if x.ndim == 1:
            return x
        if x.ndim == 2 and x.shape[-1] == 1:
            return x.squeeze(-1)
        raise ValueError(f"Expected scalar/1D/(num_envs,1) tensor, got shape={tuple(x.shape)}")

    # Current state [E]
    current_wz = _to_1d(obs["wz"])
    current_lin_vel = torch.stack([_to_1d(obs["vx"]), _to_1d(obs["vy"]), _to_1d(obs["vz"])], dim=-1)   # [E, 3]
    current_gravity = torch.stack([_to_1d(obs["gx"]), _to_1d(obs["gy"]), _to_1d(obs["gz"])], dim=-1)   # [E, 3]
    current_height  = _to_1d(obs["base_height"]).unsqueeze(-1)                                         # [E, 1]

    # Target state [num_envs]
    target_wz = _to_1d(cmd["wz"])
    target_lin_vel = torch.stack([_to_1d(cmd["vx"]), _to_1d(cmd["vy"]), _to_1d(cmd["vz"])], dim=-1)     # [num_envs, 3]
    target_gravity = torch.stack([_to_1d(cmd["gx"]), _to_1d(cmd["gy"]), _to_1d(cmd["gz"])], dim=-1)     # [num_envs, 3]
    target_height  = _to_1d(cmd["base_height"]).unsqueeze(-1)                                           # [num_envs, 1]

    match broadcast:

        # element-wise: [num_envs]
        case False:
            error_lin_vel = torch.norm(current_lin_vel - target_lin_vel, dim=-1)
            error_ang_vel = torch.abs(current_wz - target_wz)
            error_gravity = torch.norm(current_gravity - target_gravity, dim=-1)
            error_height  = torch.abs(current_height - target_height).squeeze(-1)

            reward_lin_vel = torch.exp(-torch.square(error_lin_vel / 0.3))
            reward_wz      = torch.exp(-torch.square(error_ang_vel / 0.2))
            reward_gravity = torch.exp(-torch.square(error_gravity / 0.1))
            reward_height  = torch.exp(-torch.square(error_height / 0.05))

            return reward_lin_vel * reward_wz * reward_gravity * reward_height

        # broadcast + chunk: return [num_envs, E]
        case True:
            num_inference = target_wz.shape[0]
            outputs = []
            # compute in chunks to avoid OOM
            for start in range(0, num_inference, chunk_size):
                end = min(start + chunk_size, num_inference)
                lin_chunk = target_lin_vel[start:end]   # [C, 3]
                wz_chunk  = target_wz[start:end]        # [C]
                g_chunk   = target_gravity[start:end]   # [C, 3]
                h_chunk   = target_height[start:end]    # [C, 1]

                error_lin_vel = torch.norm(current_lin_vel.unsqueeze(0) - lin_chunk.unsqueeze(1), dim=-1)       # [C, E]
                error_ang_vel = torch.abs(current_wz.unsqueeze(0) - wz_chunk.unsqueeze(1))                      # [C, E]
                error_gravity = torch.norm(current_gravity.unsqueeze(0) - g_chunk.unsqueeze(1), dim=-1)         # [C, E]
                error_height  = torch.abs(current_height.unsqueeze(0) - h_chunk.unsqueeze(1)).squeeze(-1)       # [C, E]

                reward_chunk = (
                    torch.exp(-torch.square(error_lin_vel / 0.3))
                    * torch.exp(-torch.square(error_ang_vel / 0.2))
                    * torch.exp(-torch.square(error_gravity / 0.1))
                    * torch.exp(-torch.square(error_height / 0.05))
                )  # [C, E]
                outputs.append(reward_chunk)

            return torch.cat(outputs, dim=0)
