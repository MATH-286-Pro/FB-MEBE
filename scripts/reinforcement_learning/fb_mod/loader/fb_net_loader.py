import torch
import torch.nn as nn
from pathlib import Path
import omegaconf as omgcf
from . import nn_models as meta
import math
import torch.nn.functional as F

class FBPolicyLoader:

    def __init__(self, path: str, device: str = 'cpu'):
        self.device = str(torch.device(device))

        model_path = Path(path)
        config_path = model_path.parent.parent / 'hydra_config.yaml'
        
        # 用 OmegaConf 加载 hydra 配置文件
        hydra_config = omgcf.OmegaConf.load(str(config_path))
        omgcf.OmegaConf.resolve(hydra_config)

        # 获取配置
        self.agent_cfg = hydra_config.agent
        self.model_cfg = hydra_config.agent.model


        # 创建网络结构
        policy_dim = self.model_cfg.policy_dim
        goal_dim   = self.model_cfg.goal_dim
        action_dim = self.model_cfg.action_dim
        archi      = self.model_cfg.archi

        self._backward_map = meta.build_backward(goal_dim, archi.z_dim, archi.b)
        self._actor        = meta.build_actor(policy_dim, archi.z_dim, action_dim, archi.actor)
        self._policy_normalizer = nn.BatchNorm1d(policy_dim)
        self._B_normalizer   = nn.BatchNorm1d(goal_dim)

        # 加载 checkpoint
        self.checkpoint = torch.load(
            model_path,
            weights_only=True,
            map_location=self.device,
        )
        self._backward_map.load_state_dict(self.checkpoint['B'])
        self._actor.load_state_dict(self.checkpoint['actor'])
        
        # 对于 normalizer，只加载统计信息（running_mean, running_var）
        obs_norm_state = self.checkpoint['policy_normalizer']
        self._policy_normalizer.running_mean.copy_(obs_norm_state['running_mean'])
        self._policy_normalizer.running_var.copy_(obs_norm_state['running_var'])
        
        B_norm_state = self.checkpoint['B_normalizer']
        self._B_normalizer.running_mean.copy_(B_norm_state['running_mean'])
        self._B_normalizer.running_var.copy_(B_norm_state['running_var'])

        # 移动到指定设备
        self._backward_map   = self._backward_map.to(self.device)
        self._actor          = self._actor.to(self.device)
        self._policy_normalizer = self._policy_normalizer.to(self.device)
        self._B_normalizer   = self._B_normalizer.to(self.device)

        # 评估模式
        self._backward_map.eval()
        self._actor.eval()
        self._policy_normalizer.eval()
        self._B_normalizer.eval()
    
    # 访问接口
    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, std: float):
        return self._actor(self._policy_normalizer(obs), z, std)
    
    @torch.no_grad()
    def backward_map(self, goal: torch.Tensor):
        return self._backward_map(self._B_normalizer(goal))
    
    @torch.no_grad()
    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.model_cfg.actor_std)
        if mean:
            return dist.mean
        return dist.sample()
    
    @torch.no_grad()
    def __call__(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self.act(obs, z, mean)

    @torch.no_grad()
    def reward_inference(self, Z_Bs: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:

        if len(reward.shape) == 1:
            reward = reward.unsqueeze(-1)

        Z    = torch.matmul(reward.T, Z_Bs)
        Z    = math.sqrt(Z.shape[-1]) * F.normalize(Z, dim=-1) 
        return Z
    
    ###############################################################
    @torch.no_grad()
    def refresh_z(self, z: torch.Tensor | None, step_count: torch.Tensor) -> torch.Tensor:

        # get mask for environmets where we need to change z
        if z is not None:
            mask_reset_z = step_count % self.agent_cfg.train.update_z_every_step == 0
            new_z = self.sample_z(z.shape[0], device=self.device)
            z = torch.where(mask_reset_z, new_z, z.to(self.device))
        else:
            z = self.sample_z(step_count.shape[0], device=self.device)
        return z
    
    ######################## convinient tools ####################
    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.agent_cfg.model.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.agent_cfg.model.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z
