import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from toolbox.dataclass_pylance import MODEL_CFG
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from . import fb_meta_module as meta
import math


########################################################################################################################
class eval_mode:
    def __init__(self, *models) -> None:
        self.models = models
        self.prev_states = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


class NORMALIZER_CLASS(nn.BatchNorm1d):
    @property
    def mean(self):
        return self.running_mean
    
    @property
    def std(self):
        return torch.sqrt(self.running_var)

########################################################################################################################
class FBModel(nn.Module):

    def __init__(self, model_cfg: MODEL_CFG):

        super().__init__()

        self.model_cfg = model_cfg

        self.policy_dim = model_cfg.policy_dim
        self.obs_dim    = model_cfg.obs_dim
        self.goal_dim   = model_cfg.goal_dim
        self.critic_dim = model_cfg.critic_dim
        self.action_dim = model_cfg.action_dim

        # obs normalizer
        self._policy_normalizer = NORMALIZER_CLASS(self.policy_dim, affine=False, momentum=model_cfg.momentum).to(model_cfg.device) if model_cfg.norm_obs else nn.Identity().to(model_cfg.device)
        self._F_normalizer      = NORMALIZER_CLASS(self.obs_dim,    affine=False, momentum=model_cfg.momentum).to(model_cfg.device) if model_cfg.norm_obs else nn.Identity().to(model_cfg.device)
        self._B_normalizer      = NORMALIZER_CLASS(self.goal_dim,   affine=False, momentum=model_cfg.momentum).to(model_cfg.device) if model_cfg.norm_obs else nn.Identity().to(model_cfg.device)
        self._critic_normalizer = NORMALIZER_CLASS(self.critic_dim, affine=False, momentum=model_cfg.momentum).to(model_cfg.device) if model_cfg.norm_obs else nn.Identity().to(model_cfg.device)

        # Forward Network
        self._forward_map    = meta.ForwardMap(obs_dim=self.obs_dim, z_dim = model_cfg.archi.z_dim, action_dim = self.action_dim, archi = model_cfg.archi.f).to(model_cfg.device)

        # Backward Network
        self._backward_map   = meta.BackwardMap(goal_dim = self.goal_dim, z_dim = model_cfg.archi.z_dim, archi = model_cfg.archi.b).to(model_cfg.device)

        # Actor Network
        self._actor = meta.Actor(obs_dim     = self.policy_dim, 
                                 z_dim       = model_cfg.archi.z_dim, 
                                 action_dim  = self.action_dim, 
                                 archi       = model_cfg.archi.actor).to(model_cfg.device)

        # Critic Network
        if self.model_cfg.archi.critic.enable:
            self._critic = meta.ForwardMap(obs_dim=self.critic_dim, z_dim = model_cfg.archi.z_dim, action_dim = self.action_dim, archi = model_cfg.archi.critic).to(model_cfg.device)
            self._target_critic = copy.deepcopy(self._critic)
            self._target_critic.load_state_dict(self._critic.state_dict())

        self.train(False)
        self.requires_grad_(False)
        self.to(self.model_cfg.device)

    ####################################################################################################################
    def _prepare_for_train(self) -> None:
        # 创建 TARGET networks
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map  = copy.deepcopy(self._forward_map)

        self._target_forward_map.load_state_dict(self._forward_map.state_dict())
        self._target_backward_map.load_state_dict(self._backward_map.state_dict())

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs) # type: ignore
        if device is not None:
            self.model_cfg.device = device.type  # type: ignore
        return super().to(*args, **kwargs)
    

    ###################### 便捷方法 ####################
    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.model_cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.model_cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    ################ 网络相关 #################
    def _policy_normalize(self, policy: torch.Tensor):
        with torch.no_grad(), eval_mode(self._policy_normalizer):
            return self._policy_normalizer(policy)

    def _F_normalize(self, obs: torch.Tensor):
        with torch.no_grad(), eval_mode(self._F_normalizer):
            return self._F_normalizer(obs)
    
    def _B_normalize(self, goal: torch.Tensor):
        with torch.no_grad(), eval_mode(self._B_normalizer):
            return self._B_normalizer(goal)
    
    def _critic_normalize(self, critic: torch.Tensor):
        with torch.no_grad(), eval_mode(self._critic_normalizer):
            return self._critic_normalizer(critic)

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        """
        Args:
            obs: original obs, not normalized
        Returns:
            return self._forward_map(self._normalize(obs))
        """
        return self._forward_map(self._F_normalize(obs), z, action)

    @torch.no_grad()
    def target_forward_map(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        return self._target_forward_map(self._F_normalize(obs), z, action)

    @torch.no_grad()
    def backward_map(self, goal: torch.Tensor):
        """
        Args:
            obs: original obs, not normalized
        Returns:
            return self._backward_map(self._normalize(obs))
        """
        return self._backward_map(self._B_normalize(goal))
    
    @torch.no_grad()
    def target_backward_map(self, obs: torch.Tensor):
        return self._target_backward_map(self._B_normalize(obs))

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, std: float):
        """
        Args:
            obs: original obs, not normalized
        Returns:
            return self._actor(self._normalize(obs), z, std)
        """
        return self._actor(self._policy_normalize(obs), z, std)
    
    @torch.no_grad()
    def critic(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        """
        Args:
            obs: original obs, not normalized
        Returns:
            return self._critic(self._normalize(obs), z, action)
        """
        return self._critic(self._critic_normalize(obs), z, action)

    @torch.no_grad()
    def target_critic(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        return self._target_critic(self._critic_normalize(obs), z, action)

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        """
        Args:
            obs: original obs, not normalized
        """
        dist = self.actor(obs, z, self.model_cfg.actor_std)
        if mean:
            return dist.mean
        return dist.sample()