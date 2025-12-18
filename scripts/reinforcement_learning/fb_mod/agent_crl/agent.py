import copy
import math
from collections import OrderedDict, defaultdict
import typing as tp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

################# USER INCLUDE ##################
import copy
from . import utils
from . import fb_meta_module as meta

from .model import FBModel, eval_mode
from .weight import weight_init

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from toolbox.dataclass_pylance import AGENT_CFG

def get_targets_uncertainty(
        preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),
            )
            / num_parallel_scaling
        )
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

class FB_CRL_AGENT:

    def __init__(self, agent_cfg: AGENT_CFG):

        self.agent_cfg = agent_cfg
        self._model = FBModel(agent_cfg.model)
        self.setup_training()
        self.setup_compile()
        self._model.to(self.agent_cfg.model.device)

    @property
    def device(self):
        return self._model.cfg.device

    def setup_compile(self):
        print(f"compile {self.agent_cfg.compile}")
        mode = "reduce-overhead"
        if self.agent_cfg.compile:
            self.update_fb      = torch.compile(self.update_fb, mode=mode)
            self.update_actor   = torch.compile(self.update_actor, mode=mode)
            self.sample_mixed_z = torch.compile(self.sample_mixed_z, mode=mode)
            if self.agent_cfg.model.archi.critic.enable:
                self.update_critic  = torch.compile(self.update_critic, mode=mode)

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # type: ignore  # ensure that target nets are initialized after applying the weights

        # optimizers
        self.forward_opt  = torch.optim.Adam(self._model._forward_map.parameters(),  lr=self.agent_cfg.train.lr_f)
        self.backward_opt = torch.optim.Adam(self._model._backward_map.parameters(), lr=self.agent_cfg.train.lr_b)
        self.actor_opt    = torch.optim.Adam(self._model._actor.parameters(),        lr=self.agent_cfg.train.lr_actor)

        if self.agent_cfg.model.archi.critic.enable:
            self.Q_reg_opt = torch.optim.Adam(self._model._critic.parameters(), lr=self.agent_cfg.train.lr_critic)

        ######## USER DEFINE BEGIN ########
        self.off_diag = 1 - torch.eye(self.agent_cfg.train.batch_size, self.agent_cfg.train.batch_size, device=self.agent_cfg.model.device)
        self.off_diag_sum = self.off_diag.sum()
        ######## USER DEFINE END ########


    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)
    

    ################################################### z 相关 #################################################################
    @torch.no_grad()
    def sample_mixed_z(self, NEXT_GOAL: dict | None = None) -> torch.Tensor:

        # 原始 z
        z = self._model.sample_z(self.agent_cfg.train.batch_size, device=self.agent_cfg.model.device)

        # 混合 z        
        if self.agent_cfg.train.train_goal_ratio > 0:
            perm = torch.randperm(self.agent_cfg.train.batch_size, device=self.agent_cfg.model.device)
            z_B = self._model._backward_map(NEXT_GOAL['goal'][perm])
            z_B = F.normalize(z_B, dim=-1) * math.sqrt(self.agent_cfg.model.archi.z_dim)
            mask = torch.rand((self.agent_cfg.train.batch_size, 1), device=self.agent_cfg.model.device) < self.agent_cfg.train.train_goal_ratio
            z = torch.where(mask, z_B, z)
        return z
    
    def refresh_z(self, z: torch.Tensor | None, step_count: torch.Tensor):
        
        # if first time, sample z
        # 如果是第一次，则直接采样
        if z is None:
            z = self._model.sample_z(step_count.shape[0], device=self.agent_cfg.model.device)
        
        # if not first time, update based on termination
        # 如果不是第一次，根据
        else:
            refresh_index = step_count % self.agent_cfg.train.update_z_every_step == 0
            new_z = self._model.sample_z(z.shape[0], device=self.agent_cfg.model.device)
            z = torch.where(refresh_index, new_z, z.to(self.agent_cfg.model.device))
        return z
    
    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.agent_cfg.model.archi.norm_z:
            z = F.normalize(z, dim=-1) * math.sqrt(self.agent_cfg.model.archi.z_dim)
        return z
    
    ################################ inference 函数 #########################################
    def reward_inference(self, obs: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: un-normalized observation
        """
        z_B = self._model.backward_map(obs)
        z_r = torch.matmul(reward.T, z_B)
        z_r = self.project_z(z_r)
        return z_r

    def goal_inference(self, goal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            goal: un-normalized goal
        """
        z_goal = self._model.backward_map(goal)
        z_goal = self.project_z(z_goal)
        return z_goal

    ###################################################### pretrain 相关 ##############################################################
    def update(self, relpay_buffer, step: int) -> tp.Dict[str, float]:

        metrics: tp.Dict[str, torch.Tensor] = {}

        # 获取一个 batch 的数据
        batch = relpay_buffer['train'].sample(self.agent_cfg.train.batch_size)

        obs        = batch["observation"]
        action     = batch["action"]
        next_obs   = batch["next"]["observation"]
        terminated = batch["next"]["terminated"]

        next_reward_reg = batch["next"]["reward_reg"]

        discount = self.agent_cfg.train.discount * ~terminated

        # Update normalizer
        self._model._policy_normalizer(obs['policy'])
        self._model._F_normalizer(obs['obs'])
        self._model._B_normalizer(obs['goal'])
        self._model._critic_normalizer(obs['critic'])

        self._model._policy_normalizer(next_obs['policy'])
        self._model._F_normalizer(next_obs['obs'])
        self._model._B_normalizer(next_obs['goal'])
        self._model._critic_normalizer(next_obs['critic'])
        
        # 计算 normalized data
        OBS = {
            'policy': self._model._policy_normalize(obs['policy']),
            'obs':    self._model._F_normalize(obs['obs']),
            'goal':   self._model._B_normalize(obs['goal']),
            'critic': self._model._critic_normalize(obs['critic']),
        }

        NEXT_OBS = {
            'policy': self._model._policy_normalize(next_obs['policy']),
            'obs':    self._model._F_normalize(next_obs['obs']),
            'goal':   self._model._B_normalize(next_obs['goal']),
            'critic': self._model._critic_normalize(next_obs['critic']),
        }

        # 采样 z
        torch.compiler.cudagraph_mark_step_begin()
        z = self.sample_mixed_z(NEXT_GOAL=NEXT_OBS).clone()

        q_loss_coef = self.agent_cfg.train.q_loss_coef if self.agent_cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.agent_cfg.train.clip_grad_norm if self.agent_cfg.train.clip_grad_norm > 0 else None

        # torch.compiler.cudagraph_mark_step_begin()
        metrics.update(self.update_fb(
            OBS=OBS, 
            action=action, 
            discount=discount,
            NEXT_OBS=NEXT_OBS, 
            NEXT_GOAL=NEXT_OBS, 
            z=z, 
            step=step,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm))

        # update critic
        if self.agent_cfg.model.archi.critic.enable:
            metrics.update(self.update_critic(OBS=OBS, action=action, discount=discount,
                                                NEXT_OBS=NEXT_OBS, z=z, rew_reg=next_reward_reg))

        # update actor
        metrics.update(self.update_actor(OBS, z, step, clip_grad_norm))

        # update critic target
        with torch.no_grad():
            utils.soft_update_params(self._model._forward_map, self._model._target_forward_map, self.agent_cfg.train.fb_target_tau)
            utils.soft_update_params(self._model._backward_map, self._model._target_backward_map, self.agent_cfg.train.fb_target_tau)
            # Update critic target networks
            if self.agent_cfg.model.archi.critic.enable:
                utils.soft_update_params(self._model._critic, self._model._target_critic, self.agent_cfg.train.critic_target_tau)


        return metrics


    def update_fb(
        self,
        OBS: dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        NEXT_OBS: dict[str, torch.Tensor],
        NEXT_GOAL: dict[str, torch.Tensor],
        z: torch.Tensor,
        step: int,
        q_loss_coef: float | None,
        clip_grad_norm: float | None,
    ) -> tp.Dict[str, torch.Tensor]:
        
        # compute target successor measure
        with torch.no_grad():
            # stddev = utils.schedule(self.cfg.stddev_schedule, step)
            # stddev = 0.2
            dist        = self._model._actor(NEXT_OBS['policy'], z, self.agent_cfg.model.actor_std)
            next_action = dist.sample(clip=self.agent_cfg.train.stddev_clip)
            target_Fs   = self._model._target_forward_map(NEXT_OBS['obs'], z, next_action)
            target_B    = self._model._target_backward_map(NEXT_GOAL['goal'])
            target_Ms   = torch.matmul(target_Fs, target_B.T)
            _, _, target_M    = get_targets_uncertainty(target_Ms, pessimism_penalty=self.agent_cfg.train.fb_pessimism_penalty)

        Fs = self._model._forward_map(OBS['obs'], z, action)  # e x batch x z_dim     #TODO
        B  = self._model._backward_map(NEXT_GOAL['goal'])        # batch x z_dim      #TODO
        Ms = torch.matmul(Fs, B.T)                     # e x batch x batch

        diff = Ms - discount * target_M
        fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
        fb_diag    = -torch.diagonal(Ms, dim1=1, dim2=2).mean() * Ms.shape[0]          

        fb_loss = fb_offdiag + fb_diag

        # orthonormality loss for backward embedding
        Cov = torch.matmul(B, B.T)
        orth_loss_diag = -Cov.diag().mean()
        orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.agent_cfg.train.ortho_coef * orth_loss
        
        # 这个是 META 新加的 FB loss
        # 参考论文 Algorithm Details 20 页
        # Paper: Algorithm Details page 20 
        fb_q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
        if q_loss_coef is not None:
            with torch.no_grad():

                # 这里的 z 是在 update 中使用 sample_mixed_z 采样出来的

                # Part 1: Z_F(obs) and z
                next_Qs = (target_Fs * z).sum(dim=-1)  # num_parallel x batch
                _, _, next_Q = get_targets_uncertainty(next_Qs, pessimism_penalty=self.agent_cfg.train.fb_pessimism_penalty)  #0000ff

                # Part 2: Z_B(goal) and z
                cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                inv_cov = torch.inverse(cov)  # z_dim x z_dim
                implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=-1)  # batch

                # Q_pred = R + gamma * Q'
                target_Q = implicit_reward.detach() + discount.squeeze() * next_Q  # batch
                expanded_targets = target_Q.expand(Fs.shape[0], -1)

            Qs = (Fs * z).sum(dim=-1)  # num_parallel x batch
            fb_q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)  # (Qs - Qs_pred)^2
            fb_loss += self.agent_cfg.train.q_loss_coef * fb_q_loss

        # optimize FB
        self.forward_opt.zero_grad(set_to_none=True)
        self.backward_opt.zero_grad(set_to_none=True)
        fb_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)
        self.forward_opt.step()
        self.backward_opt.step()

        with torch.no_grad():
            metrics = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B.mean(),
                # "B_norm": torch.norm(B, dim=-1).mean(),
                # "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss":    fb_loss,
                "fb_diag":    fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss":  orth_loss,
                "orth_loss_diag":    orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "fb_q_loss":         fb_q_loss
            }

        return metrics

    def update_actor(self, OBS: torch.Tensor, z: torch.Tensor, step: int, clip_grad_norm: float | None) -> tp.Dict[str, torch.Tensor]:

        metrics: tp.Dict[str, torch.Tensor] = {}
        # stddev = utils.schedule(self.cfg.stddev_schedule, step)
        # stddev = 0.2
        dist = self._model._actor(OBS['policy'], z, self.agent_cfg.model.actor_std)
        action = dist.sample(clip=self.agent_cfg.train.stddev_clip)  # uses overriden method thats is differentiable 可导

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        # Critic FB
        Fs = self._model._forward_map(OBS['obs'], z, action)
        Q_fbs = (Fs * z).sum(-1)
        _, _, Q_fb  = get_targets_uncertainty(preds=Q_fbs, pessimism_penalty=self.agent_cfg.train.actor_pessimism_penalty)

        actor_loss = -Q_fb.mean()

        # Critic regularizer - 双网络
        if self.agent_cfg.model.archi.critic.enable: 
            Q_regs = self._model._critic(OBS['critic'], z, action)
            _, _, Q_reg = get_targets_uncertainty(Q_regs, self.agent_cfg.train.critic_pessimism_penalty)

            # imitate META-motivo to add changable weight
            weight = Q_fb.abs().mean().detach() 
            actor_loss -= self.agent_cfg.train.reg_coeff * Q_reg.mean() * weight
            
            metrics['actor_loss_reg'] = -self.agent_cfg.train.reg_coeff * weight * Q_reg.mean().detach()
            metrics['Q_reg'] = Q_reg.mean().detach()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.detach()   # use detach() instead of item()
        metrics['actor_loss_fb'] = -Q_fb.mean().detach()  #ff0000
        metrics['actor_logprob'] = log_prob.mean().detach()
        # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics


    def update_critic(self, 
                        OBS: torch.Tensor, 
                        action: torch.Tensor, 
                        discount: torch.Tensor, 
                        NEXT_OBS: torch.Tensor, 
                        z: torch.Tensor, 
                        rew_reg: torch.Tensor): #0000ff
            
        # 当前 Q 值 - 双网络
        Q_regs = self._model._critic(OBS['critic'], z, action)
        num_parallel = Q_regs.shape[0]

        # 目标 Q 值 - 使用双网络的最小值作为目标
        with torch.no_grad():
            dist = self._model._actor(NEXT_OBS['policy'], z, std=self.agent_cfg.model.actor_std)
            next_action = dist.sample(clip=self.agent_cfg.train.stddev_clip)                                        

            next_Q_regs = self._model._target_critic(NEXT_OBS['critic'], z, next_action)
            _, _, next_Q_reg = get_targets_uncertainty(next_Q_regs, self.agent_cfg.train.critic_pessimism_penalty)

            target_Q = rew_reg + discount * next_Q_reg
            expanded_targets = target_Q.expand(num_parallel, -1, -1)  

        
        # TD 损失 - 分别计算两个网络的损失
        # compute critic loss
        Q_reg_loss = 0.5 * num_parallel * F.mse_loss(Q_regs, expanded_targets)

        # optimize critic 优化
        self.Q_reg_opt.zero_grad()
        Q_reg_loss.backward()
        self.Q_reg_opt.step()

        metrics = {}
        metrics['Critic_reg_loss'] = Q_reg_loss.detach()
        return metrics


    def save(self, path: str):
        torch.save(
            {'actor':                self._model._actor.state_dict(),
                'policy_normalizer': self._model._policy_normalizer.state_dict(),
                'B':                 self._model._backward_map.state_dict(),
                'B_normalizer':      self._model._B_normalizer.state_dict(),},
            path
        )
        print(f"Model saved to {path}")