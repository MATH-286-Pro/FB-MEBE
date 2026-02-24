# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from .model import FBModel, config_from_dict
from ..nn_models import weight_init, _soft_update_params, eval_mode
from ..misc.zbuffer import ZBuffer
from pathlib import Path
import json
import safetensors

########### user define config start ############
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from toolbox.dataclass_pylance import AGENT_CFG, MODEL_CFG, TRAIN_CFG
########### user define config end   ############

torch._inductor.config.pattern_matcher = False

class FBAgent:
    def __init__(self, **kwargs):
        self.cfg = config_from_dict(kwargs, AGENT_CFG)
        self.cfg.train.fb_target_tau = float(min(max(self.cfg.train.fb_target_tau, 0), 1))
        self._model = FBModel(**dataclasses.asdict(self.cfg.model))
        self.setup_training()
        self.setup_compile()
        self._model.to(self.cfg.model.device)

    @property
    def device(self):
        return self._model.cfg.device

    
    def reward_inference(self, next_obs: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        return self._model.reward_inference(next_obs, reward, None)

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights

        self.backward_optimizer = torch.optim.Adam(
            self._model._backward_map.parameters(),
            lr=self.cfg.train.lr_b,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.forward_optimizer = torch.optim.Adam(
            self._model._forward_map.parameters(),
            lr=self.cfg.train.lr_f,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(),
            lr=self.cfg.train.lr_actor,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

        if self.cfg.model.archi.critic.enable:
            self.critic_optimizer = torch.optim.Adam(
                self._model._critic.parameters(),
                lr=self.cfg.train.lr_critic,
                capturable=self.cfg.cudagraphs and not self.cfg.compile,
                weight_decay=self.cfg.train.weight_decay,
            )

        # prepare parameter list
        self._forward_map_paramlist = tuple(x for x in self._model._forward_map.parameters())
        self._target_forward_map_paramlist = tuple(x for x in self._model._target_forward_map.parameters())
        self._backward_map_paramlist = tuple(x for x in self._model._backward_map.parameters())
        self._target_backward_map_paramlist = tuple(x for x in self._model._target_backward_map.parameters())

        if self.cfg.model.archi.critic.enable:
            self._critic_paramlist = tuple(x for x in self._model._critic.parameters())
            self._target_critic_paramlist = tuple(x for x in self._model._target_critic.parameters())

        # precompute some useful variables
        self.off_diag = 1 - torch.eye(self.cfg.train.batch_size, self.cfg.train.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

        self.z_buffer = ZBuffer(self.cfg.train.z_buffer_size, self.cfg.model.archi.z_dim, self.cfg.model.device)

    def setup_compile(self):
        print(f"compile {self.cfg.compile}")
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            print(f"compiling with mode '{mode}'")
            self.update_fb = torch.compile(self.update_fb, mode=mode)  # use fullgraph=True to debug for graph breaks
            self.update_actor = torch.compile(self.update_actor, mode=mode)  # use fullgraph=True to debug for graph breaks
            self.sample_mixed_z = torch.compile(self.sample_mixed_z, mode=mode, fullgraph=True)

            if self.cfg.model.archi.critic.enable:
                self.update_critic = torch.compile(self.update_critic, mode=mode)

        print(f"cudagraphs {self.cfg.cudagraphs}")
        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_fb = CudaGraphModule(self.update_fb, warmup=5)
            self.update_actor = CudaGraphModule(self.update_actor, warmup=5)

            if self.cfg.model.archi.critic.enable:
                self.update_critic = CudaGraphModule(self.update_critic, warmup=5)

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)

    @torch.no_grad()
    def sample_mixed_z(self, TRAIN_GOAL: dict[str, torch.Tensor] | None = None, *args, **kwargs):
        # samples a batch from the z distribution used to update the networks
        z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)

        if TRAIN_GOAL is not None:
            perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
            z_B = self._model._backward_map(TRAIN_GOAL['goal'][perm])
            z_B = self._model.project_z(z_B)
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) < self.cfg.train.train_goal_ratio
            z = torch.where(mask, z_B, z)
        return z

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        obs, action, next_obs, terminated, reward_reg = (
            batch["observation"],
            batch["action"],
            batch["next"]["observation"],
            batch["next"]["terminated"],
            batch["next"]["reward_reg"], # r(s,a) is the reg_reward of s', so should use next
        )
        discount = self.cfg.train.discount * ~terminated

        # use obs without noise to update normalizer
        self._model._policy_normalizer(obs['policy'])
        self._model._F_normalizer(obs['obs'])
        self._model._B_normalizer(obs['goal'])
        self._model._critic_normalizer(obs['critic'])

        self._model._policy_normalizer(next_obs['policy'])
        self._model._F_normalizer(next_obs['obs'])
        self._model._B_normalizer(next_obs['goal'])
        self._model._critic_normalizer(next_obs['critic'])

        OBS = {
            "policy": self._model._policy_normalize(obs['policy']),
            "obs"   : self._model._F_normalize(obs['obs']),
            "goal"  : self._model._B_normalize(obs['goal']),
            "critic": self._model._critic_normalize(obs['critic']),
        }
        NEXT_OBS = {
            "policy": self._model._policy_normalize(next_obs['policy']),
            "obs"   : self._model._F_normalize(next_obs['obs']),
            "goal"  : self._model._B_normalize(next_obs['goal']),
            "critic": self._model._critic_normalize(next_obs['critic']),
        }

        # OBS, NEXT_OBS these capitalized name means they are normalized

        torch.compiler.cudagraph_mark_step_begin()
        z = self.sample_mixed_z(TRAIN_GOAL=NEXT_OBS).clone()
        self.z_buffer.add(z)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        torch.compiler.cudagraph_mark_step_begin()
        metrics = self.update_fb(
            OBS=OBS,
            action=action,
            discount=discount,
            NEXT_OBS=NEXT_OBS,
            NEXT_GOAL=NEXT_OBS,
            z=z,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm,
        )

        if self.cfg.model.archi.critic.enable:
            metrics.update(
                self.update_critic(
                    OBS=OBS,
                    action=action,
                    discount=discount,
                    NEXT_OBS=NEXT_OBS,
                    z=z,
                    reward_reg=reward_reg,
                    clip_grad_norm=clip_grad_norm,
                )
            )

        metrics.update(
            self.update_actor(
                OBS=OBS,
                action=action,
                z=z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.cfg.train.fb_target_tau)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.cfg.train.fb_target_tau)

            if self.cfg.model.archi.critic.enable:
                _soft_update_params(self._critic_paramlist, self._target_critic_paramlist, self.cfg.train.fb_target_tau)

        return metrics

    def update_fb(
        self,
        OBS: dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        NEXT_OBS: dict[str, torch.Tensor],
        NEXT_GOAL: dict[str, torch.Tensor],
        z: torch.Tensor,
        q_loss_coef: float | None,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            dist = self._model._actor(NEXT_OBS['policy'], z, self._model.cfg.actor_std)
            next_action = dist.sample(clip=self.cfg.train.stddev_clip)
            target_Fs = self._model._target_forward_map(NEXT_OBS['obs'], z, next_action)  # num_parallel x batch x z_dim
            target_B = self._model._target_backward_map(NEXT_GOAL['goal'])  # batch x z_dim
            target_Ms = torch.matmul(target_Fs, target_B.T)  # num_parallel x batch x batch
            _, _, target_M = self.get_targets_uncertainty(target_Ms, self.cfg.train.fb_pessimism_penalty)  # batch x batch

        # compute FB loss
        Fs = self._model._forward_map(OBS['obs'], z, action)  # num_parallel x batch x z_dim
        B = self._model._backward_map(NEXT_GOAL['goal'])  # batch x z_dim
        Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch

        diff = Ms - discount * target_M  # num_parallel x batch x batch
        fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
        fb_diag = -torch.diagonal(Ms, dim1=1, dim2=2).mean() * Ms.shape[0]
        fb_loss = fb_offdiag + fb_diag

        # compute orthonormality loss for backward embedding
        Cov = torch.matmul(B, B.T)
        orth_loss_diag = -Cov.diag().mean()
        orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.cfg.train.ortho_coef * orth_loss

        q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
        if q_loss_coef is not None:
            with torch.no_grad():
                next_Qs = (target_Fs * z).sum(dim=-1)  # num_parallel x batch
                _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.cfg.train.fb_pessimism_penalty)  # batch
                cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                inv_cov = torch.inverse(cov)  # z_dim x z_dim
                implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=-1)  # batch
                target_Q = implicit_reward.detach() + discount.squeeze() * next_Q  # batch
                expanded_targets = target_Q.expand(Fs.shape[0], -1)
            Qs = (Fs * z).sum(dim=-1)  # num_parallel x batch
            q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
            fb_loss += q_loss_coef * q_loss

        # optimize FB
        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B.mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "fb_q_loss": q_loss,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
            }
        return output_metrics

    def update_actor(
        self,
        OBS: dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        return self.update_td3_actor(OBS=OBS, z=z, clip_grad_norm=clip_grad_norm)

    def update_td3_actor(self, OBS: torch.Tensor, z: torch.Tensor, clip_grad_norm: float | None) -> Dict[str, torch.Tensor]:

        metrics = {}

        dist = self._model._actor(OBS['policy'], z, self._model.cfg.actor_std)
        action = dist.sample(clip=self.cfg.train.stddev_clip)
        Fs = self._model._forward_map(OBS['obs'], z, action)  # num_parallel x batch x z_dim
        Qs_fb = (Fs * z).sum(-1)  # num_parallel x batch
        _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)  # batch
        actor_loss = -Q_fb.mean()
        metrics["actor_loss_fb"] = actor_loss.detach().clone()

        if self.cfg.model.archi.critic.enable:
            Qs_critic = self._model._critic(OBS['critic'], action)  # with grad
            _, _, Q_critic = self.get_targets_uncertainty(Qs_critic, self.cfg.train.critic_pessimism_penalty)
            
            actor_loss -= Q_critic.mean() * self.cfg.train.reg_coeff
            metrics["actor_loss_critic"] = -(Q_critic.mean() * self.cfg.train.reg_coeff).detach().clone()

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        metrics["actor_loss"] = actor_loss.detach().clone()
        return metrics

    def update_critic(
            self,
            OBS: dict[str, torch.Tensor],
            action: torch.Tensor,
            discount: torch.Tensor,
            NEXT_OBS: dict[str, torch.Tensor],
            z: torch.Tensor,
            reward_reg: torch.Tensor,
            clip_grad_norm: float | None,
        ) -> Dict[str, torch.Tensor]:

        num_parallel = self.cfg.model.archi.critic.num_parallel

        with torch.no_grad():
            dist = self._model._actor(NEXT_OBS['policy'], z, std=self.cfg.model.actor_std)
            next_action = dist.sample(clip=self.cfg.train.stddev_clip)
            next_Qs = self._model._target_critic(NEXT_OBS['critic'], next_action)
            Q_mean, Q_unc, next_Q = self.get_targets_uncertainty(next_Qs, self.cfg.train.critic_pessimism_penalty)

            target_Q = reward_reg + discount * next_Q
            target_Qs = target_Q.expand(num_parallel, -1, -1)

        Qs = self._model._critic(OBS['critic'], action)  # with grad
        critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, target_Qs)

        # optimize critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._critic.parameters(), clip_grad_norm)
        self.critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_Q_reg": target_Q.mean().detach(),
                "Q_reg_1": Q_mean.mean().detach(),
                "unc_Q_reg": Q_unc.mean().detach(),
                "critic_loss": critic_loss.mean().detach(),
                "mean_reg_reward": reward_reg.mean().detach(),
            }
        
        return output_metrics



    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def refresh_z(self, z: torch.Tensor | None, step_count: torch.Tensor) -> torch.Tensor:
        # get mask for environmets where we need to change z
        if z is not None:
            mask_reset_z = step_count % self.cfg.train.update_z_every_step == 0
            if self.cfg.train.use_mix_rollout and not self.z_buffer.empty():
                new_z = self.z_buffer.sample(z.shape[0], device=self.cfg.model.device)
            else:
                new_z = self._model.sample_z(z.shape[0], device=self.cfg.model.device)
            z = torch.where(mask_reset_z, new_z, z.to(self.cfg.model.device))
        else:
            z = self._model.sample_z(step_count.shape[0], device=self.cfg.model.device)
        return z
    

    def save(self, path: str):
        torch.save(
            {'actor':                self._model._actor.state_dict(),
                'policy_normalizer': self._model._policy_normalizer.state_dict(),
                'B':                 self._model._backward_map.state_dict(),
                'B_normalizer':      self._model._B_normalizer.state_dict(),},
            path
        )
        print(f"Model saved to {path}")