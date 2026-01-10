import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
from .agent import FBAgent

class FBAWAgent(FBAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    # Imitate AWAC implementation at:
    # https://github.com/hari-sikchi/AWAC/blob/master/AWAC/awac.py
    def update_actor(
        self,
        OBS: dict[str, torch.Tensor],
        action: torch.Tensor,    #00ff00 action from buffer
        z: torch.Tensor,
        clip_grad_norm: float | None,
        ) -> Dict[str, torch.Tensor]:

        metrics = {}
        action_buf = action

        dist = self._model._actor(OBS['policy'], z, self._model.cfg.actor_std)
        action_pi = dist.sample(clip=self.cfg.train.stddev_clip)

        # Q function = F(s,a,z) * z.T
        @torch.no_grad()
        def compute_Q_fb(obs, action, z):
            Fs = self._model._forward_map(obs, z, action)
            Qs_fb = (Fs * z).sum(-1)
            _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)
            return Q_fb

        # compute weights
        with torch.no_grad():
            # V(s)
            Q_fb_pi = compute_Q_fb(OBS['obs'], action_pi, z)
            V_fb_pi = Q_fb_pi

            # Q(s,a_buf)
            Q_fb_buf = compute_Q_fb(OBS['obs'], action_buf, z)

            # A(s,a_buf) = Q(s,a_buf) - V(s)
            A_pi = Q_fb_buf - V_fb_pi
            beta = 2
            weights = F.softmax(A_pi / beta, dim=0)

        # Compute AWAC loss
        # weight 只是一个权重，本身没有任何梯度
        policy_logpp = dist.log_prob(action_buf).sum(axis=-1)                                  # 计算当前策略在 obs 输出 action_buf 的 log prob
        policy_logpp -= (2*(np.log(2) - action_buf - F.softplus(-2*action_buf))).sum(axis=-1)  # Tanh correction
        loss_pi = (-policy_logpp * len(weights) * weights.detach()).mean()

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss_pi.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        metrics["actor_loss"] = loss_pi.detach().clone()
        return metrics