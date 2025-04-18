# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=unused-import
import copy
import math
import logging
import dataclasses
from collections import OrderedDict, defaultdict
import typing as tp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf
from url_benchmark import utils
# from url_benchmark import replay_buffer as rb
from url_benchmark.rollout_storage import RolloutStorage
from url_benchmark.vecenv_wrapper import ExtendedTimeStep, TimeStep
from .fb_modules import Actor, DiagGaussianActor, ForwardMap, BackwardMap, OnlineCov, EnsembleMLP, HighLevelActor, Critic, RNDCuriosity
from url_benchmark.logger import AverageMeter

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FBDDPGAgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.fb_ddpg.FBDDPGAgent"
    name: str = "fb_ddpg"
    # reward_free: ${reward_free}
    obs_type: str = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    goal_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")  # ${device}
    lr: float = 1e-4
    lr_coef: float = 1
    fb_target_tau: float = 0.01  # 0.001-0.01
    num_expl_steps: int = omegaconf.MISSING  # ???  # to be specified later
    num_inference_steps: int = 5120
    hidden_dim: int = 1024   # 128, 2048
    backward_hidden_dim: int = 526   # 512
    feature_dim: int = 512   # 128, 1024
    z_dim: int = 50  # 100
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)" #
    stddev_clip: float = 0.3  # 1
    update_z_every_step: int = 300
    update_z_proba: float = 1.0
    nstep: int = 1
    batch_size: int = 1024  # 512
    init_fb: bool = True
    ortho_coef: float = 1.0  # 0.01-10
    log_std_bounds: tp.Tuple[float, float] = (-5, 2)  # param for DiagGaussianActor
    temp: float = 1  # temperature for DiagGaussianActor
    future_ratio: float = 0.0
    mix_ratio: float = 0.5  # 0-1
    rand_weight: bool = False  # True, False
    preprocess: bool = True
    norm_z: bool = True
    add_trunk: bool = False
    uncertainty: bool = omegaconf.II("uncertainty")
    n_ensemble: int = 5
    sampling: bool = False  # use argmax to sample curious z (True), otw policy
    num_z_samples: int = 100
    critic_reg: bool = False
    coef_critic_reg: float = 0.3
    epoch_repeats: int = 4


cs = ConfigStore.instance()
cs.store(group="agent", name="fb_ddpg", node=FBDDPGAgentConfig)


class FBDDPGAgent:

    # pylint: disable=unused-argument
    def __init__(self,
                 **kwargs: tp.Any
                 ):
        cfg = FBDDPGAgentConfig(**kwargs)
        self.cfg = cfg
        assert len(cfg.action_shape) == 1
        self.action_dim = cfg.action_shape[0]

        # models
        self.obs_dim = cfg.obs_shape[0]
        if cfg.feature_dim < self.obs_dim:
            logger.warning(f"feature_dim {cfg.feature_dim} should not be smaller that obs_dim {self.obs_dim}")

        goal_dim = cfg.goal_shape[0]

        if cfg.z_dim < goal_dim:
            logger.warning(f"z_dim {cfg.z_dim} should not be smaller that goal_dim {goal_dim}")
        # create the network
        self.actor = Actor(self.obs_dim, cfg.z_dim, self.action_dim,
                           cfg.feature_dim, cfg.hidden_dim,
                           preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        if self.cfg.uncertainty and not self.cfg.sampling:
            self.high_expl_actor = HighLevelActor(self.obs_dim, cfg.z_dim, cfg.hidden_dim).to(cfg.device)

        f_dict = {'obs_dim': self.obs_dim, 'z_dim': cfg.z_dim, 'action_dim': self.action_dim,
                  'feature_dim': cfg.feature_dim, 'hidden_dim': cfg.hidden_dim,
                  'preprocess': cfg.preprocess, 'add_trunk': self.cfg.add_trunk}
        if not cfg.uncertainty:
            self.forward_net = ForwardMap(**f_dict).to(cfg.device)
        else:
            self.forward_net = EnsembleMLP(f_dict, n_ensemble=self.cfg.n_ensemble, device=cfg.device)

        self.backward_net = BackwardMap(goal_dim, cfg.z_dim, cfg.backward_hidden_dim, norm_z=cfg.norm_z).to(cfg.device)
        self.backward_target_net = BackwardMap(goal_dim,
                                               cfg.z_dim, cfg.backward_hidden_dim, norm_z=cfg.norm_z).to(cfg.device)
        # build up the target network
        if not cfg.uncertainty:
            self.forward_target_net = ForwardMap(**f_dict).to(cfg.device)
        else:
            self.forward_target_net = EnsembleMLP(f_dict, n_ensemble=self.cfg.n_ensemble, device=cfg.device)
        # load the weights into the target networks
        self.forward_target_net.load_state_dict(self.forward_net.state_dict())
        self.backward_target_net.load_state_dict(self.backward_net.state_dict())
        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        if self.cfg.uncertainty and not self.cfg.sampling:
            self.high_expl_actor_opt = torch.optim.Adam(self.high_expl_actor.parameters(), lr=cfg.lr)
        self.fb_opt = torch.optim.Adam([{'params': self.forward_net.parameters()},  # type: ignore
                                        {'params': self.backward_net.parameters(), 'lr': cfg.lr_coef * cfg.lr}],
                                       lr=cfg.lr)
        if self.cfg.critic_reg:
            self.Qreg = Critic(self.obs_dim, self.action_dim, cfg.z_dim, layers=[256, 256], out_dim=1).to(cfg.device)
            self.target_Qreg = Critic(self.obs_dim, self.action_dim, cfg.z_dim, layers=[256, 256], out_dim=1).to(cfg.device)
            self.target_Qreg.load_state_dict(self.Qreg.state_dict())
            self.Qreg_opt = torch.optim.Adam([{'params': self.Qreg.parameters()}], lr=cfg.lr)
        self.train()
        self.forward_target_net.train()
        self.backward_target_net.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.actor, self.forward_net, self.backward_net]:
            net.train(training)

    def init_from(self, other) -> None:
        # copy parameters over
        names = ["actor"]
        if self.cfg.init_fb:
            names += ["forward_net", "backward_net", "backward_target_net", "forward_target_net"]
        for name in names:
            utils.hard_update_params(getattr(other, name), getattr(self, name))
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

    def get_goal_meta(self, goal_array: np.ndarray) -> dict:
        desired_goal = torch.tensor(goal_array).unsqueeze(0).to(self.cfg.device)
        with torch.no_grad():
            z = self.backward_net(desired_goal)
        # I think this is not needed, it's already noramlized inside bakkward_net! Nuria
        # if self.cfg.norm_z:
        #     z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        z = z.squeeze(0).cpu().numpy()
        meta = OrderedDict()
        meta['z'] = z
        return meta

    def infer_meta_from_obs_and_rewards(self, obs: torch.Tensor, reward: torch.Tensor) -> dict:
        with torch.no_grad():
            B = self.backward_net(obs)
        z = torch.matmul(reward.T, B) / reward.shape[0]
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        meta = OrderedDict()
        meta['z'] = z
        return meta

    def sample_z(self, size, device: str = "cpu"):
        gaussian_rdv = torch.randn((size, self.cfg.z_dim), dtype=torch.float32, device=device)
        gaussian_rdv = F.normalize(gaussian_rdv, dim=1)
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * gaussian_rdv
        else:
            uniform_rdv = torch.rand((size, self.cfg.z_dim), dtype=torch.float32, device=device)
            z = np.sqrt(self.cfg.z_dim) * uniform_rdv * gaussian_rdv
        return z

    def init_meta(self, obs: torch.Tensor) -> dict:
        if self.cfg.uncertainty:
            split_size = 200
            if obs.shape[0] > split_size:
                meta = defaultdict(torch.Tensor)
                for obs_chunk in obs.split(split_size):
                    partial_meta = self.init_curious_meta(obs_chunk)
                    meta = utils.update_merged_dict(meta, partial_meta)

            else:
                meta = self.init_curious_meta(obs,)
            return meta
        else:
            z = self.sample_z(size=obs.shape[0])
            z = z.squeeze().to(self.cfg.device)
            meta = OrderedDict()
            meta['z'] = z
        meta['updated'] = torch.tensor([[True]] * obs.shape[0], device=self.cfg.device)
        return meta

    def init_curious_meta(self, obs: torch.Tensor) -> dict:
        meta = OrderedDict()
        num_obs, _ = obs.shape
        if self.cfg.sampling:
            with torch.no_grad():
                num_zs = self.cfg.num_z_samples * obs.shape[0]  # call this M
                z = self.sample_z(size=num_zs, device=self.cfg.device)  # M x z_dim
                obs = obs.repeat_interleave(self.cfg.num_z_samples, 0)  # each obs repeated num_zs times x obs_dim --> (M) x obs_dim
                acts = self.actor(obs, z, std=1.).mean  # (M x act_dim take the mean, although querying with std 0 anyways
                F1, F2 = self.forward_net((obs, z, acts))  # ensemble_size x (M) x z_dim
                Q1, Q2 = [torch.einsum('esd, ...sd -> es', Fi, z) for Fi in [F1, F2]]  # ensemble_size x (M)
                epistemic_std1, epistemic_std2 = Q1.std(dim=0), Q2.std(dim=0)  # M # TODO only using epistemic_std1
                epistemic_std1 = epistemic_std1.reshape(num_obs, self.cfg.num_z_samples)  # reshape ---> num_obs x num_z_samples
                idxs = torch.argmax(epistemic_std1, dim=-1)  # num obs
                z = z.reshape(num_obs, self.cfg.num_z_samples, -1)
                meta['z'] = z[torch.arange(z.shape[0]), idxs]  # take the z with the highest epistemic uncertainty
                meta['disagr'] = epistemic_std1.std(dim=-1)  # num_obs, std over disagreements
                meta['updated'] = torch.tensor([[True]] * num_obs, device=self.cfg.device)
        else:
            raise not NotImplementedError
            with torch.no_grad():
                obs = torch.as_tensor(obs, device=self.cfg.device)
                z = self.high_expl_actor(obs, std=1.).sample()  # TODO check std
                if self.cfg.norm_z:
                    z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)
                meta['z'] = z.cpu().numpy()
                meta['updated'] = True
        return meta

    def update_meta(
        self,
        meta: dict,
        env_step: torch.tensor,
        obs: torch.Tensor,
    ) -> dict:
        # substitute old zs for the new ones if it's time to update
        indices = env_step % self.cfg.update_z_every_step == 0  # boolean tensor, reset indices
        if indices.any():
            new_meta = self.init_meta(obs[indices])
            meta['z'][indices] = new_meta['z']
            meta['updated'][indices] = new_meta['updated']
            meta['updated'][~indices] = torch.full_like(meta['updated'][~indices], False)

        return meta

    def act(self, obs, meta, step, eval_mode) -> tp.Any:
        z = meta['z']
        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(obs, z, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action

    def compute_z_correl(self, time_step: TimeStep, meta: dict) -> float:
        goal = time_step.goal  # type: ignore
        with torch.no_grad():
            zs = [torch.Tensor(x).unsqueeze(0).float().to(self.cfg.device) for x in [goal, meta["z"]]]
            zs[0] = self.backward_net(zs[0])
            zs = [F.normalize(z, 1) for z in zs]
            return torch.matmul(zs[0], zs[1].T).item()

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        next_goal: torch.Tensor,
        z: torch.Tensor,
        step: int,
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # compute target successor measure
        with torch.no_grad():
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(next_obs, z, stddev)
            next_action = dist.sample(clip=self.cfg.stddev_clip)
            target_F1, target_F2 = self.forward_target_net((next_obs, z, next_action))  # e? x batch x z_dim
            target_B = self.backward_target_net(next_goal)  # batch x z_dim # TODO changed to next_goal for fb_diag to make sense
            if not self.cfg.uncertainty:
                target_M1 = torch.einsum('sd, td -> st', target_F1, target_B)  # batch x batch
                target_M2 = torch.einsum('sd, td -> st', target_F2, target_B)  # batch x batch
            else:
                target_M1 = torch.einsum('esd, ...td -> est', target_F1, target_B)
                target_M2 = torch.einsum('esd, ...td -> est', target_F2, target_B)
            target_M = torch.min(target_M1, target_M2)
        F1, F2 = self.forward_net((obs, z, action))  # batch x z_dim
        B = self.backward_net(next_goal)  # batch x z_dim # TODO changed to next_goal for fb_diag to make sense
        if not self.cfg.uncertainty:
            M1 = torch.einsum('sd, td -> st', F1, B)  # batch x batch
            M2 = torch.einsum('sd, td -> st', F2, B)  # batch x batch
            I = torch.eye(*M1.size(), device=M1.device)
            off_diag = ~I.bool()
            fb_offdiag: tp.Any = 0.5 * sum((M - discount * target_M)[off_diag].pow(2).mean() for M in [M1, M2])
            fb_diag: tp.Any = -sum(M.diag().mean() for M in [M1, M2])
        else:
            M1 = torch.einsum('esd, ...td -> est', F1, B)  # e x batch x batch
            M2 = torch.einsum('esd, ...td -> est', F2, B)  # e x batch x batch
            I = torch.eye(*M1.size()[1:], device=M1.device)
            off_diag = ~I.bool()
            # # get indices for the first dimension of the M ensemble matrix
            E_indices = torch.arange(M1.shape[0]).unsqueeze(-1).unsqueeze(-1)  # (e, 1, 1)
            # compute the offdiagonal term for each member averaging over batch dim, and summing over E and over M1 and M2
            # this one seems to be quite costly
            scaled_T = discount * target_M
            fb_offdiag: tp.Any = 0.5 * (sum((M - scaled_T)[E_indices, off_diag].pow(2).mean() for M in [M1, M2]))

            # M.diagonal(dim1=-2, dim2=-1) returns diagonals over every ensemble so size is: E x batch
            # then we average over B and sum over E and over M1 and M2
            fb_diag: tp.Any = -sum(M.diagonal(dim1=-2, dim2=-1).mean() for M in [M1, M2])
        fb_loss = fb_offdiag + fb_diag

        # orthonormality loss for backward embedding
        Cov = torch.matmul(B, B.T)
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.cfg.ortho_coef * orth_loss

        # Cov = torch.cov(B.T)  # Vicreg loss
        # var_loss = F.relu(1 - Cov.diag().clamp(1e-4, 1).sqrt()).mean()  # eps avoids inf. sqrt gradient at 0
        # cov_loss = 2 * torch.triu(Cov, diagonal=1).pow(2).mean() # 2x upper triangular part
        # orth_loss =  var_loss + cov_loss
        # fb_loss += self.cfg.ortho_coef * orth_loss

        metrics['target_M'] = target_M.mean().item()
        metrics['M1'] = M1.mean().item()
        metrics['F1'] = F1.mean().item()
        metrics['B'] = B.mean().item()
        metrics['B_norm'] = torch.norm(B, dim=-1).mean().item()
        metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
        metrics['fb_loss'] = fb_loss.item()
        metrics['fb_diag'] = fb_diag.item()
        metrics['fb_offdiag'] = fb_offdiag.item()
        metrics['orth_loss'] = orth_loss.item()
        metrics['orth_loss_diag'] = orth_loss_diag.item()
        metrics['orth_loss_offdiag'] = orth_loss_offdiag.item()
        eye_diff = torch.matmul(B.T, B) / B.shape[0] - torch.eye(B.shape[1], device=B.device)
        metrics['orth_linf'] = torch.max(torch.abs(eye_diff)).item()
        metrics['orth_l2'] = eye_diff.norm().item() / math.sqrt(B.shape[1])
        if isinstance(self.fb_opt, torch.optim.Adam):
            metrics["fb_opt_lr"] = self.fb_opt.param_groups[0]["lr"]

        # optimize FB
        self.fb_opt.zero_grad(set_to_none=True)
        fb_loss.backward()
        self.fb_opt.step()
        return metrics

    def update_actor(self, obs: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        stddev = utils.schedule(self.cfg.stddev_schedule, step)
        dist = self.actor(obs, z, stddev)
        action = dist.sample(clip=self.cfg.stddev_clip)  # non differentiable / differentiable?

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        F1, F2 = self.forward_net((obs, z, action))

        if self.cfg.uncertainty:
            # TODO: Average over ensembles?
            Q1 = torch.einsum('esd, ...sd -> es', F1, z).mean(0)  # the broadcasting ... (for the ensemble dim) is not needed. remove?
            Q2 = torch.einsum('esd, ...sd -> es', F2, z).mean(0)
        else:
            Q1 = torch.einsum('sd, sd -> s', F1, z)
            Q2 = torch.einsum('sd, sd -> s', F2, z)

        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        if self.cfg.critic_reg:
            Q_reg = self.Qreg(obs, action, z).mean()
            actor_loss -= self.cfg.coef_critic_reg * Q_reg
            metrics['actor_loss_reg'] = -self.cfg.coef_critic_reg * Q_reg.item()
            metrics['actor_loss_fb'] = -Q.mean().item()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['q'] = Q.mean().item()
        metrics['actor_logprob'] = log_prob.mean().item()
        # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_Qreg(self, obs: torch.Tensor, action: torch.Tensor, discount: torch.Tensor, next_obs: torch.Tensor, z: torch.Tensor, rew: torch.Tensor):

        values_Qreg = self.Qreg(obs, action, z)
        with torch.no_grad():
            next_action = self.actor(next_obs, z, std=1.).sample()
            target_values_Qreg = self.target_Qreg(next_obs, next_action, z)
        Qreg_loss = F.mse_loss(values_Qreg, rew + discount * target_values_Qreg)
        self.Qreg_opt.zero_grad()
        Qreg_loss.backward()
        self.Qreg_opt.step()

        metrics = {}
        metrics['Qreg_loss'] = Qreg_loss.item()
        return metrics

    def update_high_expl_actor(self, obs: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        stddev = utils.schedule(self.cfg.stddev_schedule, step)  # TODO check
        dist = self.high_expl_actor(obs, stddev)
        z = dist.sample()
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        with torch.no_grad():  # TODO check
            a = self.actor(obs, z, std=1.).mean  # TODO mean?
        F1, F2 = self.forward_net((obs, z, a))

        Q1, Q2 = [torch.einsum('esd, ...sd -> es', Fi, z) for Fi in [F1, F2]]
        epistemic_std1, epistemic_std2 = Q1.std(dim=0), Q2.std(dim=0)  # TODO not using epistemic_std2
        epistemic_std = epistemic_std1.mean()
        actor_loss = -epistemic_std
        # optimize actor
        self.high_expl_actor_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.high_expl_actor_opt.step()

        metrics['high_actor_loss'] = actor_loss.item()
        return metrics

    def update(self, replay_loader: RolloutStorage, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        average_meter = defaultdict(AverageMeter)
        generator = replay_loader.mini_batch_generator(mini_batch_size=512, num_epochs=self.cfg.epoch_repeats)  # TODO update numbers?
        for batch in generator:
            # batch = batch.to(self.cfg.device)

            obs = batch.obs
            goal = batch.goal
            action = batch.action
            discount = batch.discount
            next_obs = batch.next_obs
            batch_size = obs.size(0)
            next_goal = batch.next_goal
            rew = batch.reward

            z = self.sample_z(batch_size, device=self.cfg.device)
            if not z.shape[-1] == self.cfg.z_dim:
                raise RuntimeError("There's something wrong with the logic here")
            backward_input = batch.goal
            future_goal = batch.future_goal

            perm = torch.randperm(batch_size)
            backward_input = backward_input[perm]

            if self.cfg.mix_ratio > 0:
                mix_idxs: tp.Any = np.where(np.random.uniform(size=batch_size) < self.cfg.mix_ratio)[0]
                if not self.cfg.rand_weight:
                    with torch.no_grad():
                        mix_z = self.backward_net(backward_input[mix_idxs]).detach()
                else:
                    # generate random weight
                    weight = torch.rand(size=(mix_idxs.shape[0], batch_size)).to(self.cfg.device)
                    weight = F.normalize(weight, dim=1)
                    uniform_rdv = torch.rand(mix_idxs.shape[0], 1).to(self.cfg.device)
                    weight = uniform_rdv * weight
                    with torch.no_grad():
                        mix_z = torch.matmul(weight, self.backward_net(backward_input).detach())
                if self.cfg.norm_z:
                    mix_z = math.sqrt(self.cfg.z_dim) * F.normalize(mix_z, dim=1)
                z[mix_idxs] = mix_z

            # hindsight replay
            if self.cfg.future_ratio > 0:
                print('I think one needs to also normalize the zs afterwards (Nuria)')
                assert future_goal is not None
                future_idxs = np.where(np.random.uniform(size=batch_size) < self.cfg.future_ratio)
                z[future_idxs] = self.backward_net(future_goal[future_idxs]).detach()
            metrics.update(self.update_fb(obs=obs, action=action, discount=discount,
                                          next_obs=next_obs, next_goal=next_goal, z=z, step=step,
                                          ))  # type: ignore

            # update critic regularizer

            if self.cfg.critic_reg:
                metrics.update(self.update_Qreg(obs=obs, action=action, discount=discount,
                                                next_obs=next_obs, z=z, rew=rew))
            # update high expl actor
            if self.cfg.uncertainty and not self.cfg.sampling:
                metrics.update(self.update_high_expl_actor(obs, step))

            # update actor
            metrics.update(self.update_actor(obs, z, step))

            # update critic target
            utils.soft_update_params(self.forward_net, self.forward_target_net,
                                     self.cfg.fb_target_tau)
            utils.soft_update_params(self.backward_net, self.backward_target_net,
                                     self.cfg.fb_target_tau)

            # average metrics
            for k, v in metrics.items():
                average_meter[k].update(v)
        replay_loader.clear()

        average_metrics = {k: v.value() for k, v in average_meter.items()}

        return average_metrics
