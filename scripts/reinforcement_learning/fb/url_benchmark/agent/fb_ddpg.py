# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=unused-import
import pdb
import copy
import math
import logging
import dataclasses
from collections import OrderedDict
import typing as tp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf
from dm_env import specs

from url_benchmark import utils
# from url_benchmark import replay_buffer as rb
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.dmc import TimeStep
from url_benchmark import goals as _goals
from .ddpg import MetaDict
from .fb_modules import IdentityMap
from .ddpg import Encoder
from .fb_modules import Actor, DiagGaussianActor, ForwardMap, BackwardMap, OnlineCov, EnsembleMLP, HighLevelActor, Critic, RNDCuriosity


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FBDDPGAgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.fb_ddpg.FBDDPGAgent"
    name: str = "fb_ddpg"
    # reward_free: ${reward_free}
    obs_type: str = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")  # ${device}
    lr: float = 1e-4
    lr_coef: float = 1
    fb_target_tau: float = 0.01  # 0.001-0.01
    update_every_steps: int = 2
    use_tb: bool = omegaconf.II("use_tb")  # ${use_tb}
    use_wandb: bool = omegaconf.II("use_wandb")  # ${use_wandb}
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
    update_encoder: bool = omegaconf.II("update_encoder")  # ${update_encoder}
    goal_space: tp.Optional[str] = omegaconf.II("goal_space")
    ortho_coef: float = 1.0  # 0.01-10
    log_std_bounds: tp.Tuple[float, float] = (-5, 2)  # param for DiagGaussianActor
    temp: float = 1  # temperature for DiagGaussianActor
    boltzmann: bool = False  # set to true for DiagGaussianActor
    debug: bool = False
    future_ratio: float = 0.0
    mix_ratio: float = 0.5  # 0-1
    rand_weight: bool = False  # True, False
    preprocess: bool = True
    norm_z: bool = True
    q_loss: bool = False
    q_loss_coef: float = 0.01
    additional_metric: bool = False
    add_trunk: bool = False
    uncertainty: bool = omegaconf.II("uncertainty")
    one_target: bool = False
    n_ensemble: int = 5
    sampling: bool = False  # use argmax to sample curious z (True), otw policy
    myopic: bool = True
    num_z_samples: int = 100
    num_obs_samples: int = 1000
    rnd_coeff: float = 0.5
    rnd: bool = False
    rnd_embed_dim: int = 100


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
        self.solved_meta: tp.Any = None

        # models
        if cfg.obs_type == 'pixels':
            self.aug: nn.Module = utils.RandomShiftsAug(pad=4)
            self.encoder: nn.Module = Encoder(cfg.obs_shape).to(cfg.device)
            self.obs_dim = self.encoder.repr_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = cfg.obs_shape[0]
        if cfg.feature_dim < self.obs_dim:
            logger.warning(f"feature_dim {cfg.feature_dim} should not be smaller that obs_dim {self.obs_dim}")
        assert not (cfg.rnd and cfg.uncertainty), 'Cannot use both RND and uncertainty'
        
        goal_dim = self.obs_dim
        if cfg.goal_space is not None:
            goal_dim = _goals.get_goal_space_dim(cfg.goal_space)
        if cfg.z_dim < goal_dim:
            logger.warning(f"z_dim {cfg.z_dim} should not be smaller that goal_dim {goal_dim}")
        # create the network
        if self.cfg.boltzmann:
            self.actor: nn.Module = DiagGaussianActor(self.obs_dim, cfg.z_dim, self.action_dim,
                                                      cfg.hidden_dim, cfg.log_std_bounds).to(cfg.device)
        else:
            self.actor = Actor(self.obs_dim, cfg.z_dim, self.action_dim,
                               cfg.feature_dim, cfg.hidden_dim,
                               preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        if self.cfg.uncertainty and self.cfg.myopic and not self.cfg.sampling:
            self.high_expl_actor = HighLevelActor(self.obs_dim, cfg.z_dim, cfg.hidden_dim).to(cfg.device)
       
        f_dict = {'obs_dim': self.obs_dim, 'z_dim': cfg.z_dim, 'action_dim': self.action_dim,
                  'feature_dim': cfg.feature_dim, 'hidden_dim': cfg.hidden_dim,
                  'preprocess': cfg.preprocess, 'add_trunk': self.cfg.add_trunk}
        if not cfg.uncertainty:
            self.forward_net = ForwardMap(**f_dict).to(cfg.device)
        else:
            self.forward_net = EnsembleMLP(f_dict, n_ensemble=self.cfg.n_ensemble, device=cfg.device)
        if cfg.debug:
            self.backward_net: nn.Module = IdentityMap().to(cfg.device)
            self.backward_target_net: nn.Module = IdentityMap().to(cfg.device)
        else:
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
        if cfg.one_target and cfg.uncertainty:
            # Initialise target params for each ensemble member to the mean of the forward_net
            utils.soft_update_params_mean(self.forward_net, self.forward_target_net, tau=1.0)
        self.backward_target_net.load_state_dict(self.backward_net.state_dict())
        # optimizers
        self.encoder_opt: tp.Optional[torch.optim.Adam] = None
        if cfg.obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        if self.cfg.uncertainty and self.cfg.myopic and not self.cfg.sampling:
            self.high_expl_actor_opt = torch.optim.Adam(self.high_expl_actor.parameters(), lr=cfg.lr)
        self.fb_opt = torch.optim.Adam([{'params': self.forward_net.parameters()},  # type: ignore
                                        {'params': self.backward_net.parameters(), 'lr': cfg.lr_coef * cfg.lr}],
                                       lr=cfg.lr)

        if self.cfg.rnd:
            self.Q_rnd = Critic(self.obs_dim, self.action_dim, cfg.z_dim, self.cfg.hidden_dim).to(cfg.device)
            self.target_Q_rnd = Critic(self.obs_dim, self.action_dim, cfg.z_dim, self.cfg.hidden_dim).to(cfg.device)
            self.target_Q_rnd.load_state_dict(self.Q_rnd.state_dict())
            self.Q_rnd_opt = torch.optim.Adam(self.Q_rnd.parameters(), lr=cfg.lr)
            self.rnd_module = RNDCuriosity(self.obs_dim, self.cfg.hidden_dim, self.cfg.rnd_embed_dim, self.cfg.lr, cfg.device)

        if self.cfg.uncertainty and not self.cfg.myopic:
            if self.cfg.update_z_proba > 0.:
                print('Setting update z_proba to zero when use nonmyopic approach!')
                self.cfg.update_z_proba = 0.
        self.train()
        self.forward_target_net.train()
        self.backward_target_net.train()
        self.actor_success: tp.List[float] = []  # only for debugging, can be removed eventually
        # self.inv_cov = torch.eye(self.cfg.z_dim, dtype=torch.float32, device=self.cfg.device)
        # self.online_cov = OnlineCov(mom=0.99, dim=self.cfg.z_dim).to(self.cfg.device)
        # self.online_cov.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.encoder, self.actor, self.forward_net, self.backward_net]:
            net.train(training)

    def init_from(self, other) -> None:
        # copy parameters over
        names = ["encoder", "actor"]
        if self.cfg.init_fb:
            names += ["forward_net", "backward_net", "backward_target_net", "forward_target_net"]
        for name in names:
            utils.hard_update_params(getattr(other, name), getattr(self, name))
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

    def get_goal_meta(self, goal_array: np.ndarray) -> MetaDict:
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

    def infer_meta(self, replay_loader: ReplayBuffer) -> MetaDict:
        obs_list, reward_list = [], []
        batch_size = 0
        while batch_size < self.cfg.num_inference_steps:
            batch = replay_loader.sample(self.cfg.batch_size)
            batch = batch.to(self.cfg.device)
            obs_list.append(batch.next_goal if self.cfg.goal_space is not None else batch.next_obs)
            reward_list.append(batch.reward)
            batch_size += batch.next_obs.size(0)
        obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
        obs, reward = obs[:self.cfg.num_inference_steps], reward[:self.cfg.num_inference_steps]
        return self.infer_meta_from_obs_and_rewards(obs, reward)

    def infer_meta_from_obs_and_rewards(self, obs: torch.Tensor, reward: torch.Tensor) -> MetaDict:
        # print('max reward: ', reward.max().cpu().item())
        # print('99 percentile: ', torch.quantile(reward, 0.99).cpu().item())
        # print('median reward: ', reward.median().cpu().item())
        # print('min reward: ', reward.min().cpu().item())
        # print('mean reward: ', reward.mean().cpu().item())
        # print('num reward: ', reward.shape[0])

        # filter out small reward
        # pdb.set_trace()
        # idx = torch.where(reward >= torch.quantile(reward, 0.99))[0]
        # obs = obs[idx]
        # reward = reward[idx]
        with torch.no_grad():
            B = self.backward_net(obs)
        z = torch.matmul(reward.T, B) / reward.shape[0]
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        meta = OrderedDict()
        meta['z'] = z.squeeze().cpu().numpy()
        # self.solved_meta = meta
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

    def init_meta(self, obs: np.ndarray = None, replay_loader: tp.Optional[ReplayBuffer] = None) -> MetaDict:
        if self.cfg.uncertainty:
            if self.cfg.myopic:
                return self.init_curious_meta(obs,)
            else:
                return self.init_nonmyopic_curious_meta(obs, replay_loader)
        if self.solved_meta is not None:
            print('solved_meta')
            return self.solved_meta
        else:
            z = self.sample_z(1)
            z = z.squeeze().numpy()
            meta = OrderedDict()
            meta['z'] = z
        meta['updated'] = True
        return meta

    def init_nonmyopic_curious_meta(self, obs: np.ndarray, replay_loader: tp.Optional[ReplayBuffer]) -> MetaDict:
        meta = OrderedDict()
        num_steps = self.cfg.num_obs_samples  # type: ignore
        if len(replay_loader) * replay_loader._episodes_length[0] < num_steps:
            # print("Not enough data for inference, random z")
            z = self.sample_z(1)
            z = z.squeeze().numpy()
            meta['z'] = z
        else:
            obs_list, reward_list = [], []
            batch_size = 0
            while batch_size < num_steps:
                batch = replay_loader.sample(self.cfg.batch_size)
                batch = batch.to(self.cfg.device)
                # Compute reward:
                with torch.no_grad():
                    num_zs = self.cfg.num_z_samples
                    z = self.sample_z(size=num_zs, device=self.cfg.device)  # num_zs x z_dim
                    for obs in batch.obs:
                        obs = obs.expand(num_zs, -1)
                        acts = self.actor(obs, z, std=1.).mean  # num_zs x act_dim take the mean
                        F1, F2 = self.forward_net((obs, z, acts))  # ensemble_size x num_zs x z_dim
                        Q1, Q2 = [torch.einsum('esd, ...sd -> es', Fi, z) for Fi in [F1, F2]]  # ensemble_size x num_zs
                        epistemic_std1, epistemic_std2 = Q1.std(dim=0), Q2.std(dim=0)  # num_zs # TODO only using epistemic_std1
                        reward_list.append(epistemic_std1.mean()) #TODO MEAN? std1 only?
                obs_list.append(batch.goal if self.cfg.goal_space is not None else batch.obs)
                batch_size += batch.obs.size(0)
            obs, reward = torch.cat(obs_list, 0), torch.stack(reward_list).unsqueeze(dim=1)  # type: ignore
            obs_t, reward_t = obs[:num_steps], reward[:num_steps]
            meta = self.infer_meta_from_obs_and_rewards(obs_t, reward_t)
        meta['updated'] = True
        return meta

    def init_curious_meta(self, obs: np.ndarray) -> MetaDict:
        meta = OrderedDict()
        if not self.cfg.myopic:
            return self.init_nonmyopic_curious_meta(obs)
        if self.cfg.sampling:
            with torch.no_grad():
                num_zs = self.cfg.num_z_samples
                z = self.sample_z(size=num_zs, device=self.cfg.device)  # num_zs x z_dim
                obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float32).expand(num_zs, -1) # num_zs x obs_dim
                h = self.encoder(obs)
                acts = self.actor(h, z, std=1.).mean  # num_zs x act_dim take the mean, although querying with std 0 anyways
                F1, F2 = self.forward_net((obs, z, acts))  # ensemble_size x num_zs x z_dim
                Q1, Q2 = [torch.einsum('esd, ...sd -> es', Fi, z) for Fi in [F1, F2]]  # ensemble_size x num_zs
            epistemic_std1, epistemic_std2 = Q1.std(dim=0), Q2.std(dim=0)  # num_zs # TODO only using epistemic_std1
            idxs = torch.argmax(epistemic_std1, dim=0)
            uncertain_z = z[idxs].cpu().numpy()  # take the z with the highest epistemic uncertainty
            meta['z'] = uncertain_z
            meta['disagr'] = epistemic_std1.std().item()
            meta['updated'] = True
        else:
            with torch.no_grad():
                obs = torch.as_tensor(obs, device=self.cfg.device)
                z = self.high_expl_actor(obs, std=1.).sample()  # TODO check std
                if self.cfg.norm_z:
                    z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)
                meta['z'] = z.cpu().numpy()
                meta['updated'] = True
        return meta

    # pylint: disable=unused-argument
    def update_meta(
        self,
        meta: MetaDict,
        global_step: int,
        time_step: TimeStep,
        finetune: bool = False,
        replay_loader: tp.Optional[ReplayBuffer] = None,
        obs: np.ndarray = None,
    ) -> MetaDict:
        if global_step % self.cfg.update_z_every_step == 0 and np.random.rand() < self.cfg.update_z_proba:
            return self.init_meta() if not self.cfg.uncertainty else self.init_curious_meta(obs)
        meta['updated'] = False
        return meta

    def act(self, obs, meta, step, eval_mode) -> tp.Any:
        obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float32).unsqueeze(0)  # type: ignore
        h = self.encoder(obs)
        z = torch.as_tensor(meta['z'], device=self.cfg.device).unsqueeze(0)  # type: ignore
        if self.cfg.boltzmann:
            dist = self.actor(h, z)
        else:
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(h, z, stddev)
        if eval_mode:
            action = dist.mean
            if self.cfg.additional_metric:
                # the following is doing extra computation only used for metrics,
                # it should be deactivated eventually
                F_mean_s = self.forward_net((obs, z, action))
                # F_samp_s = self.forward_net(obs, z, dist.sample())
                F_rand_s = self.forward_net((obs, z, torch.zeros_like(action).uniform_(-1.0, 1.0)))
                Qs = [torch.min(*(torch.einsum('sd, sd -> s', F, z) for F in Fs)) for Fs in [F_mean_s, F_rand_s]]
                self.actor_success = (Qs[0] > Qs[1]).cpu().numpy().tolist()
        else:
            action = dist.sample()
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def compute_z_correl(self, time_step: TimeStep, meta: MetaDict) -> float:
        goal = time_step.goal if self.cfg.goal_space is not None else time_step.observation  # type: ignore
        with torch.no_grad():
            zs = [torch.Tensor(x).unsqueeze(0).float().to(self.cfg.device) for x in [goal, meta["z"]]]
            zs[0] = self.backward_net(zs[0])
            zs = [F.normalize(z, 1) for z in zs]
            return torch.matmul(zs[0], zs[1].T).item()

    def compute_disagreement_metrics(self) -> float:
        if self.eval_states is None:
            return {}
        with torch.no_grad():
            h = self.encoder(self.eval_states)
            acts = self.actor(h, self.eval_zs, std=0.).mean  # num_zs x act_dim take the mean, although querying with std 0 anyways
            F1, F2 = self.forward_net((self.eval_states, self.eval_zs, acts))  # ensemble_size x num_zs x z_dim
            if self.cfg.uncertainty:
                self.Q1, self.Q2 = [torch.einsum('esd, ...sd -> es', Fi, self.eval_zs) for Fi in [F1, F2]]  # ensemble_size x num_zs
            else:
                self.Q1, self.Q2 = [torch.einsum('sd, sd -> s', Fi, self.eval_zs) for Fi in [F1, F2]]
        metrics = {}
        if self.cfg.uncertainty:
            epistemic_std1, epistemic_std2 = self.Q1.std(dim=0), self.Q2.std(dim=0)  # num_zs
            metrics.update({f'disagreement{i+1}': std1.item() for i, std1 in enumerate(epistemic_std1)})
            metrics['disagreement'] = epistemic_std1.mean().item()
            metrics.update({f'disagreement{i+1}_scale': std1.item()/q.item() for i, std1, q in zip(range(len(epistemic_std1)), epistemic_std1, self.Q1.mean(dim=0))})
            for room in range(len(self.Q1.mean(dim=0))):
                metrics.update({f'Qroom{room+1}_{i+1}': q.item() for i, q in enumerate(self.Q1[:, room])})
        metrics.update({f'Qroom{i+1}': q.item() for i, q in enumerate(self.Q1.mean(dim=0))} if self.Q1.dim() > 1 else {f'Qroom{i+1}': q.item() for i, q in enumerate(self.Q1)})   
        return metrics


    def compute_disagreement_metrics_new(self) -> float:
        with torch.no_grad():
            bs = len(self.eval_states)
            num_zs = len(self.eval_zs)
            self.eval_states = self.eval_states.repeat(len(self.eval_zs), 1) #( eval_states x eval_zs) x s_dim
            self.eval_zs = torch.repeat_interleave(self.eval_zs, bs, 0) #( eval_states x eval_zs) x z_dim
            h = self.encoder(self.eval_states)
            acts = self.actor(h, self.eval_zs, std=0.).mean  # num_zs x act_dim take the mean, although querying with std 0 anyways
            F1, F2 = self.forward_net((self.eval_states, self.eval_zs, acts))  # ensemble_size x (num_zs x eval_states) x z_dim
            eF1, eF2 = [], []        
            for i in range(len(self.eval_zs)):
                F11 = F1[:,i,:].squeeze().to(torch.float64)
                F11 = F11.transpose(1,0) # d x ensemble_size
                cov_F1 = torch.cov(F11)
                epistemic_F1 = torch.linalg.det(cov_F1)
                # eigenvals = torch.linalg.eigvals(cov_F1)
                # print(len(eigenvals), eigenvals)
                # epistemic_F1 = torch.trace(cov_F1)
                eF1.append(epistemic_F1)
                
            self.Q1, _= [torch.einsum('esd, ...sd -> es', Fi, self.eval_zs) for Fi in [F1, F2]]  # ensemble_size x (num_zs x eval_states)
            
            epistemic_q1 = self.Q1.std(dim=0)
            ef = torch.tensor(eF1).reshape(-1, bs).mean(-1)
            eq = epistemic_q1.reshape(-1, bs).mean(-1)

        return ef, eq

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        next_goal: torch.Tensor,
        z: torch.Tensor,
        step: int,
        goal: torch.Tensor,
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # compute target successor measure
        with torch.no_grad():
            if self.cfg.boltzmann:
                dist = self.actor(next_obs, z)
                next_action = dist.sample()
            else:
                stddev = utils.schedule(self.cfg.stddev_schedule, step)
                dist = self.actor(next_obs, z, stddev)
                next_action = dist.sample(clip=self.cfg.stddev_clip)
            # target_F1, target_F2 = torch.ones((self.n_ensemble, 1024, 50), device = 'cuda:0'), torch.ones((self.n_ensemble, 1024, 50), device = 'cuda:0') #self.forward_target_net((next_obs, z, next_action))  # batch x z_dim
            target_F1, target_F2 = self.forward_target_net((next_obs, z, next_action))  # e? x batch x z_dim
            target_B = self.backward_target_net(goal)  # batch x z_dim
            if not self.cfg.uncertainty:
                target_M1 = torch.einsum('sd, td -> st', target_F1, target_B)  # batch x batch
                target_M2 = torch.einsum('sd, td -> st', target_F2, target_B)  # batch x batch
            else:
                target_M1 = torch.einsum('esd, ...td -> est', target_F1, target_B)  # e x batch x batch torch.ones((self.n_ensemble, 1024, 1024), device = target_F1.device) #
                target_M2 = torch.einsum('esd, ...td -> est', target_F2, target_B)  # e x batch x batch torch.ones((self.n_ensemble, 1024, 1024), device = target_F1.device) #
            target_M = torch.min(target_M1, target_M2)
        # TODO Should I sample different batches for every different F ensemble?
        # compute FB loss
        # F1, F2 = torch.ones((self.n_ensemble, 1024, 50), device = 'cuda:0'), torch.ones((self.n_ensemble, 1024, 50), device = 'cuda:0') #self.forward_net((obs, z, action))  # batch x z_dim
        F1, F2 = self.forward_net((obs, z, action))  # batch x z_dim
        B = self.backward_net(goal)  # batch x z_dim
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
            # compute the offidagonal term for each member averaging over batch dim, and summing over E and over M1 and M2
            # this one seems to be quite costly
            scaled_T = discount * target_M
            # fb_offdiag: tp.Any = 1/(2*self.cfg.n_ensemble) * sum(sum((M - scaled_T)[E_indices, off_diag].pow(2).mean(-1) for M in [M1, M2]))
            fb_offdiag: tp.Any = 0.5*(sum((M - scaled_T)[E_indices, off_diag].pow(2).mean() for M in [M1, M2]))

            # M.diagonal(dim1=-2, dim2=-1) returns diagonals over every ensemble so size is: E x batch
            # then we average over B and sum over E and over M1 and M2
            # fb_diag: tp.Any = -sum(sum(M.diagonal(dim1=-2, dim2=-1).mean(-1) for M in [M1, M2]))
            fb_diag: tp.Any = -sum(M.diagonal(dim1=-2, dim2=-1).mean() for M in [M1, M2])
        fb_loss = fb_offdiag + fb_diag
        # Q LOSS
        if self.cfg.q_loss:  # TODO This is not updated with curiosity
            with torch.no_grad():
                next_Q1, nextQ2 = [torch.einsum('sd, sd -> s', target_Fi, z) for target_Fi in [target_F1, target_F2]]
                next_Q = torch.min(next_Q1, nextQ2)
                cov = torch.matmul(B.T, B) / B.shape[0]
                inv_cov = torch.inverse(cov)
                implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=1)  # batch_size
                target_Q = implicit_reward.detach() + discount.squeeze(1) * next_Q  # batch_size
            Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
            q_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
            fb_loss += self.cfg.q_loss_coef * q_loss

        # ORTHONORMALITY LOSS FOR BACKWARD EMBEDDING

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

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['target_M'] = target_M.mean().item()
            metrics['M1'] = M1.mean().item()
            metrics['F1'] = F1.mean().item()
            metrics['B'] = B.mean().item()
            metrics['B_norm'] = torch.norm(B, dim=-1).mean().item()
            metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
            metrics['fb_loss'] = fb_loss.item()
            metrics['fb_diag'] = fb_diag.item()
            metrics['fb_offdiag'] = fb_offdiag.item()
            if self.cfg.q_loss:
                metrics['q_loss'] = q_loss.item()
            metrics['orth_loss'] = orth_loss.item()
            metrics['orth_loss_diag'] = orth_loss_diag.item()
            metrics['orth_loss_offdiag'] = orth_loss_offdiag.item()
            if self.cfg.q_loss:
                metrics['q_loss'] = q_loss.item()
            eye_diff = torch.matmul(B.T, B) / B.shape[0] - torch.eye(B.shape[1], device=B.device)
            metrics['orth_linf'] = torch.max(torch.abs(eye_diff)).item()
            metrics['orth_l2'] = eye_diff.norm().item() / math.sqrt(B.shape[1])
            if isinstance(self.fb_opt, torch.optim.Adam):
                metrics["fb_opt_lr"] = self.fb_opt.param_groups[0]["lr"]

        # optimize FB
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.fb_opt.zero_grad(set_to_none=True)
        fb_loss.backward()
        self.fb_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    @torch.compile
    def diag_fcts(self, M1, M2, target_M, discount, E_indices, off_diag):
        fb_offdiag: tp.Any = 0.5 * sum(sum((M - discount * target_M)[E_indices, off_diag].pow(2).mean(-1) for M in [M1, M2])) 
        # fb_offdiag1: tp.Any = (M1 - scaled_T)[E_indices, off_diag].pow(2).mean(-1) # e x 1
        # fb_offdiag2: tp.Any = (M2 - scaled_T)[E_indices, off_diag].pow(2).mean(-1)
        # fb_offdiag = 0.5 * (fb_offdiag1 + fb_offdiag2).sum()
        # M.diagonal(dim1=-2, dim2=-1) returns diagonals over every ensemble so size is: E x batch
        # then we average over B and sum over E and over M1 and M2
        fb_diag: tp.Any = -sum(sum(M.diagonal(dim1=-2, dim2=-1).mean(-1) for M in [M1, M2]))
        return fb_diag, fb_offdiag
    
    def update_actor(self, obs: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        if self.cfg.boltzmann:
            dist = self.actor(obs, z)
            action = dist.rsample() # differentiable/ non differentiable?
        else:
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(obs, z, stddev)
            action = dist.sample(clip=self.cfg.stddev_clip) # non differentiable / differentiable?

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        F1, F2 = self.forward_net((obs, z, action))
        
        if self.cfg.uncertainty:
            # TODO: Average over ensembles?
            Q1 = torch.einsum('esd, ...sd -> es', F1, z).mean(0)  # the broadcasting ... (for the ensemble dim) is not needed. remove?
            Q2 = torch.einsum('esd, ...sd -> es', F2, z).mean(0)
        else:
            Q1 = torch.einsum('sd, sd -> s', F1, z)
            Q2 = torch.einsum('sd, sd -> s', F2, z)
        if self.cfg.additional_metric:
            q1_success = Q1 > Q2
        Q = torch.min(Q1, Q2)
        actor_loss = (self.cfg.temp * log_prob - Q).mean() if self.cfg.boltzmann else -Q.mean()
        if self.cfg.rnd:
            rnd_loss = -self.Q_rnd(obs, action, z).mean()
            actor_loss += self.cfg.rnd_coeff * rnd_loss
            metrics['actor_loss_rnd'] = rnd_loss.item() if self.cfg.rnd else 0
            metrics['actor_loss_exploit'] = -Q.mean().item()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['q'] = Q.mean().item()
            if self.cfg.additional_metric:
                metrics['q1_success'] = q1_success.float().mean().item()
            metrics['actor_logprob'] = log_prob.mean().item()
            # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_high_expl_actor(self, obs: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        stddev = utils.schedule(self.cfg.stddev_schedule, step) # TODO check
        dist = self.high_expl_actor(obs, stddev)
        z = dist.sample()
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        with torch.no_grad(): # TODO check
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

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['high_actor_loss'] = actor_loss.item()
        return metrics

    def update_Qrnd(self, obs: torch.Tensor, action: torch.Tensor, discount: torch.Tensor, next_obs: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        values_rnd = self.Q_rnd(obs, action, z)

        with torch.no_grad():
            next_action = self.actor(next_obs, z, std=1.).sample()
            next_values_rnd = self.target_Q_rnd(next_obs, next_action, z)
        reward, rnd_loss = self.rnd_module.update_curiosity(next_obs)
        value_loss = F.mse_loss(values_rnd, reward + discount * next_values_rnd)
        self.Q_rnd_opt.zero_grad()
        value_loss.backward()
        self.Q_rnd_opt.step()

        metrics['Qrnd_loss'] = value_loss.item()
        metrics['RND_net_loss'] = rnd_loss.item()
        return metrics

    def aug_and_encode(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.cfg.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size)
        batch = batch.to(self.cfg.device)

        obs = batch.obs
        goal = batch.obs
        action = batch.action
        discount = batch.discount
        next_obs = next_goal = batch.next_obs
        if self.cfg.goal_space is not None:
            # in case goal_space is defined and next_goal is not in batch (case of prefill)
            if batch.next_goal is None and self.cfg.goal_space == 'simplified_point_mass_maze':
                batch.next_goal = batch.next_obs[:, :2]
            assert batch.next_goal is not None
            next_goal = batch.next_goal

        z = self.sample_z(self.cfg.batch_size, device=self.cfg.device)
        if not z.shape[-1] == self.cfg.z_dim:
            raise RuntimeError("There's something wrong with the logic here")
        backward_input = batch.obs
        future_goal = batch.future_obs
        if self.cfg.goal_space is not None:
            # in case goal_space is defined and next_goal is not in batch (case of prefill)
            if batch.goal is None and self.cfg.goal_space == 'simplified_point_mass_maze':
                batch.goal = batch.obs[:, :2]
            assert batch.goal is not None
            backward_input = batch.goal
            goal = batch.goal
            future_goal = batch.future_goal

        perm = torch.randperm(self.cfg.batch_size)
        backward_input = backward_input[perm]

        if self.cfg.mix_ratio > 0:
            mix_idxs: tp.Any = np.where(np.random.uniform(size=self.cfg.batch_size) < self.cfg.mix_ratio)[0]
            if not self.cfg.rand_weight:
                with torch.no_grad():
                    mix_z = self.backward_net(backward_input[mix_idxs]).detach()
            else:
                # generate random weight
                weight = torch.rand(size=(mix_idxs.shape[0], self.cfg.batch_size)).to(self.cfg.device)
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
            future_idxs = np.where(np.random.uniform(size=self.cfg.batch_size) < self.cfg.future_ratio)
            z[future_idxs] = self.backward_net(future_goal[future_idxs]).detach()

        metrics.update(self.update_fb(obs=obs, action=action, discount=discount,
                                      next_obs=next_obs, next_goal=next_goal, z=z, step=step,
                                      goal=goal))

        # update high expl actor
        if self.cfg.uncertainty and self.cfg.myopic and not self.cfg.sampling:
            metrics.update(self.update_high_expl_actor(obs, step))

        if self.cfg.rnd:
            metrics.update(self.update_Qrnd(obs=obs, action=action, discount=discount, next_obs=next_obs, z=z, step=step))

        # update actor
        metrics.update(self.update_actor(obs, z, step))

        # compute disagr
        metrics.update(self.compute_disagreement_metrics())

        # update critic target
        if self.cfg.one_target and self.cfg.uncertainty:
            utils.soft_update_params_mean(self.forward_net, self.forward_target_net,
                                          self.cfg.fb_target_tau)
        else:
            utils.soft_update_params(self.forward_net, self.forward_target_net,
                                     self.cfg.fb_target_tau)
        utils.soft_update_params(self.backward_net, self.backward_target_net,
                                 self.cfg.fb_target_tau)

        if self.cfg.rnd:
            utils.soft_update_params(self.Q_rnd, self.target_Q_rnd,
                                     self.cfg.fb_target_tau)
        return metrics
