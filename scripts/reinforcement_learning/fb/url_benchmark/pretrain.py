# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

from isaaclab.app import AppLauncher

# # local imports
# import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# # append RSL-RL cli arguments
# cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from vecenv_wrapper import FBVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import register_task_to_hydra, hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import json
import pdb  # pylint: disable=unused-import
import logging
import dataclasses
import typing as tp
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=DeprecationWarning)


os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# if the default egl does not work, you may want to try:
# export MUJOCO_GL=glfw
os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import wandb
import omegaconf as omgcf
# from dm_env import specs

from url_benchmark import dmc
# from dm_env import specs
from url_benchmark import utils
from url_benchmark import goals as _goals
from url_benchmark.logger import Logger
from url_benchmark.rollout_storage import RolloutStorage
from url_benchmark.video_record import VideoRecorder
from url_benchmark import agent as agents

logger = logging.getLogger(__name__)
# torch.backends.cudnn.benchmark = True
# os.environ['WANDB_MODE']='offline'

# # # Config # # #


@dataclasses.dataclass
class Config:
    agent: tp.Any
    # misc
    seed: int = 1
    device: str = "cuda"
    save_video: bool = False
    use_wandb: bool = False
    # experiment
    experiment: str = "online"
    # task settings
    task: str = "walker_stand"
    obs_type: str = "states"  # [states, pixels]
    frame_stack: int = 3  # only works if obs_type=pixels
    discount: float = 0.99
    future: float = 0.99  # discount of future sampling, future=1 means no future sampling
    goal_space: tp.Optional[str] = None
    append_goal_to_observation: bool = False
    # eval
    num_eval_episodes: int = 10
    custom_reward: tp.Optional[str] = None  # activates custom eval if not None
    final_tests: int = 10
    # checkpoint # num episode * length of episode
    snapshot_at: tp.Tuple[int, ...] = (0, 250, 500, 1000, 1500, 2000)
    checkpoint_every: int = 100000
    load_model: tp.Optional[str] = None
    # training
    num_seed_frames: int = 4000
    update_encoder: bool = True
    uncertainty: bool = False
    update_every_steps: int = 1
    num_agent_updates: int = 1
    # to avoid hydra issues
    project_dir: str = ""
    results_dir: str = ""
    id: int = 0
    working_dir: str = ""
    # mode
    reward_free: bool = True
    # train settings
    num_train_frames: int = 2000010
    # snapshot
    eval_every_frames: int = 10000
    load_replay_buffer: tp.Optional[str] = None
    save_train_video: bool = False


#  Name the Config as "workspace_config".
#  When we load workspace_config in the main config, we are telling it to load: Config.
ConfigStore.instance().store(name="workspace_config", node=Config)


# # # Implem # # #


def make_agent(
    obs_type: str, obs_spec, goal_spec, action_spec, num_expl_steps: int, cfg: omgcf.DictConfig
) -> tp.Union[agents.FBDDPGAgent]:
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.goal_shape = goal_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


C = tp.TypeVar("C", bound=Config)


# def _init_eval_meta(workspace: "BaseWorkspace", custom_reward: tp.Optional[_goals.BaseReward] = None) -> agents.MetaDict:
#     if custom_reward is not None:
#         try:  # if the custom reward implements a goal, return it
#             goal = custom_reward.get_goal(workspace.cfg.goal_space)
#             return workspace.agent.get_goal_meta(goal)
#         except Exception:  # pylint: disable=broad-exceptf
#             pass
#         # TODO Assuming fix episode length:
#         num_steps = workspace.agent.cfg.num_inference_steps  # type: ignore
#         if len(workspace.replay_loader) * workspace.replay_loader._episodes_length[0] < num_steps:
#             # print("Not enough data for inference, skipping eval")
#             return None
#         obs_list, reward_list = [], []
#         batch_size = 0
#         while batch_size < num_steps:
#             batch = workspace.replay_loader.sample(workspace.cfg.batch_size, custom_reward=custom_reward)
#             batch = batch.to(workspace.cfg.device)
#             obs_list.append(batch.next_goal if workspace.cfg.goal_space is not None else batch.next_obs)  # TODO: why next_obs and not obs?
#             reward_list.append(batch.reward)
#             batch_size += batch.next_obs.size(0)
#         obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
#         obs_t, reward_t = obs[:num_steps], reward[:num_steps]
#         return workspace.agent.infer_meta_from_obs_and_rewards(obs_t, reward_t)

#     if workspace.cfg.goal_space is not None:
#         funcs = _goals.goals.funcs.get(workspace.cfg.goal_space, {})
#         if workspace.cfg.task in funcs:
#             g = funcs[workspace.cfg.task]()
#             return workspace.agent.get_goal_meta(g)
#     print('\n***\n inferring eval meta from replay buffer :s\n***\n')
#     return workspace.agent.infer_meta(workspace.replay_loader)


class BaseWorkspace(tp.Generic[C]):
    def __init__(self, cfg: C, env_cfg) -> None:
        self.work_dir = Path.cwd() if len(cfg.working_dir) == 0 else Path(cfg.working_dir)
        self.model_dir = self.work_dir if 'cluster' not in str(self.work_dir) else Path(str(self.work_dir).replace('home', 'scratch'))
        print(f'Workspace: {self.work_dir}')
        print(f'Running code in : {Path(__file__).parent.resolve().absolute()}')
        logger.info(f'Workspace: {self.work_dir}')
        logger.info(f'Running code in : {Path(__file__).parent.resolve().absolute()}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        if not torch.cuda.is_available():
            if cfg.device != "cpu":
                logger.warning(f"Falling back to cpu as {cfg.device} is not available")
                cfg.device = "cpu"
                cfg.agent.device = "cpu"
        self.device = torch.device(cfg.device)

        self.train_env = self._make_env(env_cfg)
        # self.eval_env = self._make_env()
        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec,
                                self.train_env.goal_spec,
                                self.train_env.action_spec,
                                cfg.num_seed_frames,
                                cfg.agent)

        # create logger
        self.logger = Logger(self.work_dir,
                             use_wandb=cfg.use_wandb)

        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, str(cfg.id)
            ])
            wandb.init(project="fb_hw", group=cfg.experiment, name=exp_name,  # mode="disabled",
                       config=omgcf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), dir=self.work_dir)  # type: ignore
        self.num_transitions_per_env = 32
        self.replay_loader = RolloutStorage(num_envs=env_cfg.scene.num_envs, num_transitions_per_env=self.num_transitions_per_env, discount=cfg.discount,
                                            num_obs=int(self.train_env.observation_spec.shape[0]),  # type: ignore
                                            num_goal=int(self.train_env.goal_spec.shape[0]),  # type: ignore
                                            num_actions=int(self.train_env.action_spec.shape[0]),  # type: ignore
                                            num_z=cfg.agent.z_dim,
                                            device=str(self.device))

        self.timer = utils.Timer()
        self.global_step = 0
        self.global_episode = 0
        self.eval_rewards_history: tp.List[float] = []
        self.eval_dist_history: tp.List[float] = []
        self._checkpoint_filepath = self.model_dir / "models" / "latest.pt"
        # This is for continuing training in case workdir is the same
        if self._checkpoint_filepath.exists():
            self.load_checkpoint(self._checkpoint_filepath)
        # This is for loading an existing model
        elif cfg.load_model is not None:
            self.load_checkpoint(cfg.load_model)  # , exclude=["replay_loader"])

    def _make_env(self, env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
        """Train with RSL-RL agent."""
        # override configurations with non-hydra CLI arguments
        # agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        # agent_cfg.max_iterations = (
        #     args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        # )

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        # env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # # specify directory for logging experiments
        # log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        # log_root_path = os.path.abspath(log_root_path)
        # print(f"[INFO] Logging experiment in directory: {log_root_path}")
        # # specify directory for logging runs: {time-stamp}_{run_name}
        # log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # # This way, the Ray Tune workflow can extract experiment name.
        # print(f"Exact experiment name requested from command line: {log_dir}")
        # if agent_cfg.run_name:
        #     log_dir += f"_{agent_cfg.run_name}"
        # log_dir = os.path.join(log_root_path, log_dir)

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if args_cli.video:
            print("[INFO] Recording videos during training.")

        # save resume path before creating a new log_dir
        # if agent_cfg.resume:
        #     resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        # wrap around environment for fb
        env = FBVecEnvWrapper(env)

        return env

    def _make_custom_reward(self, seed: int) -> tp.Optional[_goals.BaseReward]:
        """Creates a custom reward function if provided in configuration
        else returns None
        """
        if self.cfg.custom_reward is None:
            return None
        return _goals.get_reward_function(self.cfg.custom_reward, seed)

    _CHECKPOINTED_KEYS = ('agent', 'global_step', 'global_episode', "replay_loader")

    def save_checkpoint(self, fp: tp.Union[Path, str], exclude: tp.Sequence[str] = ()) -> None:
        logger.info(f"Saving checkpoint to {fp}")
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        fp = Path(fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        assert isinstance(self.replay_loader, RolloutStorage), "Is this buffer designed for checkpointing?"
        # this is just a dumb security check to not forget about it
        payload = {k: self.__dict__[k] for k in self._CHECKPOINTED_KEYS if k not in exclude}
        with fp.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)

    def load_checkpoint(self, fp: tp.Union[Path, str], only: tp.Optional[tp.Sequence[str]] = None, exclude: tp.Sequence[str] = ()) -> None:
        """Reloads a checkpoint or part of it

        Parameters
        ----------
        only: None or sequence of str
            reloads only a specific subset (defaults to all)
        exclude: sequence of str
            does not reload the provided keys
        """
        print(f"loading checkpoint from {fp}")
        fp = Path(fp)
        with fp.open('rb') as f:
            payload = torch.load(f)
        if isinstance(payload, RolloutStorage):  # compatibility with pure buffers pickles
            payload = {"replay_loader": payload}
        if only is not None:
            only = list(only)
            assert all(x in self._CHECKPOINTED_KEYS for x in only)
            payload = {x: payload[x] for x in only}
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        for x in exclude:
            payload.pop(x, None)
        for name, val in payload.items():
            logger.info("Reloading %s from %s", name, fp)
            if name == "agent":
                self.agent.init_from(val)
            else:
                assert hasattr(self, name)
                setattr(self, name, val)
                if name == "global_episode":
                    logger.warning(f"Reloaded agent at global episode {self.global_episode}")


class Workspace(BaseWorkspace[Config]):
    def __init__(self, cfg: Config, env_cfg) -> None:
        super().__init__(cfg, env_cfg)
        self.train_video_recorder = VideoRecorder(self.train_env, str(self.work_dir), video_interval=args_cli.video_interval, video_length=args_cli.video_length, wandb=self.cfg.use_wandb, enabled=args_cli.video)
        if not self._checkpoint_filepath.exists():  # don't relay if there is a checkpoint
            if cfg.load_replay_buffer is not None:
                self.load_checkpoint(cfg.load_replay_buffer, only=["replay_loader"])

    def train(self) -> None:
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames)
        seed_until_step = utils.Until(self.num_transitions_per_env)
        eval_every_step = utils.Every(self.cfg.eval_every_frames)
        update_every_step = utils.Every(self.num_transitions_per_env)
        log_every = self.num_transitions_per_env * 100
        log_every_step = utils.Every(log_every)

        time_step = self.train_env.reset()
        meta = self.agent.init_meta(time_step.observation)
        metrics = None

        while train_until_step(self.global_step):
            # try to update the agent: only if we collected enough random data (seed until step steps)
            # and if we collected update_every_step steps
            if not seed_until_step(self.global_step) and update_every_step(self.global_step):
                # Num of updates is managed by update function now!
                metrics = self.agent.update(self.replay_loader, self.global_step)
                if log_every_step(self.global_step):
                    self.logger.log_metrics(metrics, step=self.global_step, ty='train')
                    self.logger.dump(step=self.global_step, ty='train')

            meta = self.agent.update_meta(meta, self.train_env.episode_length_buf,
                                          obs=time_step.observation)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)
            # take env step
            time_step = self.train_env.step(action)
            self.replay_loader.add_transitions(time_step, meta)
            self.global_step += 1
            self.train_video_recorder.step(self.global_step)

            # save checkpoint to reload
            if not self.global_step % self.cfg.checkpoint_every:
                self.save_checkpoint(self._checkpoint_filepath.with_name(f'snapshot_ep{self.global_episode}_frame{self.global_step}.pt'))

        self.save_checkpoint(self._checkpoint_filepath)  # make sure we save the final checkpoint
        self.train_env.close()


@hydra.main(config_path='configs', config_name='base_config', version_base="1.1")
def main(cfg: omgcf.DictConfig) -> None:
    env_cfg, _ = register_task_to_hydra(args_cli.task, 'rsl_rl_cfg_entry_point')
    # calls Config
    workspace = Workspace(cfg, env_cfg)  # type: ignore
    workspace.train()


if __name__ == '__main__':
    main()
