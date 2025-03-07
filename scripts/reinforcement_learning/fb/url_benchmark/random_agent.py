# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import pdb  # pylint: disable=unused-import
import logging
import dataclasses
import typing as tp
import warnings
from pathlib import Path
import sys

warnings.filterwarnings('ignore', category=DeprecationWarning)


os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# if the default egl does not work, you may want to try:
# export MUJOCO_GL=glfw
os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import torch
import wandb
import omegaconf as omgcf
# from dm_env import specs

from url_benchmark import dmc
from dm_env import specs
from url_benchmark import utils
from url_benchmark import goals as _goals
from url_benchmark.logger import Logger
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.video import TrainVideoRecorder, VideoRecorder
from url_benchmark import agent as agents
from url_benchmark.d4rl_benchmark import D4RLReplayBufferBuilder, D4RLWrapper
from url_benchmark.gridworld.env import build_gridworld_task

logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True
# os.environ['WANDB_MODE']='offline'

# from url_benchmark.dmc_benchmark import PRIMAL_TASKS


# # # Config # # #

@dataclasses.dataclass
class Config:
    agent: tp.Any
    # misc
    seed: int = 1
    device: str = "cuda"
    save_video: bool = False
    use_tb: bool = False
    use_wandb: bool = False
    # experiment
    experiment: str = "online"
    # task settings
    task: str = "walker_stand"
    obs_type: str = "states"  # [states, pixels]
    frame_stack: int = 3  # only works if obs_type=pixels
    action_repeat: int = 1  # set to 2 for pixels
    discount: float = 0.99
    future: float = 0.99  # discount of future sampling, future=1 means no future sampling
    goal_space: tp.Optional[str] = None
    append_goal_to_observation: bool = False
    # eval
    num_eval_episodes: int = 10
    custom_reward: tp.Optional[str] = None  # activates custom eval if not None
    final_tests: int = 10
    # checkpoint
    snapshot_at: tp.Tuple[int, ...] = (100000, 200000, 500000, 800000, 1000000, 1500000,
                                       2000000, 3000000, 4000000, 5000000, 9000000, 10000000)
    checkpoint_every: int = 100000
    load_model: tp.Optional[str] = None
    # training
    num_seed_frames: int = 4000
    replay_buffer_episodes: int = 5000
    update_encoder: bool = True
    batch_size: int = omgcf.II("agent.batch_size")
    uncertainty: bool = False
    update_every_steps: int = 1
    num_agent_updates: int = 1
    warmup: bool = True
    pretrain_update_steps: int = 1000
    # to avoid hydra issues
    project_dir: str = ""
    results_dir: str = ""
    id: int = 0
    working_dir: str = ""


@dataclasses.dataclass
class PretrainConfig(Config):
    # mode
    reward_free: bool = True
    # train settings
    num_train_frames: int = 2000010
    # snapshot
    eval_every_frames: int = 10000
    load_replay_buffer: tp.Optional[str] = None
    # replay buffer
    # replay_buffer_num_workers: int = 4
    # nstep: int = omgcf.II("agent.nstep")
    # misc
    save_train_video: bool = False


# loaded as base_pretrain in pretrain.yaml
# we keep the yaml since it's easier to configure plugins from it
# Name the PretrainConfig as "workspace_config".
# When we load workspace_config it in the main config, we are telling it to load: PretrainConfig.
ConfigStore.instance().store(name="workspace_config", node=PretrainConfig)


# # # Implem # # #



C = tp.TypeVar("C", bound=Config)

class BaseWorkspace(tp.Generic[C]):
    def __init__(self, cfg: C) -> None:
        self.work_dir = Path.cwd() if len(cfg.working_dir) == 0 else Path(cfg.working_dir)
        self.model_dir = self.work_dir if 'cluster' not in str(self.work_dir) else Path(str(self.work_dir).replace('home', 'project/hilliges'))
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
        self.device = torch.device(cfg.device)
        task = cfg.task
        if task.startswith('point_mass_maze'):
            self.domain = 'point_mass_maze'
        else:
            self.domain = task.split('_', maxsplit=1)[0]

        self.train_env = self._make_env()
        self.eval_env = self._make_env()
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb,
                             )

        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, self.domain, str(cfg.id)
            ])
            wandb.init(project="controllable_agent", group=cfg.experiment, name=exp_name,  # mode="disabled",
                       config=omgcf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), dir=self.work_dir)  # type: ignore
        if cfg.goal_space is not None:
            if cfg.goal_space not in _goals.goal_spaces.funcs[self.domain]:
                raise ValueError(f"Unregistered goal space {cfg.goal_space} for domain {self.domain}")

        self.replay_loader = ReplayBuffer(max_episodes=cfg.replay_buffer_episodes, discount=cfg.discount, future=cfg.future)

        cam_id = 0 if 'quadruped' not in self.domain else 2

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None,
                                            camera_id=cam_id, use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self.global_step = 0
        self.global_episode = 0
        self.eval_rewards_history: tp.List[float] = []
        self._checkpoint_filepath = self.model_dir / "models" / "latest.pt"
        # This is for continuing training in case workdir is the same
        if self._checkpoint_filepath.exists():
            self.load_checkpoint(self._checkpoint_filepath)
        elif cfg.load_model is not None:
            self.load_checkpoint(cfg.load_model, exclude=["replay_loader"])

        self.reward_cls: tp.Optional[_goals.BaseReward] = None
        if self.cfg.custom_reward == "maze_multi_goal":
            self.reward_cls = self._make_custom_reward(seed=self.cfg.seed)

    def _make_env(self) -> dmc.EnvWrapper:
        cfg = self.cfg
        if self.domain == "grid":
            return dmc.EnvWrapper(build_gridworld_task(self.cfg.task.split('_')[1]))
        if self.domain == "d4rl":
            import d4rl  # type: ignore # pylint: disable=unused-import
            import gym
            return dmc.EnvWrapper(D4RLWrapper(gym.make(self.cfg.task.split('_')[1])))
        return dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed,
                        goal_space=cfg.goal_space, append_goal_to_observation=cfg.append_goal_to_observation)

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    def _make_custom_reward(self, seed: int) -> tp.Optional[_goals.BaseReward]:
        """Creates a custom reward function if provided in configuration
        else returns None
        """
        if self.cfg.custom_reward is None:
            return None
        return _goals.get_reward_function(self.cfg.custom_reward, seed)


    _CHECKPOINTED_KEYS = ("replay_loader",)

    def save_checkpoint(self, fp: tp.Union[Path, str], exclude: tp.Sequence[str] = ()) -> None:
        logger.info(f"Saving checkpoint to {fp}")
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        fp = Path(fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        assert isinstance(self.replay_loader, ReplayBuffer), "Is this buffer designed for checkpointing?"
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
        if isinstance(payload, ReplayBuffer):  # compatibility with pure buffers pickles
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
            elif name == "replay_loader":
                assert isinstance(val, ReplayBuffer)
                # pylint: disable=protected-access
                # drop unecessary meta which could make a mess
                val._current_episode.clear()  # make sure we can start over
                val._future = self.cfg.future
                val._discount = self.cfg.discount
                # val._max_episodes = len(val._storage["discount"])
                val._idx = len(val._storage["discount"]) % self.cfg.replay_buffer_episodes
                val._full = val._idx == 0
                val._max_episodes = self.cfg.replay_buffer_episodes
                val._episodes_length = np.array([len(array) - 1 for array in val._storage["discount"]], dtype=np.int32)
                self.replay_loader = val
                # TODO:  This leads to out of RAM for now (to be fixed)
                if not self.replay_loader._full:  # if buffer is not full we need to recreate the storage
                    self.replay_loader.prefill(val._storage)
                
            else:
                assert hasattr(self, name)
                setattr(self, name, val)
                if name == "global_episode":
                    logger.warning(f"Reloaded agent at global episode {self.global_episode}")

    def finalize(self) -> None:
        print("Running final test", flush=True)
        repeat = self.cfg.final_tests
        if not repeat:
            return

        if self.cfg.custom_reward == "maze_multi_goal":
            eval_hist = self.eval_rewards_history
            rewards = {}
            self.eval_rewards_history = []
            self.cfg.num_eval_episodes = repeat
            self.eval_maze_goals()
            rewards["rewards"] = self.eval_rewards_history
            self.eval_rewards_history = eval_hist  # restore
        else:
           pass
        self.eval_rewards_history = eval_hist  # restore
        with (self.work_dir / "test_rewards.json").open("w") as f:
            json.dump(rewards, f)


class Workspace(BaseWorkspace[PretrainConfig]):
    def __init__(self, cfg: PretrainConfig) -> None:
        super().__init__(cfg)
        self.train_video_recorder = TrainVideoRecorder(self.work_dir if cfg.save_train_video else None,
                                                       camera_id=self.video_recorder.camera_id, use_wandb=self.cfg.use_wandb)
        if not self._checkpoint_filepath.exists():  # don't relay if there is a checkpoint
            if cfg.load_replay_buffer is not None:
                if self.cfg.task.split('_')[0] == "d4rl":
                    d4rl_replay_buffer_builder = D4RLReplayBufferBuilder()
                    self.replay_storage = d4rl_replay_buffer_builder.prepare_replay_buffer_d4rl(self.train_env, self.agent.init_meta(), self.cfg)
                    self.replay_loader = self.replay_storage
                else:
                    self.load_checkpoint(cfg.load_replay_buffer, only=["replay_loader"])
                if not cfg.warmup:
                    print('\nNot warming up when loading a replay buffer!!!\n')
            else:
                assert not cfg.warmup, "Trying to warmup without a preloaded replay buffer"

    def train(self) -> None:
        import numpy as np
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        episode_step, episode_reward, z_correl = 0, 0.0, 0.0
        time_step = self.train_env.reset()
        self.replay_loader.add(time_step, meta={'z': 1})
        metrics = None
        meta_disagr = []
        physics_agg = dmc.PhysicsAggregator()

        while train_until_step(self.global_step):
            if time_step.last():
                self.global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_loader))
                        log('step', self.global_step)
                        if self.cfg.uncertainty and len(meta_disagr) > 0:
                            log('z_disagr', np.mean(meta_disagr))

                        for key, val in physics_agg.dump():
                            log(key, val)
                # reset env
                time_step = self.train_env.reset()
                self.replay_loader.add(time_step, meta={'z': 1})
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshot_at:
                    self.save_checkpoint(self._checkpoint_filepath.with_name(f'snapshot_{self.global_frame}.pt'))
                episode_step = 0
                episode_reward = 0.0
                meta_disagr = []

            # sample action
            action = np.random.uniform(low = self.train_env.action_spec().minimum, high=self.train_env.action_spec().maximum, size = 2)
            # take env step
            time_step = self.train_env.step(action)
            physics_agg.add(self.train_env)
            episode_reward += time_step.reward
            self.replay_loader.add(time_step, meta={'z': 1})
            episode_step += 1
            self.global_step += 1
            # save checkpoint to reload
            if not self.global_frame % self.cfg.checkpoint_every:
                self.save_checkpoint(self._checkpoint_filepath)
        self.save_checkpoint(self._checkpoint_filepath)  # make sure we save the final checkpoint
        self.finalize()


@hydra.main(config_path='.', config_name='base_config', version_base="1.1")
def main(cfg: omgcf.DictConfig) -> None:
    # we assume cfg is a PretrainConfig (but actually not really)
    # calls Config and PretrainConfig
    workspace = Workspace(cfg)  # type: ignore
    workspace.train()


if __name__ == '__main__':
    main()
