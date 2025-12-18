# ############ Metamotivo Dependency #############
# from metamotivo.fb_cpr import FBcprAgent, FBcprAgentConfig
# from metamotivo.fb import FBAgent

############ Hydra Dependency #############
import os, sys
import hydra
import omegaconf as omgcf

# read hydra config
config_dir = os.path.join(os.path.dirname(__file__), 'configs')

# Parse config name from command line arguments
config_name = 'Isaaclab_fb_play_config_base'
with hydra.initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base="1.1"):
    play_cfg = hydra.compose(config_name=config_name)

with hydra.initialize_config_dir(config_dir=os.path.abspath(play_cfg.path), version_base="1.1"):
    hydra_cfg = hydra.compose(config_name="hydra_config")

app_cfg = {
    "headless":       play_cfg.env.headless,
    "device":         hydra_cfg.env.device,
    "enable_cameras": hydra_cfg.env.video,
}

############ Launch Isaaclab APP #############
from isaaclab.app import AppLauncher
simulation_app = AppLauncher(app_cfg).app

############ IsaacLab Dependency #############
import gymnasium as gym
from isaaclab.envs import DirectRLEnvCfg
from isaaclab_tasks.utils import load_cfg_from_registry

############ User Dependency #############
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wrapper.wrapper_env import FB_VecEnvWrapper
from wrapper.wrapper_video import RecordVideo_TRAIN_GC, RecordVideo_EVAL_GC

from buffer import DictBuffer # faster buffer implementation
from toolbox.dataclass_pylance import AGENT_CFG
from toolbox.dataclass_metrics import TRAIN_METRICS_CLASS, EVAL_METRICS_CLASS
from toolbox.functions_reward import reward_fn

from agent_crl.agent import FB_CRL_AGENT
from agent_meta.fb.agent import FBAgent as FB_META_Agent

from loader.fb_net_loader import FBPolicyLoader
##########################################
import torch
torch.set_float32_matmul_precision("high")

import random
import time
from pathlib import Path
import numpy as np
import wandb
import datetime
import copy

#########################################

def set_seed_everywhere(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

########################################################################################################################

class WORKSPACE:
    def __init__(self, train_cfg, env_cfg, agent_cfg: AGENT_CFG):

        # 创建变量
        self.env_cfg   = env_cfg
        self.agent_cfg = agent_cfg
        self.train_cfg = train_cfg
        self.device    = hydra_cfg.env.device

        self._default_command = {
            "lin_vel": torch.tensor([0.0, 0.0, 0.0], device=self.device),
            "ang_vel": torch.tensor([0.0, 0.0, 0.0], device=self.device),
            "gravity": torch.tensor([0.0, 0.0, -1.0], device=self.device),
            "height": 0.28,
        }

        # 创建保存路径
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.work_dir = Path.cwd() / 'exp_local' / 'fb_mod' / hydra_cfg.env.task / f"{self.timestamp}_play"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # 实例化 class
        # 0.创建 wandb
        if play_cfg.wandb.use_wandb:
            wandb.init(entity  = hydra_cfg.wandb.entity, 
                       project = hydra_cfg.wandb.project,
                       group   = hydra_cfg.wandb.group, 
                       name    = f"{hydra_cfg.wandb.name}_{self.timestamp}",
                       config  = omgcf.OmegaConf.to_container(hydra_cfg, resolve=True))  # type: ignore

        # 1.创建 env
        self.env = gym.make(
            hydra_cfg.env.task,
            cfg = self.env_cfg,
            render_mode = 'rgb_array' if hydra_cfg.env.video else None,
        )
        if hydra_cfg.env.video:
            video_args_eval = {
                'video_folder': str(self.work_dir / 'videos_eval'),
                'step_trigger': lambda step: step % 250 == 0,
                'video_length': hydra_cfg.env.video_length,
                'name_prefix': 'eval',
                'disable_logger': True,
                'use_wandb': play_cfg.wandb.use_wandb,
            }
            self.eval_env  = RecordVideo_EVAL_GC(self.env, **video_args_eval)
            self.eval_env  = FB_VecEnvWrapper(self.eval_env)  # type: ignore
        
        else:
            self.eval_env  = FB_VecEnvWrapper(self.env)  # type: ignore

        # 2.创建 agent
        self.agent = FBPolicyLoader(path=f"{play_cfg.path}/models/model_step_150000.pt", device=self.device)

        # 3.创建 buffer
        self.replay_buffer = {
            "train": DictBuffer(self.train_cfg.replay_buffer_capacity, self.device)
        }
        data = torch.load(f"{play_cfg.path}/models/replay_buffer_step_150000.pt", weights_only=True)
        self.replay_buffer['train'].extend(data)
        del data

        # 4.创建各种记录器
        self.train_metrics = TRAIN_METRICS_CLASS()
        self.eval_metrics  = EVAL_METRICS_CLASS()


    def close(self):
        self.eval_env.close()
        wandb.finish()

    @property
    def time(self):
        if not hasattr(self, '_start_time'):
            self._start_time = time.time()
            return f"00:00:00"
        else:
            elapsed = time.time() - self._start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
    ####################################################################################################################
    def eval(self):
        self.env.unwrapped.set_debug_vis(True)  # type: ignore
        print(f"T:{self.time} Start Evaluation")
        self.eval_task([0.0, 0.0, 0.0])           
        self.eval_task([+0.5, 0.0, 0.0])
        self.eval_task([+1.0, 0.0, 0.0])
        self.eval_task([-1.0, 0.0, 0.0])
        self.eval_task([0.0, +0.5, 0.0])
        self.eval_task([0.0, 0.0, +1.0])
        print("Mean Rewards: ", self.eval_metrics.get_tasks_episode_reward())

        if play_cfg.wandb.use_wandb:
            wandb.log(self.eval_metrics.get_tasks_metrics(), step = 0)
        self.eval_metrics.clear()


    def eval_task(self, command_xyw):

        self.eval_env.eval_task(command_xyw)

        with torch.no_grad():

            # 推理阶段
            data = self.replay_buffer['train'].sample(self.train_cfg.num_eval_sample)
            obs  = data['raw']
            goal = data['goal'].detach().clone()  # no noise
            Z_Bs = self.agent.backward_map(goal)

            command = copy.deepcopy(self._default_command)
            command['lin_vel'][:2] = torch.tensor(command_xyw[0:2], device=self.device)
            command['ang_vel'][2]  = torch.tensor(command_xyw[2],   device=self.device)

            reward = reward_fn(obs, command)

            z_r = self.agent.reward_inference(Z_Bs, reward)
            z_r = z_r.expand(self.eval_env.num_envs, -1)  # type: ignore

            # 开始测试
            td, _ = self.eval_env.reset()

            for _ in range(250):
                obs = td['obs']
                action = self.agent.act(obs['policy'][:, :self.agent_cfg.model.policy_dim], z_r, mean=True)
                td, _, _, _, info = self.eval_env.step(action)
                
                self.eval_metrics.update_rew(f"{command_xyw}", td['reward_task'], info['rew_dict'])
                self.eval_metrics.update_reg(f"{command_xyw}", info['reg_dict'])
        
        if hydra_cfg.env.video:
            self.eval_env.stop_recording(f"{command_xyw}", step=250)
        self.eval_env.train_task()

        print(f"T:{self.time} | Task: {command_xyw}")

########################################################################################################################

def env_cfg_from_hydra(hydra_cfg) -> DirectRLEnvCfg:
    env_cfg = load_cfg_from_registry(hydra_cfg.env.task, "env_cfg_entry_point")
    env_cfg.sim.device     = hydra_cfg.env.device    # type: ignore
    env_cfg.scene.num_envs = play_cfg.env.num_envs  # type: ignore
    env_cfg.seed           = play_cfg.env.seed      # type: ignore
    return env_cfg                                   # type: ignore

def main():

    set_seed_everywhere(hydra_cfg.env.seed)

    env_cfg: DirectRLEnvCfg = env_cfg_from_hydra(hydra_cfg)

    agent_cfg = hydra_cfg.agent
    train_cfg = hydra_cfg.train

    ws = WORKSPACE(train_cfg, env_cfg, agent_cfg)
    ws.eval()
    ws.close()
    
    simulation_app.close()

if __name__ == '__main__':
    main()
