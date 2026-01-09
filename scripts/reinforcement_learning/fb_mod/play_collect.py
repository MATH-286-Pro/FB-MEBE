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
from tqdm import tqdm

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
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.work_dir = Path(play_cfg.path)
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
            video_args_collect = {
                'video_folder': str(self.work_dir / 'videos_collect'),
                'step_trigger': lambda step: step % 1000 == 0,
                'video_length': hydra_cfg.env.video_length,
                'name_prefix': 'collect',
                'disable_logger': True,
                'use_wandb': play_cfg.wandb.use_wandb,
            }
            self.eval_env  = RecordVideo_TRAIN_GC(self.env, **video_args_collect)
            self.eval_env  = FB_VecEnvWrapper(self.eval_env)  # type: ignore
        
        else:
            self.eval_env  = FB_VecEnvWrapper(self.env)  # type: ignore

        # 2.创建 agent
        self.agent = FBPolicyLoader(path=f"{play_cfg.path}/models/model_step_150000.pt", device=self.device)

        # 3.创建 buffer
        self.replay_buffer = {
            "train": DictBuffer(self.train_cfg.replay_buffer_capacity, self.device)
        }

        # 4.创建各种记录器
        self.train_metrics = TRAIN_METRICS_CLASS()
        self.eval_metrics  = EVAL_METRICS_CLASS()


    def close(self):
        self.eval_env.close()
        wandb.finish()
        
    ####################################################################################################################
    def collect(self):

        td, _ = self.eval_env.reset()
        z = None

        pbar = tqdm(total=self.train_cfg.replay_buffer_capacity, desc="Collecting data")
        while self.replay_buffer['train'].size < self.train_cfg.replay_buffer_capacity:
            pbar.update(self.replay_buffer['train'].size - pbar.n)

            # Environment interaction
            with torch.no_grad():
                obs = td['obs']
                step_count = td['time'].detach().clone()
                z = self.agent.refresh_z(z, step_count)
                action = self.agent.act(obs['policy'], z, mean=False)
                
            new_td, _, terminated, _, new_info = self.eval_env.step(action)
            
            # 添加数据
            # 保证 (s,a,s') 时序，如果 s' 被重置则 td['time'] > new_td['time'] = 0
            indices = ((td['time'] + 1 == new_td['time']) & (td['time'] != 1)).squeeze()
            
            def index_dict_recursively(data_dict, indices):
                indexed_dict = {}
                for key, value in data_dict.items():
                    if isinstance(value, dict):
                        indexed_dict[key] = index_dict_recursively(value, indices)
                    else:
                        indexed_dict[key] = value[indices]
                return indexed_dict
            
            observation_indexed      = index_dict_recursively(td['obs'], indices)
            next_observation_indexed = index_dict_recursively(new_td['obs'], indices)
                
            transition = {
                "observation": copy.deepcopy(observation_indexed),
                "action":      action[indices],
                "next": {
                    "observation": copy.deepcopy(next_observation_indexed),
                    "terminated": terminated[indices].reshape(-1, 1),             # type: ignore
                    "reward_reg": new_td["reward_reg"][indices].reshape(-1, 1),   # type: ignore
                }
            }
            self.replay_buffer["train"].extend(transition)

            # Update td
            td = new_td

        pbar.close()


    def save_offline_data(self):
        buffer_data = self.replay_buffer['train'].get_full_buffer()
        torch.save(buffer_data, self.work_dir / 'offline_data.pt')
        print(f"Offline data saved to {self.work_dir / 'offline_data.pt'}")


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
    ws.collect()
    ws.save_offline_data()
    ws.close()
    
    simulation_app.close()

if __name__ == '__main__':
    main()
