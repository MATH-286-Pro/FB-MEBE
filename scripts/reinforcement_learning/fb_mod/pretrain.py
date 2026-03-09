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
config_name = 'Isaaclab_pretrain_config_base'  # default config
overrides = []
for i, arg in enumerate(sys.argv[1:]):
    if arg.startswith('--config-name='):
        config_name = arg.split('=', 1)[1]
    elif i > 0 and sys.argv[i] == '--config-name':
        config_name = arg
    else:
        overrides.append(arg)

with hydra.initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base="1.1"):
    hydra_cfg = hydra.compose(config_name=config_name, overrides=overrides)

app_cfg = {
    "headless": hydra_cfg.env.headless,
    "device":   hydra_cfg.env.device,
    "enable_cameras": hydra_cfg.env.video_train or hydra_cfg.env.video_eval,
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
from buffer import DictBuffer
from toolbox.dataclass_metrics import TRAIN_METRICS_CLASS, EVAL_METRICS_CLASS
from toolbox.dataclass_pylance import AGENT_CFG
from toolbox.functions_entropy import compute_entropy
from toolbox.functions_reward import CMDSampler, RewardFunction

from agent_crl.agent import FB_CRL_AGENT
from agent_meta.fb.agent import FBAgent as FB_META_Agent
from toolbox.functions_visualization import PLOT_VALUE, PLOT_DENSITY

##########################################
import torch
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import random
import time
from pathlib import Path
import numpy as np
import wandb
import datetime
import copy

#################### USER DEFINE VARIABLES #####################
VX = 0
VY = 1
VZ = 2
WX = 3
WY = 4
WZ = 5
GX = 6
GY = 7
GZ = 8
H  = 9

ENTROPY_RANGE_DICT = {
    'vx': [-3.0, 3.0],
    'vy': [-3.0, 3.0],
    'vz': [-3.0, 3.0],
    'wx': [-5.0, 5.0],
    'wy': [-5.0, 5.0],
    'wz': [-5.0, 5.0],
    'gx': [-1.0, +1.0],
    'gy': [-1.0, +1.0],
    'gz': [-1.0, +1.0],
    'h ': [0.0, 1.0],
}

#################### USER DEFINE FUNCTIONS #####################

def set_seed_everywhere(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

##################### MAIN FUNCTIONS ###########################

class WORKSPACE:
    def __init__(self, train_cfg, env_cfg, agent_cfg: AGENT_CFG):

        # record variable and cfg
        self.env_cfg   = env_cfg
        self.agent_cfg = agent_cfg
        self.train_cfg = train_cfg
        self.device    = hydra_cfg.env.device


        # build save_path
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.work_dir = Path.cwd() / f"exp_{hydra_cfg.train.machine}" / 'fb_mod' / hydra_cfg.env.task / hydra_cfg.wandb.group / self.timestamp
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Save hydra_cfg file
        omgcf.OmegaConf.save(config=hydra_cfg, f=self.work_dir / 'hydra_config.yaml')

        # Class
        # 0.create wandb
        if hydra_cfg.wandb.use_wandb:
            wandb.init(entity  = hydra_cfg.wandb.entity, 
                       project = hydra_cfg.wandb.project,
                       group   = hydra_cfg.wandb.group, 
                       name    = f"{hydra_cfg.wandb.name}_{self.timestamp}",
                       config  = omgcf.OmegaConf.to_container(hydra_cfg, resolve=True))  # type: ignore

        # 1.create env
        self.env = gym.make(
            hydra_cfg.env.task,
            cfg = self.env_cfg,
            render_mode = 'rgb_array' if (hydra_cfg.env.video_train or hydra_cfg.env.video_eval) else None,
        )
        
        # Setup train environment with video recording if enabled
        if hydra_cfg.env.video_train:
            video_args_pretrain = {
                'video_folder': str(self.work_dir / 'videos_pretrain'),
                'step_trigger': lambda step: step % hydra_cfg.env.video_train_interval == 0,
                'video_length': hydra_cfg.env.video_train_length,
                'name_prefix': 'pretrain',
                'disable_logger': True,
                'use_wandb': hydra_cfg.wandb.use_wandb,
            }
            self.train_env = RecordVideo_TRAIN_GC(self.env, **video_args_pretrain)
            self.train_env = FB_VecEnvWrapper(self.train_env) # type: ignore
        else:
            self.train_env = FB_VecEnvWrapper(self.env)  # type: ignore
        
        # Setup eval environment with video recording if enabled
        if hydra_cfg.env.video_eval:
            video_args_eval = {
                'video_folder': str(self.work_dir / 'videos_eval'),
                'step_trigger': lambda step: step % hydra_cfg.env.video_eval_interval == 0,
                'video_length': hydra_cfg.env.video_eval_length,
                'name_prefix': 'eval',
                'disable_logger': True,
                'use_wandb': hydra_cfg.wandb.use_wandb,
            }
            self.eval_env  = RecordVideo_EVAL_GC(self.env, **video_args_eval)
            self.eval_env  = FB_VecEnvWrapper(self.eval_env)  # type: ignore
        else:
            self.eval_env  = FB_VecEnvWrapper(self.env)  # type: ignore

        # 2.create agent
        if train_cfg.agent == 'meta':
            agent_cfg_dict = omgcf.OmegaConf.to_container(self.agent_cfg, resolve=True)  # type: ignore
            self.agent = FB_META_Agent(**agent_cfg_dict)  # type: ignore
        elif train_cfg.agent == 'crl':
            self.agent = FB_CRL_AGENT(self.agent_cfg)


        # 3.create buffer
        self.replay_buffer = {
            "train": DictBuffer(self.train_cfg.replay_buffer_capacity, self.device)
        }

        # 4.create metrics recorder
        self.train_metrics = TRAIN_METRICS_CLASS()
        self.eval_metrics  = EVAL_METRICS_CLASS()

        # 创建 cmd sampler
        self.CMD = CMDSampler(device=self.device)

        # 创建 reward function
        self.reward_fn = RewardFunction()

    def close(self):
        self.train_env.close()
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
    def train(self):

        td, _ = self.train_env.reset()
        z = None

        # Initialize timing variables
        self.sim_time    = 0.0
        self.store_time  = 0.0
        self.update_time = 0.0

        for t in range(0, self.train_cfg.num_train_steps+1):

            # Environment interaction
            sim_start = time.time()
            with torch.no_grad():
                obs = td['obs']
                step_count = td['time'].detach().clone()
                z = self.agent.refresh_z(z, step_count)

                if t < self.train_cfg.num_seeding_steps:
                    if t % 10 == 0:
                        action = torch.rand(self.env_cfg.scene.num_envs, self.agent_cfg.model.action_dim, device=self.device) * 2.0 - 1.0
                        # Do not use the random action below
                        # Because the following action is still random even if you set seed
                        # Which breaks the reproducibility
                        # action = torch.tensor(self.train_env.action_space.sample(), device = self.device) * 0.5
                else:
                    action = self.agent.act(obs['policy'], z, mean=False)
                
            new_td, _, terminated, _, new_info = self.train_env.step(action)
            self.sim_time += time.time() - sim_start
            
            # Add transition date
            # ensure (s,a,s') time sequence
            # If s' is reseted, then td['time'] > new_td['time'] = 0, and it's not valid data
            store_start = time.time()
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
            self.store_time += time.time() - store_start

            # Update agent
            if t % self.train_cfg.interval_update == 0 and t >= self.train_cfg.num_seeding_steps:
                update_start = time.time()
                for _ in range(self.train_cfg.num_updates):
                    metrics_pretrain = self.agent.update(self.replay_buffer, t)
                    metrics_reg      = self.train_env.unwrapped.get_env_metrics() # type: ignore
                    self.train_metrics.update(metrics_pretrain, key_prefix="pretrain/")
                    self.train_metrics.update(metrics_reg, key_prefix="regularization/")
                print(f"Update time: {time.time() - update_start:.2f} seconds", end='\r')
                self.update_time += time.time() - update_start
            
                # Upload metrics
                if t % self.train_cfg.interval_log == 0:
                    if hydra_cfg.wandb.use_wandb:
                        wandb.log(self.train_metrics.mean, step=t)
                    self.train_metrics.clear()

                    # upload normalizer data
                    if hydra_cfg.wandb.use_wandb and self.agent_cfg.model.norm_obs:
                        means = self.agent._model._policy_normalizer.mean
                        stds  = self.agent._model._policy_normalizer.std

                        RANGE = len(means)
                        # pad indices with leading zeros so wandb sorts keys numerically
                        width = len(str(RANGE))
                        for i in range(RANGE):
                            idx = f"{i:0{width}d}"
                            wandb.log({f"normalizer/policy/mean_{idx}": means[i].item()}, step=t)
                            wandb.log({f"normalizer/policy/std_{idx}":  stds[i].item()},  step=t)

            # 评估逻辑
            # 采样 cmd
            # 根据 cmd +  reward_fn 推理不同任务下的奖励
            # 计算 Z_Rs
            # 环境交互
            #    使用 reward_fn + cmd 计算 episode return
            #    注意 reg reward 计算



            # Evaluate agent
            if self.train_cfg.eval and t % self.train_cfg.interval_eval == 0 and t >= self.train_cfg.num_seeding_steps and self.replay_buffer["train"].size >= self.train_cfg.num_eval_sample:
                self.env.unwrapped.set_debug_vis(True)  # type: ignore     
                self.eval('locomotion', 'list')
                self.eval('locomotion', 'random')
                self.eval('orientation', 'list')
                self.env.unwrapped.set_debug_vis(False)  # type: ignore

                if hydra_cfg.wandb.use_wandb:
                    wandb.log(self.eval_metrics.get_tasks_metrics(), step = t)
                self.eval_metrics.clear()

                # # run 300 frames to prevent massive robots reset
                # if hydra_cfg.env.video_train:
                #     step_id = self.train_env.env.step_id
                    
                # for _ in range(300):
                #     obs = td['obs']
                #     step_count = td['time'].detach().clone()
                #     z = self.agent.refresh_z(z, step_count)
                #     action = self.agent.act(obs['policy'], z, mean=False)
                #     td, _, _, _, _ = self.train_env.step(action)

                # if hydra_cfg.env.video_train:
                #     self.train_env.env.step_id = step_id

                ##############################################
                #####        FB Data Quality Check       #####
                ##### To FB, Data quality is everything! #####
                ##############################################

                # Sample 10k data from replay buffer
                sampled_data = self.replay_buffer["train"].sample(self.train_cfg.save_buffer_size)
                goals = sampled_data['observation']['goal']
                
                # Scatter Plots
                dist_image_vx_vy = PLOT_DENSITY(goals[:, VX], goals[:, VY])
                dist_image_vx_wz = PLOT_DENSITY(goals[:, VX], goals[:, WZ])

                # compute entropy
                entropy_VxVyWz = compute_entropy(goals[:, [VX, VY, WZ]], ENTROPY_RANGE_DICT, bins=50)
                entropy_GxGyGz = compute_entropy(goals[:, [GX, GY, GZ]], ENTROPY_RANGE_DICT, bins=50)

                # Log to wandb
                if hydra_cfg.wandb.use_wandb:
                    wandb.log({
                        "goal_density_vx_vy": wandb.Image(dist_image_vx_vy),
                        "goal_density_vx_wz": wandb.Image(dist_image_vx_wz),
                        "exploration/entropy_VxVyWz": entropy_VxVyWz,
                        "exploration/entropu_GxGyGz": entropy_GxGyGz,
                    }, step=t)
                

            if t % self.train_cfg.interval_log == 0:
                total_time = self.sim_time + self.store_time + self.update_time
                sim_percent    = (self.sim_time / total_time) * 100
                store_percent  = (self.store_time / total_time) * 100
                update_percent = (self.update_time / total_time) * 100
                print(f"| S:{t} | T:{self.time} | Sim:{sim_percent:.1f}% | Store:{store_percent:.1f}% | Update:{update_percent:.1f}% |")

            # Save FB model
            if t > 0 and t % self.train_cfg.interval_save_model == 0:
                model_dir = self.work_dir / 'models'
                model_dir.mkdir(parents=True, exist_ok=True)

                # Save buffer
                buffer_data = self.replay_buffer['train'].sample(self.train_cfg.save_buffer_size)['observation']
                torch.save(buffer_data, model_dir / f'replay_buffer_step_{t}.pt')
                del buffer_data

                # Save model)
                self.agent.save(model_dir / f'model_step_{t}.pt')

    @torch.inference_mode()
    def eval(self, task: str, mode: str):

        # Command
        num_envs  = self.env_cfg.scene.num_envs
        cmd: dict = self.CMD.sample(num_envs, task, mode)
        self.eval_env.eval_task(torch.cat([cmd['vx'], cmd['vy'], cmd['wz']], dim=-1))

        # Inference Phase
        data = self.replay_buffer['train'].sample(self.train_cfg.num_eval_sample)
        obs  = data['observation']['raw']
        goal = data['observation']['goal'].detach().clone()  # no noise

        reward = self.reward_fn.inference(obs, cmd, task)

        z_r = self.agent.reward_inference(goal, reward.T)
        z_r = z_r.expand(self.eval_env.num_envs, -1)  # type: ignore

        # Start Simulation
        td, _ = self.eval_env.reset()

        for _ in range(250):
            obs = td['obs']
            action = self.agent.act(obs['policy'], z_r, mean=True)
            td, _, _, _, info = self.eval_env.step(action)
            
            self.eval_metrics.update(
                task,
                mode,
                cmd['log_index'],
                td['reward_task'],
                info['rew_dict'],
                info['reg_dict']
            )
        
        if hydra_cfg.env.video_eval:
            self.eval_env.stop_recording(f"{task}_{mode}", step=self.train_env.env.step_id)
        self.eval_env.train_task()

########################################################################################################################

def env_cfg_from_hydra(hydra_cfg) -> DirectRLEnvCfg:
    env_cfg = load_cfg_from_registry(hydra_cfg.env.task, "env_cfg_entry_point")
    env_cfg.sim.device     = hydra_cfg.env.device    # type: ignore
    env_cfg.scene.num_envs = hydra_cfg.env.num_envs  # type: ignore
    env_cfg.seed           = hydra_cfg.env.seed      # type: ignore
    env_cfg.viewer.resolution = tuple(hydra_cfg.env.video_resolution)  # type: ignore
    return env_cfg                                   # type: ignore

def main():

    set_seed_everywhere(hydra_cfg.env.seed)

    env_cfg: DirectRLEnvCfg = env_cfg_from_hydra(hydra_cfg)

    # update obs_dim and action_dim for 'agent_cfg', will influence 'agent.model'
    hydra_cfg.agent.model.policy_dim     = env_cfg.policy_space      # type: ignore
    hydra_cfg.agent.model.obs_dim        = env_cfg.observation_space
    hydra_cfg.agent.model.goal_dim       = env_cfg.goal_space        # type: ignore
    hydra_cfg.agent.model.critic_dim     = env_cfg.critic_space      # type: ignore
    hydra_cfg.agent.model.action_dim     = env_cfg.action_space
    hydra_cfg.train.replay_buffer_capacity = hydra_cfg.env.num_envs * hydra_cfg.train.replay_buffer_N
    hydra_cfg.env.action_scale           = env_cfg.action_scale      # type: ignore

    agent_cfg = hydra_cfg.agent
    train_cfg = hydra_cfg.train

    ws = WORKSPACE(train_cfg, env_cfg, agent_cfg)
    ws.train()
    ws.close()
    
    simulation_app.close()

if __name__ == '__main__':
    main()
