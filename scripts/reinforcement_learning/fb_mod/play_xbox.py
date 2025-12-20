# ############ Hydra Dependency #############
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

# Prompt user for simulation device selection (cpu / gpu)
def _select_simulation_device(default: str | None = None) -> str:
    default = "cpu"
    while True:
        resp = input("===============================================================================\n"
                     f"Select simulation device ([c]pu/[g]pu) [default: {default}]: \n"
                     "CPU Mode: can use shift + left-click drag to manipulate the robot \n"
                     "GPU Mode: can not drag \n"
                     "===============================================================================\n").strip().lower()
        if resp == "":
            choice = default
            break
        if resp in ("c", "cpu"):
            choice = "cpu"
            break
        if resp in ("g", "gpu"):
            choice = "cuda"
            break
        print("Invalid choice — enter 'cpu' or 'gpu' (or 'c'/'g').")
    return choice

SIM_DEVICE = _select_simulation_device(getattr(hydra_cfg.env, "device", None))

app_cfg = {
    "headless":       False,  # Force GUI mode for gamepad and camera following
    "device":         SIM_DEVICE,
    "enable_cameras": False,
}

############ Launch Isaaclab APP #############
from isaaclab.app import AppLauncher
simulation_app = AppLauncher(app_cfg).app

# 性能优化设置
import carb
settings = carb.settings.get_settings()
# 降低渲染质量以提高性能
settings.set("/rtx/reflections/enabled", False)
settings.set("/rtx/translucency/enabled", False)
settings.set("/rtx/shadows/enabled", False)
settings.set("/rtx/ambientOcclusion/enabled", False)
settings.set("/rtx/raytracing/fractionalCutoutOpacity/enabled", False)
settings.set("/rtx/post/aa/op", 0)  # Disable anti-aliasing
settings.set("/rtx/post/dlss/execMode", 0)  # Disable DLSS
print("Low quality rendering enabled for better performance")

############ IsaacLab Dependency #############
import gymnasium as gym
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab_tasks.utils import load_cfg_from_registry

############ User Dependency #############
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wrapper.wrapper_env import FB_VecEnvWrapper
from wrapper.wrapper_video import RecordVideo_EVAL_GC

from buffer import DictBuffer # faster buffer implementation
from toolbox.dataclass_pylance import AGENT_CFG
from toolbox.functions_reward import reward_fn

from loader.fb_net_loader import FBPolicyLoader
from loader.xbox import XboxGamepad
##########################################
import torch
torch.set_float32_matmul_precision("high")

import random
import time
import numpy as np
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
        
        self.device    = env_cfg.sim.device
        # agent 设备独立于环境设备，优先使用 CUDA 加速
        self.agent_device = "cuda" if torch.cuda.is_available() else self.device

        self._default_command = {
            "lin_vel": torch.tensor([0.0, 0.0, 0.0], device=self.agent_device),
            "ang_vel": torch.tensor([0.0, 0.0, 0.0], device=self.agent_device),
            "gravity": torch.tensor([0.0, 0.0, -1.0], device=self.agent_device),
            "height": 0.28,
        }

        # 实例化 class
        # 1.创建 env
        self.env = gym.make(
            hydra_cfg.env.task,
            cfg = self.env_cfg,
            render_mode = None,
        )
    
        self.eval_env  = FB_VecEnvWrapper(self.env)  # type: ignore
        self.eval_env.unwrapped.termination_type = "none"
        self.eval_env.unwrapped.set_debug_vis(True)  # type: ignore

        # 2.创建 agent - 使用 fb_net_loader
        self.agent = FBPolicyLoader(path=f"{play_cfg.path}/models/model_step_150000.pt", device=self.agent_device)

        # 3.创建 buffer
        self.replay_buffer = {
            "train": DictBuffer(self.train_cfg.replay_buffer_capacity, self.agent_device)
        }
        data = torch.load(
            f"{play_cfg.path}/models/replay_buffer_step_300000.pt",
            weights_only=True,
            map_location=self.agent_device,
        )
        self.replay_buffer['train'].extend(data)
        del data

        # 5.创建手柄控制器 (XboxGamepad for flexible control)
        self.gamepad = XboxGamepad(dead_zone=0.01)
        
        # 添加重置回调
        self.should_reset = False
        def reset_callback():
            self.should_reset = True
        
        # 使用 X 按钮作为重置键
        import carb.input
        self.gamepad.add_callback(carb.input.GamepadInput.X, reset_callback)
        
        print("=" * 80)
        print("Gamepad Controller Initialized!")
        print(self.gamepad)
        print("=" * 80)
        print("Controls:")
        print("  Left Stick (LX/LY)  : Control linear velocity (vx, vy)")
        print("  Right Stick (RX/RY) : Control angular velocity (yaw) and height")
        print("  D-Pad Up/Down       : Adjust height")
        print("  Triggers (LT/RT)    : Fine control")
        print("  X Button            : Reset environment")
        print("=" * 80)

        # 6.设置镜头跟随机器人
        # 尝试从场景中找到机器人（通常是名为 "robot" 的 articulation）
        viewer_cfg = ViewerCfg(
            eye=(2.5, 2.5, 2.15),
            lookat=(0.0, 0.0, 0.0),
            origin_type="asset_root",
            env_index=0,
            asset_name="robot"  # 假设机器人在场景中的名称为 "robot"
        )
        try:
            self.camera_controller = ViewportCameraController(self.env.unwrapped, viewer_cfg)  # type: ignore
            print("Camera following enabled: tracking robot")
        except Exception as e:
            print(f"Warning: Could not enable camera following: {e}")
            print("Camera will remain in default position")
            self.camera_controller = None

    def close(self):
        self.eval_env.close()
        
    ####################################################################################################################
    def play_with_gamepad(self):

        # 从 replay buffer 中采样用于 reward inference
        with torch.no_grad():
            data = self.replay_buffer['train'].sample(self.train_cfg.num_eval_sample)
            obs  = data['raw']
            goal = data['goal'].detach().clone()

            Z_Bs = self.agent.backward_map(goal)

        # 重置环境
        td, _ = self.eval_env.reset()
        self.gamepad.reset()
                
        print("\nStarting gamepad control loop...")
        
        while simulation_app.is_running():
            # 处理重置请求
            if self.should_reset:
                td, _ = self.eval_env.reset()
                self.gamepad.reset()
                self.should_reset = False
                continue
            
            # 获取手柄输入
            values = self.gamepad.advance()  # returns dict with all button/stick values
            
            # 姿态命令解算
            pitch = values['RY'] * 30.0  # 前后倾斜角度（度）
            roll  = (int(values['RB']) - int(values['LB'])) * 30.0  # 左右倾斜角度（度）
            
            # 将角度转换为弧度
            pitch_rad = np.deg2rad(pitch)
            roll_rad  = np.deg2rad(roll)
            
            # 根据 pitch 和 roll 计算重力在机器人坐标系中的分量
            gx = np.sin(pitch_rad)
            gy = -np.sin(roll_rad) * np.cos(pitch_rad)
            gz = -np.cos(roll_rad) * np.cos(pitch_rad)

            # 将手柄命令转换为机器人命令
            command = copy.deepcopy(self._default_command)
            command['lin_vel'][0] = torch.tensor(+1.0 * values['LY'], device=self.device)  # forward/backward
            command['lin_vel'][1] = torch.tensor(-1.0 * values['LX'], device=self.device)  # left/right
            command['ang_vel'][2] = torch.tensor(-2.0 * values['RX'], device=self.device)  # rotation
            command['gravity'][0] = torch.tensor(gx, device=self.device)
            command['gravity'][1] = torch.tensor(gy, device=self.device)
            command['gravity'][2] = torch.tensor(gz, device=self.device)
            command['height'] = float(command['height']) + values['RT'] * 0.1 - values['LT'] * 0.1  # adjust height with triggers
            
            # 更新 eval_env 任务目标
            self.eval_env.eval_task([values['LY'], -values['LX'], -2.0 * values['RX']])

            # 基于当前命令计算 reward 并推理 z
            reward = reward_fn(obs, command) # type: ignore
            z_r = self.agent.reward_inference(Z_Bs, reward)
            z_r = z_r.expand(self.eval_env.num_envs, -1)
            
            # 使用策略生成动作（将观测转移到 agent_device，动作转回 env device）
            obs_agent = td['obs']['policy'][:, :self.agent_cfg.model.policy_dim].to(self.agent_device)
            action = self.agent.act(obs_agent, z_r, mean=True)
            action = action.to(self.device)  # 转回环境设备
            td, _, _, _, _ = self.eval_env.step(action)
            
########################################################################################################################

def env_cfg_from_hydra(hydra_cfg) -> DirectRLEnvCfg:
    env_cfg = load_cfg_from_registry(hydra_cfg.env.task, "env_cfg_entry_point")
    env_cfg.sim.device     = SIM_DEVICE              # type: ignore
    env_cfg.scene.num_envs = 1                       # type: ignore
    env_cfg.seed           = play_cfg.env.seed       # type: ignore
    return env_cfg                                   # type: ignore

def main():

    set_seed_everywhere(hydra_cfg.env.seed)

    env_cfg: DirectRLEnvCfg = env_cfg_from_hydra(hydra_cfg)

    agent_cfg = hydra_cfg.agent
    train_cfg = hydra_cfg.train

    ws = WORKSPACE(train_cfg, env_cfg, agent_cfg)
    
    # 使用手柄进行交互式演示
    ws.play_with_gamepad()
    
    ws.close()
    simulation_app.close()

if __name__ == '__main__':
    main()
