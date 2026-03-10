
import torch
from torch import sin, cos
import dataclasses
from functools import wraps

# degree -> radian
D2R = lambda x: torch.deg2rad(torch.tensor(x))

DEFAULT_COMMAND: dict = {
    "vx":  0.0,
    "vy":  0.0,
    "vz":  0.0,
    "wx":  0.0,
    "wy":  0.0,
    "wz":  0.0,
    "gx":  0.0,
    "gy":  0.0,
    "gz": -1.0,
    "base_height": 0.28,
}

@dataclasses.dataclass
class TASK_CFG:
    locomotion:  "LOCOMOTION_CFG" = dataclasses.field(default_factory=lambda: LOCOMOTION_CFG())
    orientation: "ORIENTATION_CFG" = dataclasses.field(default_factory=lambda: ORIENTATION_CFG())


@dataclasses.dataclass
class LOCOMOTION_CFG:
    RANDOM = [
        ["vx", "vy", "wz"],
        [[+1.5, +1.0, +1.0],    # max
         [-1.5, -1.0, -1.0],    # min
         ]   
    ]
    LIST = [
        ["vx", "vy", "wz"],
        [[ 0.0,  0.0,  0.0],
         [+0.5,  0.0,  0.0],
         [+1.0,  0.0,  0.0],
         [+1.5,  0.0,  0.0],
         [-0.5,  0.0,  0.0],
         [-1.0,  0.0,  0.0],
         [-1.5,  0.0,  0.0],
         [+0.5, +0.5,  0.0],
         [+0.5, -0.5,  0.0],
         [-0.5, +0.5,  0.0],
         [-0.5, -0.5,  0.0],
         [ 0.0, +0.5,  0.0],
         [ 0.0, -0.5,  0.0],
         [ 0.0, +1.0,  0.0],
         [ 0.0, -1.0,  0.0],
         [ 0.0,  0.0, +1.0],
         [ 0.0,  0.0, -1.0]]
    ]

@dataclasses.dataclass
class ORIENTATION_CFG:
    LIST = [
        ["gx", "gy", "gz"],
        [
        [-sin(D2R(0)),   sin(D2R(0))*cos(D2R(0)),   -cos(D2R(0))*cos(D2R(0))],      # pitch=0°,   roll=0°
        [-sin(D2R(30)),  sin(D2R(0))*cos(D2R(30)),  -cos(D2R(0))*cos(D2R(30))],     # pitch=+30°, roll=0°
        [-sin(D2R(-30)), sin(D2R(0))*cos(D2R(-30)), -cos(D2R(0))*cos(D2R(-30))],    # pitch=-30°, roll=0°
        [-sin(D2R(0)),   sin(D2R(30))*cos(D2R(0)),  -cos(D2R(30))*cos(D2R(0))],     # pitch=0°,   roll=+30°
        [-sin(D2R(0)),   sin(D2R(-30))*cos(D2R(0)), -cos(D2R(-30))*cos(D2R(0))],    # pitch=0°,   roll=-30°
        [-sin(D2R(30)),  sin(D2R(30))*cos(D2R(30)), -cos(D2R(30))*cos(D2R(30))],    # pitch=+30°, roll=+30°
        [-sin(D2R(-30)), sin(D2R(-30))*cos(D2R(-30)),-cos(D2R(-30))*cos(D2R(-30))], # pitch=-30°, roll=-30°
        [-sin(D2R(60)),  sin(D2R(0))*cos(D2R(60)),  -cos(D2R(0))*cos(D2R(60))],     # pitch=+60°, roll=0°  极端朝向 需要倒立
        [-sin(D2R(-60)), sin(D2R(0))*cos(D2R(-60)), -cos(D2R(0))*cos(D2R(-60))],    # pitch=-60°, roll=0°  极端朝向 需要站立
        [-sin(D2R(80)),  sin(D2R(0))*cos(D2R(80)),  -cos(D2R(0))*cos(D2R(80))],     # pitch=+80°, roll=0°  极端朝向 需要倒立
        [-sin(D2R(-80)), sin(D2R(0))*cos(D2R(-80)), -cos(D2R(0))*cos(D2R(-80))],    # pitch=-80°, roll=0°  极端朝向 需要站立
        ]
    ]
