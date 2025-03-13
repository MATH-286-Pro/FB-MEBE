# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import functools
import typing as tp


import numpy as np
from url_benchmark import dmc

import torch


F = tp.TypeVar("F", bound=tp.Callable[..., np.ndarray])


class Register(tp.Generic[F]):

    def __init__(self) -> None:
        self.funcs: tp.Dict[str, tp.Dict[str, F]] = {}

    def __call__(self, name: str) -> tp.Callable[[F], F]:
        return functools.partial(self._register, name=name)

    def _register(self, func: F, name: str) -> F:
        fname = func.__name__
        subdict = self.funcs.setdefault(name, {})
        if fname in subdict:
            raise ValueError(f"Already registered a function {fname} for {name}")
        subdict[fname] = func
        return func


goal_spaces: Register[tp.Callable[[dmc.EnvWrapper], np.ndarray]] = Register()
goals: Register[tp.Callable[[], np.ndarray]] = Register()


# # # # #
# goal spaces, defined on one environment to specify:
# # # # #

# pylint: disable=function-redefined


@goal_spaces("quadruped")
def simplified_quadruped(env: dmc.EnvWrapper) -> np.ndarray:
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/quadruped.py#L145
    return np.array([env.physics.torso_upright(),
                     np.linalg.norm(env.physics.torso_velocity())],
                    dtype=np.float32)


@goal_spaces("quadruped")
def simplified_quadruped_velx(env: dmc.EnvWrapper) -> np.ndarray:
    # check the physics here:
    # https://github.com/deepmind/dm_control/blob/d72c22f3bb89178bff38728957daf62965632c2f/dm_control/suite/quadruped.py#L145
    return np.array([env.physics.torso_upright(),
                     env.physics.torso_velocity()[0]],
                    dtype=np.float32)


# # # # #
# goals, defined on one goal_space to specify:
# # # # #


@goals("simplified_quadruped_velx")
def quadruped_walk() -> np.ndarray:
    return np.array([1.0, 0.6], dtype=np.float32)


# # # Custom Reward # # #


def _make_env(domain: str) -> dmc.EnvWrapper:
    # TODO for manipulator, goal space depends on task (peg or ball). Using ball for now but needs to be adapted!
    task = {"quadruped": "stand",
            "walker": "walk",
            "jaco": "reach_top_left",
            "point_mass_maze": "reach_bottom_right",
            "manipulator": "bring_ball",
            "hopper": "hop",
            "humanoid": "walk",
            "cheetah": "walk"}[domain]
    return dmc.make(f"{domain}_{task}", obs_type="states", frame_stack=1, action_repeat=1, seed=12)


def get_goal_space_dim(name: str) -> int:
    domain = {space: domain for domain, spaces in goal_spaces.funcs.items() for space in spaces}[name]
    env = _make_env(domain)
    return goal_spaces.funcs[domain][name](env).size


class BaseReward:

    def __init__(self, seed: tp.Optional[int] = None) -> None:
        self._env: dmc.EnvWrapper  # to be instantiated in subclasses
        self._rng = np.random.RandomState(seed)

    def get_goal(self, goal_space: str) -> np.ndarray:
        raise NotImplementedError

    def from_physics(self, physics: np.ndarray) -> float:
        "careful this is not threadsafe"
        with self._env.physics.reset_context():
            self._env.physics.set_state(physics)
        return self.from_env(self._env)

    def from_env(self, env: dmc.EnvWrapper) -> float:
        raise NotImplementedError


def get_reward_function(name: str, seed: tp.Optional[int] = None) -> BaseReward:
    if name == "quadruped_mix":
        return QuadrupedReward(seed)
    # deleted rest for simplicity
    return DmcReward(name)


def _inv(distance: float) -> float:
    # print("dist", distance)
    return 1 / (1 + abs(distance))


class DmcReward(BaseReward):

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        if name.startswith('ball_in_cup'):
            env_name = 'ball_in_cup'
            _, _, _, task_name = name.split('_', 3)
        else:
            env_name, task_name = name.split("_", maxsplit=1)
        try:
            from dm_control import suite  # import
            from url_benchmark import custom_dmc_tasks as cdmc
        except ImportError as e:
            raise dmc.UnsupportedPlatform("DMC does not run on Mac") from e
        make = suite.load if (env_name, task_name) in suite.ALL_TASKS else cdmc.make
        self._env = make(env_name, task_name)

    def from_env(self, env: dmc.EnvWrapper) -> float:
        return float(self._env.task.get_reward(env.physics))


class QuadrupedReward(BaseReward):

    NUM_CASES = 7

    def __init__(self, seed: tp.Optional[int] = None) -> None:
        super().__init__(seed)
        self._env = _make_env("quadruped")
        self.x = self._rng.uniform(-5, 5, size=2)
        self.vx = self._rng.uniform(-3, 3, size=2)
        self.quadrant = self._rng.choice([1, -1], size=2, replace=True)
        self.speed = float(np.linalg.norm(self.vx))
        self._case = self._rng.randint(self.NUM_CASES)

    def from_env(self, env: dmc.EnvWrapper) -> float:
        # x = env.physics.named.data.xpos["torso"][:2]
        x = env.physics.named.data.site_xpos['workspace'][:2]
        vx = env.physics.torso_velocity()[:2]
        up = max(0, float(env.physics.torso_upright()))
        speed = float(np.linalg.norm(vx))
        if not self._case:   # specific speed norm
            return up * _inv(speed - self.speed)
        if self._case == 1:  # specific position
            return up * _inv(float(np.linalg.norm(x - self.x)))
        if self._case == 2:  # specific quadrant
            return up * float(np.all(x * self.quadrant > self.x))
        if self._case == 3:  # specific quadrant and speed norm
            return up * float(np.all(x * self.quadrant > self.x)) * _inv(self.speed - speed)
        if self._case == 4:  # specific speed
            return up * _inv(np.linalg.norm(self.vx - vx) / np.sqrt(2))
        if self._case == 5:  # specific quadrant and sufficient speed
            return up * float(np.all(x * self.quadrant > self.x)) * (speed > self.speed)
        if self._case == 6:  # sufficient speed
            return up * (speed > self.speed)
        else:
            raise ValueError(f"No case #{self._case}")
