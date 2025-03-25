import torch
from vecenv_wrapper import ExtendedGoalTimeStep, ExtendedTimeStep
import typing as tp
import dataclasses
import numpy as np

T = tp.TypeVar("T", np.ndarray, torch.Tensor)


@dataclasses.dataclass
class EpisodeBatch(tp.Generic[T]):
    """For later use
    A container for batchable replayed episodes
    """
    obs: T
    action: T
    reward: T
    next_obs: T
    discount: T
    meta: tp.Dict[str, T] = dataclasses.field(default_factory=dict)
    _physics: tp.Optional[T] = None
    goal: tp.Optional[T] = None
    next_goal: tp.Optional[T] = None
    future_obs: tp.Optional[T] = None
    future_goal: tp.Optional[T] = None


class RolloutStorage:
    # class Transition:
    #     def __init__(self):
    #         self.observations = None
    #         self.critic_observations = None
    #         self.actions = None
    #         self.rewards = None
    #         self.dones = None
    #         self.values = None
    #         self.actions_log_prob = None
    #         self.action_mean = None
    #         self.action_sigma = None

    #     def clear(self):
    #         self.__init__()

    def __init__(self,
                 num_envs,
                 num_transitions_per_env,
                 discount,
                 num_obs,
                 num_goal,
                 num_actions,
                 num_z,
                 device='cpu') -> None:

        self.device = device

        self.num_obs = num_obs  # obs dim
        self.num_actions = num_actions
        self.num_z = num_z
        self.discount = discount
        self.num_goal = num_goal  # goal dim

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, num_obs, device=self.device)
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, num_obs, device=self.device)
        self.goals = torch.zeros(num_transitions_per_env, num_envs, num_goal, device=self.device)
        self.next_goals = torch.zeros(num_transitions_per_env, num_envs, num_goal, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, num_actions, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.meta = torch.zeros(num_transitions_per_env, num_envs, num_z, device=self.device)
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, transition: ExtendedTimeStep, meta: tp.Mapping[str, tp.Any]) -> None:
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observation)
        self.next_observations[self.step].copy_(transition.next_observation)

        self.goals[self.step].copy_(transition.goal)
        self.next_goals[self.step].copy_(transition.next_goal)
        self.actions[self.step].copy_(transition.action)
        self.dones[self.step].copy_(transition.done.view(-1, 1))
        self.rewards[self.step].copy_(transition.reward.view(-1, 1))
        self.meta[self.step].copy_(meta['z'])
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        # indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        goals = self.goals.flatten(0, 1)
        next_goals = self.next_goals.flatten(0, 1)
        dones = self.dones.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        rewards = self.rewards.flatten(0, 1)
        indices = torch.arange(0, num_mini_batches * mini_batch_size, requires_grad=False, device=self.device).view(-1, 1)
        # remove from indices the indices of dones obs and the  ones between diff environments
        indices = indices[dones == 0]
        indices = indices[indices % (self.num_transitions_per_env - 1) != 0]

        # permute
        p = torch.randperm(len(indices))
        indices = indices[p]

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                next_obs_batch = next_observations[batch_idx]
                goal_batch = goals[batch_idx]
                next_goal_batch = next_goals[batch_idx]
                action_batch = actions[batch_idx]
                rew_batch = rewards[batch_idx]
                discount_batch = self.discount * torch.ones_like(rew_batch)
                future_obs = None
                future_goal = None

                yield EpisodeBatch(obs=obs_batch, goal=goal_batch, action=action_batch, reward=rew_batch, discount=discount_batch,
                                   next_obs=next_obs_batch, next_goal=next_goal_batch,
                                   future_obs=future_obs, future_goal=future_goal)

    def __len__(self):
        return self.step * self.num_envs
