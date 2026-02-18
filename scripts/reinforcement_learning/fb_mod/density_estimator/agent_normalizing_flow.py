import torch
from .model_normalizing_flow import GoalDensityEstimator
from dataclasses import dataclass

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from toolbox.dataclass_pylance import AGENT_CFG, MODEL_CFG, TRAIN_CFG

class BUFFER:
    def __init__(self, max_size: int, goal_dim: int, device: str):
        self.max_size = max_size
        self.goal_dim = goal_dim
        self.device = device
        self.buffer = torch.empty((0, goal_dim), device=device)

    def add(self, goals: torch.Tensor):
        self.buffer = torch.cat([self.buffer, goals], dim=0)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
    @property
    def size(self):
        return len(self.buffer)

class NF_AGENT:
    def __init__(self, 
                 cfg: AGENT_CFG,
                 goal_indices: list[int], 
                 update_freq: int = 1000,
                 sample_buffer_size: int = 100000,
                 train_buffer_size:  int = 10000,
                 goal_range=None,
                 device="cpu"):
        
        self.cfg = cfg

        self.goal_indices = goal_indices
        self.goal_range_extended = torch.tensor(goal_range, device=device) if goal_range is not None else None
        self.beta = cfg.train.beta
        self.device = device
        self.estimator = GoalDensityEstimator(
            goal_dim = len(goal_indices),   # dimension for sliced goal
            num_layers=10,
            hidden_dim=64,
            buffer_size=train_buffer_size,  # flow's training buffer size < sample buffer size
            device = device,
        )

        self.buffer = BUFFER(max_size=int(sample_buffer_size), 
                                           goal_dim=cfg.model.goal_dim, 
                                           device=device)

        self.update_freq = update_freq
        self.step_count = 0

    # training
    def observe(self, goal_full: torch.Tensor):

        if self.goal_range_extended is not None:
            goals_to_check = goal_full[:, self.goal_indices]

            in_range_min = (goals_to_check >= self.goal_range_extended[:, 0]).all(dim=1)
            in_range_max = (goals_to_check <= self.goal_range_extended[:, 1]).all(dim=1)
            in_range_mask = in_range_min & in_range_max

            goal_full = goal_full[in_range_mask]

        # add data reverse sampling buffer (not for training)
        self.buffer.add(goal_full)

        # add data to flow training buffer (for training)
        goal_slice = goal_full[:, self.goal_indices]
        self.estimator.update(goal_slice)
    
        # train Flow
        if self.step_count % self.update_freq == 0 and self.buffer.size > 1000:
            self.step_count = 0
            self.estimator.fit(num_epochs=30, batch_size=256, lr=1e-3)
        self.step_count += 1


    # sample from buffer
    # return full dimension goal
    def inverse_sample_from_buffer(self, num_samples: int)-> torch.Tensor:
        prob = torch.exp(self.estimator.log_prob(self.buffer.buffer[:, self.goal_indices])) + self.cfg.train.eps
        prob_inv = prob ** (-self.beta)

        # IMPORTANT TODO:
        #   After debuging, we found that the inconsistence under same seed is caused by "torch.multinomial"
        #   "torch.multinomial" is non-deterministic using "cuda" but deterministic when using "cpu"
        #   discussion at: https://github.com/pytorch/pytorch/issues/154031

        indices = torch.multinomial(prob_inv.cpu(), num_samples, replacement=True).to(prob_inv.device)
        # indices = torch.multinomial(prob_inv, num_samples, replacement=True) # this will lead to inconsistence

        return self.buffer.buffer[indices]