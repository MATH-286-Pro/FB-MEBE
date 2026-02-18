"""
Normalizing Flow for goal distribution estimation and inverse sampling exploration.

This module implements a normalizing flow to estimate the distribution of goals
and perform p(x)^(-beta) inverse sampling for Z-bias exploration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MaskedCouplingLayer(nn.Module):
    """Coupling layer with masking for normalizing flows (RealNVP-style)."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        mask_type: str = "checkerboard",  # "checkerboard" or "channel"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mask_type = mask_type
        
        # Create mask
        if mask_type == "checkerboard":
            mask_tensor = (torch.arange(input_dim) % 2).float()
            self.register_buffer("mask", mask_tensor)
        elif mask_type == "channel":
            mask_tensor = torch.zeros(input_dim)
            mask_tensor[: input_dim // 2] = 1
            self.register_buffer("mask", mask_tensor)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        
        # Type annotation for mask
        self.mask: torch.Tensor
        
        # Scale and translation networks
        # s(x) for scaling, t(x) for translation
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),  # Bound the scale to avoid numerical instability
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (data -> latent)."""
        # x: (batch, dim)
        masked_x = x * self.mask
        
        # Compute scale and translation
        s = self.scale_net(masked_x)
        t = self.translate_net(masked_x)
        
        # Apply affine transformation to unmasked dimensions
        # y = x * exp(s) + t for unmasked dims, y = x for masked dims
        y = masked_x + (1 - self.mask) * (x * torch.exp(s) + t)
        
        # Compute log determinant of Jacobian
        log_det = ((1 - self.mask) * s).sum(dim=-1)
        
        return y, log_det
    
    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass (latent -> data)."""
        masked_y = y * self.mask
        
        # Compute scale and translation using masked input
        s = self.scale_net(masked_y)
        t = self.translate_net(masked_y)
        
        # Invert the transformation
        x = masked_y + (1 - self.mask) * ((y - t) * torch.exp(-s))
        
        # Log determinant for inverse
        log_det = -((1 - self.mask) * s).sum(dim=-1)
        
        return x, log_det


class NormalizingFlow(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 6,
        hidden_dim: int = 256,
        device: str = "cuda",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.device = device
        
        # Build coupling layers with alternating masks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            mask_type = "checkerboard" if i % 2 == 0 else "channel"
            self.layers.append(
                MaskedCouplingLayer(input_dim, hidden_dim, mask_type)
            )
        
        # Base distribution (standard Gaussian)
        self.register_buffer("base_mean", torch.zeros(input_dim))
        self.register_buffer("base_std", torch.ones(input_dim))
        
        # Type annotations
        self.base_mean: torch.Tensor
        self.base_std: torch.Tensor
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_sum = 0
        z = x
        
        # Apply coupling layers
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_sum = log_det_sum + log_det
        
        # Compute log probability under base distribution
        log_prob_base = -0.5 * (
            self.input_dim * np.log(2 * np.pi)
            + ((z - self.base_mean) ** 2 / self.base_std ** 2).sum(dim=-1)
        )
        
        # Total log probability
        log_prob = log_prob_base + log_det_sum
        
        return z, log_prob
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        
        # Apply coupling layers in reverse order
        for layer in reversed(self.layers):
            x, _ = layer.inverse(x)
        
        return x
    
    def sample(self, num_samples: int) -> torch.Tensor:
        # Sample from base distribution (standard Gaussian)
        z = torch.randn(num_samples, self.input_dim, device=self.device)
        
        # Transform to data space
        samples = self.inverse(z)
        
        return samples
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        _, log_prob = self.forward(x)
        return log_prob
    
    def fit(
        self,
        data: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        print: bool = False,
    ) -> list:
        """
        Fit the normalizing flow to data.
        
        Args:
            data: Training data (N, input_dim)
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            
        Returns:
            losses: List of training losses
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        
        num_batches = (len(data) + batch_size - 1) // batch_size
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            perm = torch.randperm(len(data))
            data_shuffled = data[perm]
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(data))
                batch = data_shuffled[start_idx:end_idx]
                
                # Forward pass
                _, log_prob = self.forward(batch)
                
                # Negative log-likelihood loss
                loss = -log_prob.mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if print and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return losses


class GoalDensityEstimator:
    def __init__(
        self,
        goal_dim: int,
        buffer_size: int = 10000,
        num_layers: int = 6,
        hidden_dim: int = 256,
        beta: float = 1.0,
        device: str = "cuda",
    ):
        self.goal_dim = goal_dim
        self.beta = beta
        self.device = device
        
        # Create normalizing flow
        self.flow = NormalizingFlow(
            input_dim=goal_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            device=device,
        )
        
        # Statistics for tracking
        self.num_updates = 0
        self.goal_buffer = []
        self.max_buffer_size = buffer_size
    
    def update(self, goals: torch.Tensor) -> None:
        """
        Add new goals to the buffer.
        
        Args:
            goals: New goal observations (batch, goal_dim)
        """
        if isinstance(goals, np.ndarray):
            goals = torch.from_numpy(goals).float().to(self.device)
        
        # Add to buffer
        self.goal_buffer.append(goals.detach().cpu())
        
        # Trim buffer if too large
        if len(self.goal_buffer) > 1:
            all_goals = torch.cat(self.goal_buffer, dim=0)
            if len(all_goals) > self.max_buffer_size:
                # Keep most recent goals
                all_goals = all_goals[-self.max_buffer_size:]
                self.goal_buffer = [all_goals]
    
    def fit(
        self,
        num_epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        min_samples: int = 100,
    ) -> Optional[list]:
        """
        Fit the normalizing flow to buffered goal data.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            min_samples: Minimum number of samples required to fit
            
        Returns:
            losses: Training losses, or None if not enough data
        """
        if len(self.goal_buffer) == 0:
            return None
        
        # Concatenate buffer
        goals = torch.cat(self.goal_buffer, dim=0)
        
        if len(goals) < min_samples:
            print(f"Not enough samples ({len(goals)} < {min_samples}), skipping fit")
            return None
        
        # Move to device
        goals = goals.to(self.device)
        
        # Fit flow
        print(f"Fitting flow to {len(goals)} goal samples...")
        losses = self.flow.fit(goals, num_epochs, batch_size, lr)
        
        self.num_updates += 1
        
        return losses
    
    def sample_inverse(
        self,
        num_samples: int,
        num_candidates: int = 1000,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        # Generate candidate samples
        with torch.no_grad():
            candidates = self.flow.sample(num_candidates)
            
            # Compute log probabilities
            log_probs = self.flow.log_prob(candidates)
            
            # Compute inverse weights: p(x)^(-beta)
            # In log space: -beta * log p(x)
            log_weights = -self.beta * log_probs
            
            # Apply temperature
            log_weights = log_weights / temperature
            
            # Convert to probabilities (softmax)
            weights = F.softmax(log_weights, dim=0)
            
            # Sample indices based on weights
            indices = torch.multinomial(weights, num_samples, replacement=False)
            
            # Return selected samples
            samples = candidates[indices]
        
        return samples
    
    def sample_random(self, num_samples: int) -> torch.Tensor:
        with torch.no_grad():
            samples = self.flow.sample(num_samples)
        return samples
    
    def log_prob(self, goals: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            log_prob = self.flow.log_prob(goals)
        return log_prob