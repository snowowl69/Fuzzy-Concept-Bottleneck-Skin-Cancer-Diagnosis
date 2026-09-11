"""Fuzzy Membership Functions.

Provides differentiable Gaussian, Sigmoidal, and Trapezoidal fuzzy membership functions
with trainable center, scale, and shape parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianMembership(nn.Module):
    """Differentiable Gaussian Fuzzy Membership Function:

    mu(x) = exp( -0.5 * ((x - c) / sigma)^2 )
    """

    def __init__(self, init_center: float = 0.5, init_sigma: float = 0.2):
        super().__init__()
        self.center = nn.Parameter(torch.tensor(init_center, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(init_sigma, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma_safe = torch.clamp(torch.abs(self.sigma), min=1e-4)
        diff = (x - self.center) / sigma_safe
        return torch.exp(-0.5 * diff**2)


class SigmoidMembership(nn.Module):
    """Differentiable Sigmoidal Fuzzy Membership Function:

    mu(x) = 1 / (1 + exp( -a * (x - c) ))
    """

    def __init__(self, init_slope: float = 10.0, init_center: float = 0.5):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(init_slope, dtype=torch.float32))
        self.center = nn.Parameter(torch.tensor(init_center, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.slope * (x - self.center))


class LinearFuzzyMapping(nn.Module):
    """Linear fuzzy normalization mapping with trainable slope and intercept clamped to [0, 1]."""

    def __init__(self, init_scale: float = 1.0, init_bias: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.scale * x + self.bias, min=0.0, max=1.0)


class TrapezoidalMembership(nn.Module):
    """Differentiable Trapezoidal Fuzzy Membership Function.

    Defined by four learnable parameters a < b <= c < d:
        mu(x) = 0                      if x <= a or x >= d
        mu(x) = (x - a) / (b - a)      if a < x < b
        mu(x) = 1                       if b <= x <= c
        mu(x) = (d - x) / (d - c)      if c < x < d

    Uses soft approximations for differentiability.
    """

    def __init__(
        self,
        init_a: float = 0.1,
        init_b: float = 0.3,
        init_c: float = 0.6,
        init_d: float = 0.9,
    ):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(init_c, dtype=torch.float32))
        self.d = nn.Parameter(torch.tensor(init_d, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        # Ensure ordering: a < b <= c < d with soft constraints
        a = self.a
        b = a + torch.abs(self.b - self.a) + eps
        c = b + torch.abs(self.c - self.b)
        d = c + torch.abs(self.d - self.c) + eps

        # Rising edge: (x - a) / (b - a)
        rise = torch.clamp((x - a) / (b - a + eps), 0.0, 1.0)
        # Falling edge: (d - x) / (d - c)
        fall = torch.clamp((d - x) / (d - c + eps), 0.0, 1.0)
        # Membership is min of rise and fall
        return torch.min(rise, fall)

