"""Standalone 3×3 Patch MLP used after pySTEPS and exPreCast fields."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def millimetres_from_log_prediction(log_amount: torch.Tensor) -> torch.Tensor:
    """Convert the model's nonnegative log1p(RN60) output to millimetres."""

    if not torch.is_floating_point(log_amount):
        raise TypeError("log_amount must be floating point")
    return torch.expm1(torch.clamp_min(log_amount, 0.0))


class ChannelAttention(nn.Module):
    def __init__(self, width: int, reduction: int = 4) -> None:
        super().__init__()
        if width <= 0 or reduction <= 0:
            raise ValueError((width, reduction))
        hidden = max(4, width // reduction)
        self.gate = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.gate(inputs)


class PatchMLP(nn.Module):
    """Map six lead-aligned 3×3 field patches to one RN60 amount.

    The paper's context route uses ``aux_dim=39``: normalized x/y and lead
    (three entries), followed by a 36-entry summary derived exclusively from
    the 514 fitting-station gauge histories available by issuance time. No
    held-out target-site history is included.
    """

    def __init__(
        self,
        *,
        patch_steps: int = 6,
        patch_size: int = 3,
        aux_dim: int = 39,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_steps <= 0 or patch_size <= 0 or aux_dim < 0:
            raise ValueError((patch_steps, patch_size, aux_dim))
        if not 0.0 <= dropout < 1.0:
            raise ValueError(dropout)
        self.patch_steps = int(patch_steps)
        self.patch_size = int(patch_size)
        self.aux_dim = int(aux_dim)
        input_dim = self.patch_steps * self.patch_size**2 + self.aux_dim
        self.features = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        self.attention = ChannelAttention(128)
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self, patches: torch.Tensor, auxiliary: torch.Tensor | None = None
    ) -> torch.Tensor:
        expected_tail = (self.patch_steps, self.patch_size, self.patch_size)
        if patches.ndim < 4 or tuple(patches.shape[-3:]) != expected_tail:
            raise ValueError(
                f"patches must end in {expected_tail}, got {tuple(patches.shape)}"
            )
        if not torch.is_floating_point(patches):
            raise TypeError("patches must be floating point")
        leading = patches.shape[:-3]
        if self.aux_dim:
            expected_aux = leading + (self.aux_dim,)
            if auxiliary is None or tuple(auxiliary.shape) != expected_aux:
                shape = None if auxiliary is None else tuple(auxiliary.shape)
                raise ValueError(f"auxiliary must be {expected_aux}, got {shape}")
            if auxiliary.device != patches.device:
                raise ValueError("patches and auxiliary must share a device")
        elif auxiliary is not None and auxiliary.numel():
            raise ValueError("auxiliary must be absent when aux_dim=0")

        flattened = patches.reshape(-1, self.patch_steps * self.patch_size**2)
        if self.aux_dim:
            assert auxiliary is not None
            flattened = torch.cat(
                (flattened, auxiliary.reshape(-1, self.aux_dim)), dim=-1
            )
        latent = self.attention(self.features(flattened))
        raw = self.output(latent).squeeze(-1)
        log_amount = F.softplus(raw, beta=5.0)
        return log_amount.reshape(leading)
