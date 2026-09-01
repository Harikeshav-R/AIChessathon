"""PyTorch HalfKA NNUE Architecture.

768 -> 1024 -> SCReLU -> 1 architecture with dual perspective feature accumulator.
"""

from __future__ import annotations

import torch
import torch.nn as nn

INPUT_SIZE: int = 768
LAYER1_SIZE: int = 1024
SCALE: float = 400.0


class HalfKANNUE(nn.Module):
    """HalfKA Perspective Network architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_transform = nn.Linear(INPUT_SIZE, LAYER1_SIZE)
        self.output_layer = nn.Linear(LAYER1_SIZE * 2, 1)

    def forward(
            self,
            white_features: torch.Tensor,
            black_features: torch.Tensor,
            stm: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        white_features: (B, 768) float sparse/dense
        black_features: (B, 768) float sparse/dense
        stm: (B, 1) float: 0.0 for white, 1.0 for black
        """
        w_acc = self.feature_transform(white_features)
        b_acc = self.feature_transform(black_features)

        # Perspective selection
        stm_expanded = stm.unsqueeze(1) if stm.dim() == 1 else stm
        us = torch.where(stm_expanded == 0.0, w_acc, b_acc)
        them = torch.where(stm_expanded == 0.0, b_acc, w_acc)

        # Clipped Squared ReLU: clamp(x, 0, 1)^2
        us_act = torch.clamp(us, 0.0, 1.0).pow(2)
        them_act = torch.clamp(them, 0.0, 1.0).pow(2)

        combined = torch.cat([us_act, them_act], dim=-1)
        out = self.output_layer(combined)
        return out * SCALE
