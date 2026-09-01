"""PyTorch Dataset & Data Loader for NNUE Training.

Parses chess positions (FEN), extracts HalfKA sparse feature indices,
and prepares target values (centipawn eval + WDL).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from engine.nnue import feature_indices


def fen_to_features(fen: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Extracts active feature indices from FEN."""
    tokens = fen.strip().split()
    board_part = tokens[0]
    stm = 0.0 if len(tokens) > 1 and tokens[1] == "w" else 1.0

    piece_map = {
        "P": 2, "N": 4, "B": 6, "R": 8, "Q": 10, "K": 12,
        "p": 3, "n": 5, "b": 7, "r": 9, "q": 11, "k": 13,
    }

    white_indices = []
    black_indices = []

    sq = 56
    for char in board_part:
        if char == "/":
            sq -= 16
        elif char.isdigit():
            sq += int(char)
        elif char in piece_map:
            piece = piece_map[char]
            w_idx, b_idx = feature_indices(piece, sq)
            white_indices.append(w_idx)
            black_indices.append(b_idx)
            sq += 1

    return (
        np.array(white_indices, dtype=np.int64),
        np.array(black_indices, dtype=np.int64),
        stm,
    )


class NNUEDataset(Dataset):
    """Position evaluation dataset for HalfKA NNUE training."""

    def __init__(self, fens: list[str], evals_cp: list[int]) -> None:
        self.fens = fens
        self.evals_cp = evals_cp

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        fen = self.fens[idx]
        eval_cp = float(self.evals_cp[idx])
        w_idx, b_idx, stm = fen_to_features(fen)

        # Sigmoid WDL target
        wdl = 1.0 / (1.0 + math.exp(-eval_cp / 400.0))

        return {
            "white_indices": w_idx,
            "black_indices": b_idx,
            "stm": stm,
            "eval_cp": eval_cp,
            "wdl": wdl,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collates variable-length sparse feature lists into dense batch tensors."""
    batch_size = len(batch)
    w_dense = torch.zeros(batch_size, 768, dtype=torch.float32)
    b_dense = torch.zeros(batch_size, 768, dtype=torch.float32)
    stm_tensor = torch.zeros(batch_size, dtype=torch.float32)
    eval_tensor = torch.zeros(batch_size, 1, dtype=torch.float32)
    wdl_tensor = torch.zeros(batch_size, 1, dtype=torch.float32)

    for i, item in enumerate(batch):
        w_dense[i, item["white_indices"]] = 1.0
        b_dense[i, item["black_indices"]] = 1.0
        stm_tensor[i] = item["stm"]
        eval_tensor[i, 0] = item["eval_cp"]
        wdl_tensor[i, 0] = item["wdl"]

    return {
        "white_features": w_dense,
        "black_features": b_dense,
        "stm": stm_tensor,
        "eval_cp": eval_tensor,
        "wdl": wdl_tensor,
    }
