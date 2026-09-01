"""3-Net NNUE Incremental Accumulator & Forward Evaluation.

Implements the HalfKA perspective piece-square architecture (768 -> 1024 -> SCReLU -> 1)
with Numba JIT acceleration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

# Network Constants
INPUT_SIZE: int = 768
LAYER1_SIZE: int = 1024
QA: int = 255
QB: int = 64
QAB: int = QA * QB
SCALE: int = 400
NUM_NETS: int = 3
MAX_STACK: int = 128


@njit(fastmath=True, nogil=True)
def feature_indices(piece: int, sq: int) -> tuple[int, int]:
    """Returns (white_feature_idx, black_feature_idx) for a piece on square."""
    color_stride = 64 * 6
    piece_stride = 64
    base = (piece >> 1) - 1
    color = piece & 1

    white_idx = color * color_stride + base * piece_stride + sq
    black_idx = (color ^ 1) * color_stride + base * piece_stride + (sq ^ 56)
    return white_idx, black_idx


@njit(fastmath=True, nogil=True)
def phase_to_index(phase: int) -> int:
    """Maps engine phase (Middlegame=0, Endgame=1, Sacrifice=2) to network index."""
    if phase == 0:
        return 0
    elif phase == 1:
        return 1
    return 2


@njit(fastmath=True, nogil=True)
def nnue_refresh_side(
        dst: np.ndarray,
        bias: np.ndarray,
        feature_v: np.ndarray,
        indices: np.ndarray,
        count: int,
) -> None:
    """Full refresh of a 1024-int16 accumulator from board features."""
    for i in range(1024):
        val = np.int32(bias[i])
        for p in range(count):
            idx = indices[p]
            val += np.int32(feature_v[idx, i])
        if val > np.int32(32767):
            val = np.int32(32767)
        elif val < np.int32(-32768):
            val = np.int32(-32768)
        dst[i] = np.int16(val)


@njit(fastmath=True, nogil=True)
def nnue_apply_update(
        dst: np.ndarray,
        src: np.ndarray,
        feature_v: np.ndarray,
        add_indices: np.ndarray,
        num_adds: int,
        sub_indices: np.ndarray,
        num_subs: int,
) -> None:
    """Incremental accumulator update (adds and subtractions)."""
    for i in range(1024):
        val = np.int32(src[i])
        for a in range(num_adds):
            val += np.int32(feature_v[add_indices[a], i])
        for s in range(num_subs):
            val -= np.int32(feature_v[sub_indices[s], i])
        if val > np.int32(32767):
            val = np.int32(32767)
        elif val < np.int32(-32768):
            val = np.int32(-32768)
        dst[i] = np.int16(val)


@njit(fastmath=True, nogil=True)
def screlu_flatten(
        us_acc: np.ndarray,
        them_acc: np.ndarray,
        out_weights: np.ndarray,
        out_bias: np.int16,
) -> int:
    """Evaluates SCReLU dot product against output weights and applies scale factor."""
    acc = np.int64(0)
    for i in range(1024):
        # Clipped Square ReLU on us
        u = np.int32(us_acc[i])
        if u < np.int32(0):
            u = np.int32(0)
        elif u > np.int32(255):
            u = np.int32(255)
        acc += np.int64(u * u) * np.int64(out_weights[i])

        # Clipped Square ReLU on them
        t = np.int32(them_acc[i])
        if t < np.int32(0):
            t = np.int32(0)
        elif t > np.int32(255):
            t = np.int32(255)
        acc += np.int64(t * t) * np.int64(out_weights[1024 + i])

    biased = acc + np.int64(out_bias) * QA
    return int((biased * SCALE) // (QA * QAB))


class NNUEWeights:
    """Container holding 3 quantized int16 networks in contiguous memory."""

    def __init__(
            self,
            feature_weights: np.ndarray,
            feature_biases: np.ndarray,
            output_weights: np.ndarray,
            output_biases: np.ndarray,
    ) -> None:
        self.feature_weights = feature_weights  # shape (3, 768, 1024) int16
        self.feature_biases = feature_biases  # shape (3, 1024) int16
        self.output_weights = output_weights  # shape (3, 2048) int16
        self.output_biases = output_biases  # shape (3,) int16


class NNUEState:
    """Manages the incremental accumulator stack across search plies."""

    def __init__(self, weights: NNUEWeights) -> None:
        self.weights = weights
        self.white_stack = np.zeros((MAX_STACK, LAYER1_SIZE), dtype=np.int16)
        self.black_stack = np.zeros((MAX_STACK, LAYER1_SIZE), dtype=np.int16)
        self.curr_index: int = 0
        self.cached_eval: int = 0
        self.cached_color: int = -1
        self.cached_net: int = -1
        self.is_cached: bool = False

    def push(self) -> None:
        self.curr_index += 1
        self.is_cached = False

    def pop(self) -> None:
        if self.curr_index > 0:
            self.curr_index -= 1
        self.is_cached = False

    def reset_nnue(self, board: np.ndarray, phase: int) -> None:
        """Full accumulator reset from current board state."""
        self.curr_index = 0
        net_idx = phase_to_index(phase)
        f_w = self.weights.feature_weights[net_idx]
        f_b = self.weights.feature_biases[net_idx]

        white_indices = np.zeros(64, dtype=np.int32)
        black_indices = np.zeros(64, dtype=np.int32)
        count = 0

        for sq in range(64):
            piece = int(board[sq])
            if piece != 0:
                w_idx, b_idx = feature_indices(piece, sq)
                white_indices[count] = w_idx
                black_indices[count] = b_idx
                count += 1

        nnue_refresh_side(self.white_stack[0], f_b, f_w, white_indices, count)
        nnue_refresh_side(self.black_stack[0], f_b, f_w, black_indices, count)
        self.is_cached = False

    def evaluate(self, color: int, phase: int) -> int:
        """Evaluates position for given side to move and phase."""
        net_idx = phase_to_index(phase)
        if self.is_cached and self.cached_color == color and self.cached_net == net_idx:
            return self.cached_eval

        idx = self.curr_index
        if color == 0:  # White
            us = self.white_stack[idx]
            them = self.black_stack[idx]
        else:  # Black
            us = self.black_stack[idx]
            them = self.white_stack[idx]

        score = screlu_flatten(
            us,
            them,
            self.weights.output_weights[net_idx],
            self.weights.output_biases[net_idx],
        )

        self.cached_eval = score
        self.cached_color = color
        self.cached_net = net_idx
        self.is_cached = True
        return score

    def update_1add_1sub(
            self,
            from_piece: int,
            from_sq: int,
            to_piece: int,
            to_sq: int,
            phase: int,
    ) -> None:
        net_idx = phase_to_index(phase)
        f_w = self.weights.feature_weights[net_idx]

        w_from, b_from = feature_indices(from_piece, from_sq)
        w_to, b_to = feature_indices(to_piece, to_sq)

        add_w = np.array([w_to], dtype=np.int32)
        sub_w = np.array([w_from], dtype=np.int32)
        add_b = np.array([b_to], dtype=np.int32)
        sub_b = np.array([b_from], dtype=np.int32)

        prev = self.curr_index
        self.push()
        curr = self.curr_index

        nnue_apply_update(self.white_stack[curr], self.white_stack[prev], f_w, add_w, 1, sub_w, 1)
        nnue_apply_update(self.black_stack[curr], self.black_stack[prev], f_w, add_b, 1, sub_b, 1)

    def update_1add_2sub(
            self,
            from_piece: int,
            from_sq: int,
            to_piece: int,
            to_sq: int,
            captured_piece: int,
            captured_sq: int,
            phase: int,
    ) -> None:
        net_idx = phase_to_index(phase)
        f_w = self.weights.feature_weights[net_idx]

        w_from, b_from = feature_indices(from_piece, from_sq)
        w_to, b_to = feature_indices(to_piece, to_sq)
        w_cap, b_cap = feature_indices(captured_piece, captured_sq)

        add_w = np.array([w_to], dtype=np.int32)
        sub_w = np.array([w_from, w_cap], dtype=np.int32)
        add_b = np.array([b_to], dtype=np.int32)
        sub_b = np.array([b_from, b_cap], dtype=np.int32)

        prev = self.curr_index
        self.push()
        curr = self.curr_index

        nnue_apply_update(self.white_stack[curr], self.white_stack[prev], f_w, add_w, 1, sub_w, 2)
        nnue_apply_update(self.black_stack[curr], self.black_stack[prev], f_w, add_b, 1, sub_b, 2)

    def update_2add_2sub(
            self,
            piece1: int,
            from1: int,
            to1: int,
            piece2: int,
            from2: int,
            to2: int,
            phase: int,
    ) -> None:
        net_idx = phase_to_index(phase)
        f_w = self.weights.feature_weights[net_idx]

        wf1, bf1 = feature_indices(piece1, from1)
        wt1, bt1 = feature_indices(piece1, to1)
        wf2, bf2 = feature_indices(piece2, from2)
        wt2, bt2 = feature_indices(piece2, to2)

        add_w = np.array([wt1, wt2], dtype=np.int32)
        sub_w = np.array([wf1, wf2], dtype=np.int32)
        add_b = np.array([bt1, bt2], dtype=np.int32)
        sub_b = np.array([bf1, bf2], dtype=np.int32)

        prev = self.curr_index
        self.push()
        curr = self.curr_index

        nnue_apply_update(self.white_stack[curr], self.white_stack[prev], f_w, add_w, 2, sub_w, 2)
        nnue_apply_update(self.black_stack[curr], self.black_stack[prev], f_w, add_b, 2, sub_b, 2)


def load_weights(path: Path) -> NNUEWeights:
    """Loads 3-net weights from .npz archive or generates synthetic baseline weights."""
    if path.exists():
        data = np.load(path)
        return NNUEWeights(
            feature_weights=data["feature_weights"].astype(np.int16),
            feature_biases=data["feature_biases"].astype(np.int16),
            output_weights=data["output_weights"].astype(np.int16),
            output_biases=data["output_biases"].astype(np.int16),
        )
    # Default fallback: zeros
    return NNUEWeights(
        feature_weights=np.zeros((3, 768, 1024), dtype=np.int16),
        feature_biases=np.zeros((3, 1024), dtype=np.int16),
        output_weights=np.zeros((3, 2048), dtype=np.int16),
        output_biases=np.zeros(3, dtype=np.int16),
    )
