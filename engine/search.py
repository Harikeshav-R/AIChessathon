"""PVS Search Engine with Numba JIT Core Acceleration.

Combines iterative deepening, aspiration windows, dynamic time management,
and zero-allocation JIT-compiled Alpha-Beta tree traversal.
"""

from __future__ import annotations

import contextlib
import time

import numpy as np

from engine.defs import (
    MAX_SEARCH_DEPTH,
    MOVE_NONE,
    PHASE_BOUND,
    PHASE_ENDGAME,
    PHASE_MIDDLEGAME,
    PHASE_SACRIFICE,
    SCORE_LOST,
    SCORE_TB_WIN,
    SCORE_WIN,
    SQUARE_NONE,
)
from engine.fast_search import (
    pvs_search_fast,
    total_mat_fast,
)
from engine.movegen import legal_moves
from engine.nnue import (
    NNUEWeights,
    feature_indices,
    nnue_refresh_side,
    phase_to_index,
)
from engine.params import (
    ASP_START_WINDOW,
)
from engine.position import (
    Position,
    move_to_uci,
    set_fen,
)
from engine.tm import TimeManager

TT_SIZE: int = 1 << 20  # 1M entries (~16 MB)


class EngineSearch:
    """Stateful Alpha-Beta Search Engine with Zero-Heap JIT Acceleration."""

    def __init__(self, weights: NNUEWeights) -> None:
        self.weights = weights
        self.tm = TimeManager()

        # Preallocated Flat Stack Buffers (Zero Allocation)
        self.board_stack = np.zeros((128, 64), dtype=np.uint8)
        self.colors_stack = np.zeros((128, 2), dtype=np.uint64)
        self.pieces_stack = np.zeros((128, 7), dtype=np.uint64)
        self.material_stack = np.zeros((128, 10), dtype=np.int32)
        self.castling_stack = np.full((128, 2, 2), SQUARE_NONE, dtype=np.uint8)
        self.ep_stack = np.full(128, SQUARE_NONE, dtype=np.uint8)
        self.color_stack = np.zeros(128, dtype=np.uint8)
        self.halfmoves_stack = np.zeros(128, dtype=np.int32)
        self.zobrist_stack = np.zeros(128, dtype=np.uint64)
        self.pawn_stack = np.zeros(128, dtype=np.uint64)
        self.non_pawn_stack = np.zeros((128, 2), dtype=np.uint64)
        self.phase_stack = np.zeros(128, dtype=np.int32)

        self.white_acc_stack = np.zeros((128, 1024), dtype=np.int16)
        self.black_acc_stack = np.zeros((128, 1024), dtype=np.int16)

        self.move_stack = np.zeros((128, 256), dtype=np.uint16)
        self.score_stack = np.zeros((128, 256), dtype=np.int32)

        # Search Tables
        self.pv_table = np.zeros((128, 128), dtype=np.uint16)
        self.pv_length = np.zeros(128, dtype=np.int32)
        self.killers = np.zeros((128, 2), dtype=np.uint16)
        self.main_history = np.zeros((2, 64, 64), dtype=np.int16)
        self.cap_history = np.zeros((14, 64), dtype=np.int16)
        self.cont_history_1 = np.zeros((14, 64), dtype=np.int16)
        self.pawn_corr_hist = np.zeros((2, 16384), dtype=np.int16)
        self.non_pawn_corr_hist = np.zeros((2, 2, 16384), dtype=np.int16)

        # Transposition Table
        self.tt_size = TT_SIZE
        self.tt_mask = TT_SIZE - 1
        self.tt_keys = np.zeros(TT_SIZE, dtype=np.uint16)
        self.tt_scores = np.zeros(TT_SIZE, dtype=np.int16)
        self.tt_static_evals = np.zeros(TT_SIZE, dtype=np.int16)
        self.tt_moves = np.zeros(TT_SIZE, dtype=np.uint16)
        self.tt_depths = np.zeros(TT_SIZE, dtype=np.uint8)
        self.tt_bounds = np.zeros(TT_SIZE, dtype=np.uint8)
        self.tt_ages = np.zeros(TT_SIZE, dtype=np.uint8)

        # Search Controls
        self.nodes_count = np.zeros(1, dtype=np.int64)
        self.stop_flag = np.zeros(1, dtype=np.int32)
        self.start_time: float = 0.0
        self.soft_time_limit: float = 0.0
        self.hard_time_limit: float = 0.0
        self.current_iter: int = 0
        self.age: int = 0
        self.phase: int = PHASE_MIDDLEGAME

    @property
    def nodes(self) -> int:
        return int(self.nodes_count[0])

    @nodes.setter
    def nodes(self, val: int) -> None:
        self.nodes_count[0] = val

    def setup_root_position(self, fen: str) -> Position:
        pos = Position()
        set_fen(pos, fen)

        for i in range(64):
            self.board_stack[0, i] = pos.board[i]
        for i in range(2):
            self.colors_stack[0, i] = pos.colors_bb[i]
        for i in range(7):
            self.pieces_stack[0, i] = pos.pieces_bb[i]
        for i in range(10):
            self.material_stack[0, i] = pos.material_count[i]
        for c in range(2):
            for s in range(2):
                self.castling_stack[0, c, s] = pos.castling_squares[c, s]
        self.ep_stack[0] = pos.ep_square
        self.color_stack[0] = pos.color
        self.halfmoves_stack[0] = pos.halfmoves
        self.zobrist_stack[0] = pos.zobrist_key
        self.pawn_stack[0] = pos.pawn_key
        self.non_pawn_stack[0, 0] = pos.non_pawn_key[0]
        self.non_pawn_stack[0, 1] = pos.non_pawn_key[1]
        self.phase = (
            PHASE_ENDGAME
            if total_mat_fast(self.material_stack, 0) < PHASE_BOUND
            else PHASE_MIDDLEGAME
        )
        self.phase_stack[0] = self.phase

        # Refresh NNUE accumulator at ply 0
        net_idx = phase_to_index(self.phase)
        f_w = self.weights.feature_weights[net_idx]
        f_b = self.weights.feature_biases[net_idx]

        white_indices = np.zeros(64, dtype=np.int32)
        black_indices = np.zeros(64, dtype=np.int32)
        count = 0
        for sq in range(64):
            piece = int(pos.board[sq])
            if piece != 0:
                w_idx, b_idx = feature_indices(piece, sq)
                white_indices[count] = w_idx
                black_indices[count] = b_idx
                count += 1

        nnue_refresh_side(self.white_acc_stack[0], f_b, f_w, white_indices, count)
        nnue_refresh_side(self.black_acc_stack[0], f_b, f_w, black_indices, count)

        return pos

    def search_root(self, fen: str, time_left_ms: int) -> tuple[str, str | None]:
        """Runs iterative deepening search using Numba JIT fast search kernel."""
        pos = self.setup_root_position(fen)

        self.start_time = time.perf_counter()
        soft_limit, hard_limit = self.tm.allocate_time(time_left_ms, inc_ms=100)
        self.soft_time_limit = float(soft_limit)
        self.hard_time_limit = float(hard_limit)
        self.stop_flag[0] = 0
        self.nodes_count[0] = 0
        self.pv_table.fill(0)
        self.pv_length.fill(0)
        self.age = (self.age + 1) & 63

        all_legal = legal_moves(pos)
        if not all_legal:
            return "0000", None
        if len(all_legal) == 1:
            return move_to_uci(all_legal[0]), None

        best_move = all_legal[0]
        prev_best_move = best_move
        stability = 0
        prev_score = 0
        predicted_reply: str | None = None

        # Iterative Deepening
        for depth in range(1, MAX_SEARCH_DEPTH):
            self.current_iter = depth

            # Hard node limit based on remaining time budget
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
            rem_ms = max(5.0, self.hard_time_limit - elapsed_ms)
            node_limit = int(self.nodes_count[0] + rem_ms * 280.0)

            # Aspiration Window
            if depth >= 5:
                delta = ASP_START_WINDOW
                alpha = max(SCORE_LOST, prev_score - delta)
                beta = min(SCORE_WIN, prev_score + delta)
                score = pvs_search_fast(
                    alpha,
                    beta,
                    depth,
                    0,
                    True,
                    False,
                    MOVE_NONE,
                    self.board_stack,
                    self.colors_stack,
                    self.pieces_stack,
                    self.material_stack,
                    self.castling_stack,
                    self.ep_stack,
                    self.color_stack,
                    self.halfmoves_stack,
                    self.zobrist_stack,
                    self.pawn_stack,
                    self.non_pawn_stack,
                    self.phase_stack,
                    self.white_acc_stack,
                    self.black_acc_stack,
                    self.move_stack,
                    self.score_stack,
                    self.pv_table,
                    self.pv_length,
                    self.killers,
                    self.main_history,
                    self.cap_history,
                    self.cont_history_1,
                    self.pawn_corr_hist,
                    self.non_pawn_corr_hist,
                    self.weights.feature_weights,
                    self.weights.output_weights,
                    self.weights.output_biases,
                    self.tt_keys,
                    self.tt_scores,
                    self.tt_static_evals,
                    self.tt_moves,
                    self.tt_depths,
                    self.tt_bounds,
                    self.tt_ages,
                    self.tt_mask,
                    self.age,
                    self.nodes_count,
                    self.stop_flag,
                    node_limit,
                )

                while score <= alpha or score >= beta:
                    delta += delta // 2
                    alpha = max(SCORE_LOST, prev_score - delta)
                    beta = min(SCORE_WIN, prev_score + delta)
                    score = pvs_search_fast(
                        alpha,
                        beta,
                        depth,
                        0,
                        True,
                        False,
                        MOVE_NONE,
                        self.board_stack,
                        self.colors_stack,
                        self.pieces_stack,
                        self.material_stack,
                        self.castling_stack,
                        self.ep_stack,
                        self.color_stack,
                        self.halfmoves_stack,
                        self.zobrist_stack,
                        self.pawn_stack,
                        self.non_pawn_stack,
                        self.phase_stack,
                        self.white_acc_stack,
                        self.black_acc_stack,
                        self.move_stack,
                        self.score_stack,
                        self.pv_table,
                        self.pv_length,
                        self.killers,
                        self.main_history,
                        self.cap_history,
                        self.cont_history_1,
                        self.pawn_corr_hist,
                        self.non_pawn_corr_hist,
                        self.weights.feature_weights,
                        self.weights.output_weights,
                        self.weights.output_biases,
                        self.tt_keys,
                        self.tt_scores,
                        self.tt_static_evals,
                        self.tt_moves,
                        self.tt_depths,
                        self.tt_bounds,
                        self.tt_ages,
                        self.tt_mask,
                        self.age,
                        self.nodes_count,
                        self.stop_flag,
                        node_limit,
                    )
                    if self.stop_flag[0] != 0:
                        break
            else:
                score = pvs_search_fast(
                    SCORE_LOST,
                    SCORE_WIN,
                    depth,
                    0,
                    True,
                    False,
                    MOVE_NONE,
                    self.board_stack,
                    self.colors_stack,
                    self.pieces_stack,
                    self.material_stack,
                    self.castling_stack,
                    self.ep_stack,
                    self.color_stack,
                    self.halfmoves_stack,
                    self.zobrist_stack,
                    self.pawn_stack,
                    self.non_pawn_stack,
                    self.phase_stack,
                    self.white_acc_stack,
                    self.black_acc_stack,
                    self.move_stack,
                    self.score_stack,
                    self.pv_table,
                    self.pv_length,
                    self.killers,
                    self.main_history,
                    self.cap_history,
                    self.cont_history_1,
                    self.pawn_corr_hist,
                    self.non_pawn_corr_hist,
                    self.weights.feature_weights,
                    self.weights.output_weights,
                    self.weights.output_biases,
                    self.tt_keys,
                    self.tt_scores,
                    self.tt_static_evals,
                    self.tt_moves,
                    self.tt_depths,
                    self.tt_bounds,
                    self.tt_ages,
                    self.tt_mask,
                    self.age,
                    self.nodes_count,
                    self.stop_flag,
                    node_limit,
                )

            if self.stop_flag[0] != 0:
                break

            if self.pv_length[0] > 0:
                root_m = int(self.pv_table[0, 0])
                if root_m in all_legal:
                    best_move = root_m

                if self.pv_length[0] > 1:
                    reply_m = int(self.pv_table[0, 1])
                    if reply_m != MOVE_NONE:
                        predicted_reply = move_to_uci(reply_m)

            stability = min(8, stability + 1) if best_move == prev_best_move else 0

            # 3-Net Phase Switch at Depth >= 6
            if depth >= 6 and total_mat_fast(self.material_stack, 0) >= PHASE_BOUND:
                if score < -20:
                    self.phase = PHASE_ENDGAME
                elif score > 400:
                    self.phase = PHASE_SACRIFICE
                else:
                    self.phase = PHASE_MIDDLEGAME
                self.phase_stack[0] = self.phase

            prev_best_move = best_move
            prev_score = score

            # Dynamic time check
            elapsed = (time.perf_counter() - self.start_time) * 1000.0
            if elapsed >= self.soft_time_limit * 0.58 or abs(score) >= SCORE_TB_WIN:
                break

        return move_to_uci(best_move), predicted_reply

    def warmup(self) -> None:
        """Warms up all Numba JIT search code paths during the 60s init window."""
        warmup_fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqk2r/pp2bppp/2n5/2p5/3pP3/5NP1/PPP2PBP/R1BQK2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/8/4k3/8/8/8/4K3/4Q3 w - - 0 1",
        ]
        old_soft, old_hard = self.soft_time_limit, self.hard_time_limit
        self.soft_time_limit = 50.0
        self.hard_time_limit = 100.0

        for fen in warmup_fens:
            with contextlib.suppress(Exception):
                self.search_root(fen, 1000)

        self.soft_time_limit, self.hard_time_limit = old_soft, old_hard
