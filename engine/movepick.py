"""Move Ordering & Move Picker Pipeline.

Implements multi-stage move picking: TT move, Good Captures (MVV-LVA + SEE + CapHist),
Killer moves, Countermoves, 1-ply & 2-ply Continuation History (ContHist), Bad Captures,
and Quiets sorted by Butterfly History.
"""

from __future__ import annotations

import numpy as np

from engine.defs import (
    MOVE_CASTLING,
    MOVE_EN_PASSANT,
    MOVE_NONE,
    MOVE_PROMOTION,
    PROMO_QUEEN,
    PT_PAWN,
    SEE_VALUES,
    extract_from,
    extract_promo,
    extract_to,
    extract_type,
    get_piece_type,
)
from engine.movegen import (
    GEN_CAPTURES,
    GEN_QUIETS,
    generate_moves,
)
from engine.position import Position

STAGE_TT: int = 0
STAGE_GEN_CAPTURES: int = 1
STAGE_CAPTURES: int = 2
STAGE_GEN_QUIETS: int = 3
STAGE_QUIETS: int = 4
STAGE_BAD_CAPTURES: int = 5
STAGE_DONE: int = 6

QUEEN_PROMO_SCORE: int = 200000000
GOOD_CAPTURE_BASE: int = 100000000
KILLER_SCORE: int = 90000000
COUNTER_SCORE: int = 80000000


def see(pos: Position, move: int, threshold: int = 0) -> bool:
    """Static Exchange Evaluation (SEE) threshold test."""
    from_sq = extract_from(move)
    to_sq = extract_to(move)
    m_type = extract_type(move)

    from_piece = int(pos.board[from_sq])
    to_piece = int(pos.board[to_sq])

    if m_type == MOVE_CASTLING:
        return threshold <= 0

    val = 0
    if to_piece != 0:
        val = int(SEE_VALUES[get_piece_type(to_piece)])
    elif m_type == MOVE_EN_PASSANT:
        val = int(SEE_VALUES[PT_PAWN])

    if m_type == MOVE_PROMOTION:
        val += int(SEE_VALUES[extract_promo(move) + 2]) - int(SEE_VALUES[PT_PAWN])

    if val < threshold:
        return False

    val -= int(SEE_VALUES[get_piece_type(from_piece)])
    return bool(val >= threshold)


class MovePicker:
    """Stateful move iterator yielding sorted pseudo-legal moves on demand."""

    def __init__(
            self,
            pos: Position,
            tt_move: int = MOVE_NONE,
            threshold: int = 0,
            killer_1: int = MOVE_NONE,
            killer_2: int = MOVE_NONE,
            counter_move: int = MOVE_NONE,
            cap_hist: np.ndarray | None = None,
            main_hist: np.ndarray | None = None,
            cont_hist_1: np.ndarray | None = None,
            cont_hist_2: np.ndarray | None = None,
    ) -> None:
        self.pos = pos
        self.tt_move = tt_move
        self.threshold = threshold
        self.killer_1 = killer_1
        self.killer_2 = killer_2
        self.counter_move = counter_move
        self.cap_hist = cap_hist
        self.main_hist = main_hist
        self.cont_hist_1 = cont_hist_1
        self.cont_hist_2 = cont_hist_2

        self.stage: int = STAGE_TT
        self.captures = np.zeros(128, dtype=np.uint16)
        self.cap_scores = np.zeros(128, dtype=np.int32)
        self.num_captures: int = 0
        self.cap_idx: int = 0

        self.quiets = np.zeros(128, dtype=np.uint16)
        self.quiet_scores = np.zeros(128, dtype=np.int32)
        self.num_quiets: int = 0
        self.quiet_idx: int = 0

        self.bad_captures = np.zeros(64, dtype=np.uint16)
        self.num_bad_captures: int = 0
        self.bad_cap_idx: int = 0

    def next_move(self, skip_quiets: bool = False) -> int:
        """Yields next best move according to search priority."""
        while self.stage != STAGE_DONE:
            # 1. TT Move
            if self.stage == STAGE_TT:
                self.stage = STAGE_GEN_CAPTURES
                if self.tt_move != MOVE_NONE:
                    return self.tt_move

            # 2. Generate Captures
            elif self.stage == STAGE_GEN_CAPTURES:
                self.num_captures = generate_moves(self.pos, self.captures, GEN_CAPTURES)
                # Score captures
                for i in range(self.num_captures):
                    m = int(self.captures[i])
                    f = extract_from(m)
                    t = extract_to(m)
                    f_piece = int(self.pos.board[f])
                    t_piece = int(self.pos.board[t])
                    att_val = int(SEE_VALUES[get_piece_type(f_piece)])
                    vic_val = (
                        int(SEE_VALUES[get_piece_type(t_piece)])
                        if t_piece != 0
                        else att_val
                    )
                    c_hist = (
                        int(self.cap_hist[f_piece, t])
                        if self.cap_hist is not None
                        else 0
                    )

                    if extract_type(m) == MOVE_PROMOTION and extract_promo(m) == PROMO_QUEEN:
                        self.cap_scores[i] = (
                                QUEEN_PROMO_SCORE + vic_val * 100 - att_val + c_hist
                        )
                    else:
                        self.cap_scores[i] = (
                                GOOD_CAPTURE_BASE + vic_val * 100 - att_val + c_hist
                        )

                self.stage = STAGE_CAPTURES
                self.cap_idx = 0

            # 3. Pick Good Captures
            elif self.stage == STAGE_CAPTURES:
                while self.cap_idx < self.num_captures:
                    # Find highest score move
                    best_i = self.cap_idx
                    for i in range(self.cap_idx + 1, self.num_captures):
                        if self.cap_scores[i] > self.cap_scores[best_i]:
                            best_i = i

                    # Swap
                    self.captures[self.cap_idx], self.captures[best_i] = (
                        self.captures[best_i],
                        self.captures[self.cap_idx],
                    )
                    self.cap_scores[self.cap_idx], self.cap_scores[best_i] = (
                        self.cap_scores[best_i],
                        self.cap_scores[self.cap_idx],
                    )

                    m = int(self.captures[self.cap_idx])
                    self.cap_idx += 1

                    if m == self.tt_move:
                        continue

                    # SEE threshold check
                    if see(self.pos, m, self.threshold):
                        return m
                    else:
                        self.bad_captures[self.num_bad_captures] = m
                        self.num_bad_captures += 1

                self.stage = STAGE_GEN_QUIETS

            # 4. Generate Quiets
            elif self.stage == STAGE_GEN_QUIETS:
                if skip_quiets:
                    self.stage = STAGE_BAD_CAPTURES
                    continue

                self.num_quiets = generate_moves(self.pos, self.quiets, GEN_QUIETS)
                color = self.pos.color

                for i in range(self.num_quiets):
                    m = int(self.quiets[i])
                    if m == self.killer_1:
                        self.quiet_scores[i] = KILLER_SCORE
                    elif m == self.killer_2:
                        self.quiet_scores[i] = KILLER_SCORE - 1000
                    elif m == self.counter_move:
                        self.quiet_scores[i] = COUNTER_SCORE
                    else:
                        f = extract_from(m)
                        t = extract_to(m)
                        f_piece = int(self.pos.board[f])
                        score = (
                            int(self.main_hist[color, f, t])
                            if self.main_hist is not None
                            else 0
                        )
                        if self.cont_hist_1 is not None:
                            score += int(self.cont_hist_1[f_piece, t])
                        if self.cont_hist_2 is not None:
                            score += int(self.cont_hist_2[f_piece, t])
                        self.quiet_scores[i] = score

                self.stage = STAGE_QUIETS
                self.quiet_idx = 0

            # 5. Pick Quiets
            elif self.stage == STAGE_QUIETS:
                while self.quiet_idx < self.num_quiets:
                    best_i = self.quiet_idx
                    for i in range(self.quiet_idx + 1, self.num_quiets):
                        if self.quiet_scores[i] > self.quiet_scores[best_i]:
                            best_i = i

                    self.quiets[self.quiet_idx], self.quiets[best_i] = (
                        self.quiets[best_i],
                        self.quiets[self.quiet_idx],
                    )
                    self.quiet_scores[self.quiet_idx], self.quiet_scores[best_i] = (
                        self.quiet_scores[best_i],
                        self.quiet_scores[self.quiet_idx],
                    )

                    m = int(self.quiets[self.quiet_idx])
                    self.quiet_idx += 1

                    if m == self.tt_move:
                        continue
                    return m

                self.stage = STAGE_BAD_CAPTURES
                self.bad_cap_idx = 0

            # 6. Pick Bad Captures
            elif self.stage == STAGE_BAD_CAPTURES:
                while self.bad_cap_idx < self.num_bad_captures:
                    m = int(self.bad_captures[self.bad_cap_idx])
                    self.bad_cap_idx += 1
                    if m != self.tt_move:
                        return m

                self.stage = STAGE_DONE

        return MOVE_NONE
