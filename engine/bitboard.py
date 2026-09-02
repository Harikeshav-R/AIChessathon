"""Bitboard & Attack Table Precomputations.

Implements fast ray-sliding attacks, Knight, King, Pawn attack tables,
and bit manipulation intrinsics (popcount, lsb, clear_lsb).
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Preallocated Attack Arrays
PAWN_ATTACKS = np.zeros((2, 64), dtype=np.uint64)
KNIGHT_ATTACKS = np.zeros(64, dtype=np.uint64)
KING_ATTACKS = np.zeros(64, dtype=np.uint64)

# Direction offsets
DIRECTIONS = [8, -8, 1, -1, 9, -7, 7, -9]
BISHOP_DIRS = [9, -7, 7, -9]
ROOK_DIRS = [8, -8, 1, -1]


def _init_step_attacks() -> None:
    """Precomputes non-sliding piece attacks (Pawns, Knights, Kings)."""
    for sq in range(64):
        file = sq & 7
        rank = sq >> 3

        # White Pawn attacks (Northwest, Northeast)
        w_att = np.uint64(0)
        if file > 0 and rank < 7:
            w_att |= np.uint64(1) << np.uint64(sq + 7)
        if file < 7 and rank < 7:
            w_att |= np.uint64(1) << np.uint64(sq + 9)
        PAWN_ATTACKS[0, sq] = w_att

        # Black Pawn attacks (Southwest, Southeast)
        b_att = np.uint64(0)
        if file > 0 and rank > 0:
            b_att |= np.uint64(1) << np.uint64(sq - 9)
        if file < 7 and rank > 0:
            b_att |= np.uint64(1) << np.uint64(sq - 7)
        PAWN_ATTACKS[1, sq] = b_att

        # Knight attacks
        n_att = np.uint64(0)
        knight_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        for df, dr in knight_moves:
            nf, nr = file + df, rank + dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                n_att |= np.uint64(1) << np.uint64(nr * 8 + nf)
        KNIGHT_ATTACKS[sq] = n_att

        # King attacks
        k_att = np.uint64(0)
        king_moves = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        for df, dr in king_moves:
            nf, nr = file + df, rank + dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                k_att |= np.uint64(1) << np.uint64(nr * 8 + nf)
        KING_ATTACKS[sq] = k_att


_init_step_attacks()


@njit(inline="always")
def pop_count(bb: np.uint64) -> int:
    """Returns the number of set bits in a 64-bit word."""
    cnt = 0
    b = bb
    while b:
        b &= b - np.uint64(1)
        cnt += 1
    return cnt


@njit(inline="always")
def get_lsb(bb: np.uint64) -> int:
    """Returns the index (0..63) of the least significant set bit (or 64 if 0)."""
    if bb == np.uint64(0):
        return 64
    val = bb & (~bb + np.uint64(1))
    idx = 0
    if val > np.uint64(0xFFFFFFFF):
        val >>= np.uint64(32)
        idx += 32
    if val > np.uint64(0xFFFF):
        val >>= np.uint64(16)
        idx += 16
    if val > np.uint64(0xFF):
        val >>= np.uint64(8)
        idx += 8
    if val > np.uint64(0xF):
        val >>= np.uint64(4)
        idx += 4
    if val > np.uint64(0x3):
        val >>= np.uint64(2)
        idx += 2
    if val > np.uint64(0x1):
        idx += 1
    return idx


@njit(inline="always")
def clear_lsb(bb: np.uint64) -> np.uint64:
    """Clears the least significant set bit."""
    return bb & (bb - np.uint64(1))


@njit(fastmath=True, nogil=True)
def get_bishop_attacks(sq: int, occ: np.uint64) -> np.uint64:
    """Computes ray-cast bishop attacks on the fly."""
    attacks = np.uint64(0)
    file = sq & 7
    rank = sq >> 3

    # Northeast (+9)
    r, f = rank + 1, file + 1
    while r <= 7 and f <= 7:
        target = np.uint64(1) << np.uint64(r * 8 + f)
        attacks |= target
        if occ & target:
            break
        r += 1
        f += 1

    # Northwest (+7)
    r, f = rank + 1, file - 1
    while r <= 7 and f >= 0:
        target = np.uint64(1) << np.uint64(r * 8 + f)
        attacks |= target
        if occ & target:
            break
        r += 1
        f -= 1

    # Southeast (-7)
    r, f = rank - 1, file + 1
    while r >= 0 and f <= 7:
        target = np.uint64(1) << np.uint64(r * 8 + f)
        attacks |= target
        if occ & target:
            break
        r -= 1
        f += 1

    # Southwest (-9)
    r, f = rank - 1, file - 1
    while r >= 0 and f >= 0:
        target = np.uint64(1) << np.uint64(r * 8 + f)
        attacks |= target
        if occ & target:
            break
        r -= 1
        f -= 1

    return attacks


@njit(fastmath=True, nogil=True)
def get_rook_attacks(sq: int, occ: np.uint64) -> np.uint64:
    """Computes ray-cast rook attacks on the fly."""
    attacks = np.uint64(0)
    file = sq & 7
    rank = sq >> 3

    # North (+8)
    for r in range(rank + 1, 8):
        target = np.uint64(1) << np.uint64(r * 8 + file)
        attacks |= target
        if occ & target:
            break

    # South (-8)
    for r in range(rank - 1, -1, -1):
        target = np.uint64(1) << np.uint64(r * 8 + file)
        attacks |= target
        if occ & target:
            break

    # East (+1)
    for f in range(file + 1, 8):
        target = np.uint64(1) << np.uint64(rank * 8 + f)
        attacks |= target
        if occ & target:
            break

    # West (-1)
    for f in range(file - 1, -1, -1):
        target = np.uint64(1) << np.uint64(rank * 8 + f)
        attacks |= target
        if occ & target:
            break

    return attacks


@njit(fastmath=True, nogil=True)
def get_queen_attacks(sq: int, occ: np.uint64) -> np.uint64:
    """Computes queen attacks as union of bishop and rook attacks."""
    return get_bishop_attacks(sq, occ) | get_rook_attacks(sq, occ)
