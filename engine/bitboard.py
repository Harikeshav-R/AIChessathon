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

# Preallocated 8-directional Ray Masks (North, South, East, West, NE, NW, SE, SW)
RAY_MASKS = np.zeros((8, 64), dtype=np.uint64)


def _init_step_attacks() -> None:
    """Precomputes non-sliding and ray-sliding attack masks."""
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

        # North (+8)
        m = np.uint64(0)
        for nr in range(rank + 1, 8):
            m |= np.uint64(1) << np.uint64(nr * 8 + file)
        RAY_MASKS[0, sq] = m

        # South (-8)
        m = np.uint64(0)
        for nr in range(rank - 1, -1, -1):
            m |= np.uint64(1) << np.uint64(nr * 8 + file)
        RAY_MASKS[1, sq] = m

        # East (+1)
        m = np.uint64(0)
        for nf in range(file + 1, 8):
            m |= np.uint64(1) << np.uint64(rank * 8 + nf)
        RAY_MASKS[2, sq] = m

        # West (-1)
        m = np.uint64(0)
        for nf in range(file - 1, -1, -1):
            m |= np.uint64(1) << np.uint64(rank * 8 + nf)
        RAY_MASKS[3, sq] = m

        # Northeast (+9)
        m = np.uint64(0)
        nr, nf = rank + 1, file + 1
        while nr <= 7 and nf <= 7:
            m |= np.uint64(1) << np.uint64(nr * 8 + nf)
            nr += 1
            nf += 1
        RAY_MASKS[4, sq] = m

        # Northwest (+7)
        m = np.uint64(0)
        nr, nf = rank + 1, file - 1
        while nr <= 7 and nf >= 0:
            m |= np.uint64(1) << np.uint64(nr * 8 + nf)
            nr += 1
            nf -= 1
        RAY_MASKS[5, sq] = m

        # Southeast (-7)
        m = np.uint64(0)
        nr, nf = rank - 1, file + 1
        while nr >= 0 and nf <= 7:
            m |= np.uint64(1) << np.uint64(nr * 8 + nf)
            nr -= 1
            nf += 1
        RAY_MASKS[6, sq] = m

        # Southwest (-9)
        m = np.uint64(0)
        nr, nf = rank - 1, file - 1
        while nr >= 0 and nf >= 0:
            m |= np.uint64(1) << np.uint64(nr * 8 + nf)
            nr -= 1
            nf -= 1
        RAY_MASKS[7, sq] = m


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
def get_msb(bb: np.uint64) -> int:
    """Returns the index (0..63) of the most significant set bit (or 64 if 0)."""
    if bb == np.uint64(0):
        return 64
    val = bb
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
    """Computes fast branchless bishop attacks using bitscan ray obstruction difference."""
    att = np.uint64(0)
    # Northeast (+9, positive ray)
    b = occ & RAY_MASKS[4, sq]
    att |= (RAY_MASKS[4, sq] ^ RAY_MASKS[4, get_lsb(b)]) if b else RAY_MASKS[4, sq]
    # Northwest (+7, positive ray)
    b = occ & RAY_MASKS[5, sq]
    att |= (RAY_MASKS[5, sq] ^ RAY_MASKS[5, get_lsb(b)]) if b else RAY_MASKS[5, sq]
    # Southeast (-7, negative ray)
    b = occ & RAY_MASKS[6, sq]
    att |= (RAY_MASKS[6, sq] ^ RAY_MASKS[6, get_msb(b)]) if b else RAY_MASKS[6, sq]
    # Southwest (-9, negative ray)
    b = occ & RAY_MASKS[7, sq]
    att |= (RAY_MASKS[7, sq] ^ RAY_MASKS[7, get_msb(b)]) if b else RAY_MASKS[7, sq]
    return np.uint64(att)


@njit(fastmath=True, nogil=True)
def get_rook_attacks(sq: int, occ: np.uint64) -> np.uint64:
    """Computes fast branchless rook attacks using bitscan ray obstruction difference."""
    att = np.uint64(0)
    # North (+8, positive ray)
    b = occ & RAY_MASKS[0, sq]
    att |= (RAY_MASKS[0, sq] ^ RAY_MASKS[0, get_lsb(b)]) if b else RAY_MASKS[0, sq]
    # South (-8, negative ray)
    b = occ & RAY_MASKS[1, sq]
    att |= (RAY_MASKS[1, sq] ^ RAY_MASKS[1, get_msb(b)]) if b else RAY_MASKS[1, sq]
    # East (+1, positive ray)
    b = occ & RAY_MASKS[2, sq]
    att |= (RAY_MASKS[2, sq] ^ RAY_MASKS[2, get_lsb(b)]) if b else RAY_MASKS[2, sq]
    # West (-1, negative ray)
    b = occ & RAY_MASKS[3, sq]
    att |= (RAY_MASKS[3, sq] ^ RAY_MASKS[3, get_msb(b)]) if b else RAY_MASKS[3, sq]
    return np.uint64(att)


@njit(fastmath=True, nogil=True)
def get_queen_attacks(sq: int, occ: np.uint64) -> np.uint64:
    """Computes queen attacks as union of bishop and rook attacks."""
    return get_bishop_attacks(sq, occ) | get_rook_attacks(sq, occ)
