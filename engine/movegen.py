"""Fast Pseudo-Legal & Legal Move Generator.

Wraps the JIT fast move generator with legality checking.
"""

from __future__ import annotations

import numpy as np

from engine.defs import (
    MOVE_CASTLING,
    extract_type,
)
from engine.fast_search import (
    generate_moves_fast,
    is_in_check_fast,
)
from engine.position import (
    Position,
    make_move,
)

GEN_ALL: int = 0
GEN_CAPTURES: int = 1
GEN_QUIETS: int = 2


def generate_moves(pos: Position, move_list: np.ndarray, gen_type: int = GEN_ALL) -> int:
    """Generates pseudo-legal moves into preallocated move_list array."""
    return generate_moves_fast(
        pos.board,
        pos.colors_bb,
        pos.pieces_bb,
        pos.castling_squares,
        pos.ep_square,
        pos.color,
        move_list,
        0,
        gen_type,
    )


def is_legal(pos: Position, move: int) -> bool:
    """Verifies that executing pseudo-legal move does not leave our king in check."""
    m_type = extract_type(move)
    color = pos.color

    if m_type == MOVE_CASTLING:
        return True

    clone = pos.clone()
    make_move(clone, move)
    return not is_in_check_fast(clone.colors_bb, clone.pieces_bb, color)


def legal_moves(pos: Position) -> list[int]:
    """Returns all strictly legal moves for the current position."""
    buf = np.zeros(256, dtype=np.uint16)
    n = generate_moves_fast(
        pos.board,
        pos.colors_bb,
        pos.pieces_bb,
        pos.castling_squares,
        pos.ep_square,
        pos.color,
        buf,
        0,
        GEN_ALL,
    )
    result: list[int] = []
    for i in range(n):
        m = int(buf[i])
        if is_legal(pos, m):
            result.append(m)
    return result
