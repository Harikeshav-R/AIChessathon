"""Position State, Bitboards, FEN parser, and Move Execution.

Maintains board representation (mailbox + bitboards), incremental Zobrist hashes,
attack detection, check evasion, and move legality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engine.bitboard import (
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    PAWN_ATTACKS,
    get_bishop_attacks,
    get_lsb,
    get_rook_attacks,
)
from engine.defs import (
    CASTLING_INDEX,
    COLOR_BLACK,
    COLOR_WHITE,
    DIR_NORTH,
    DIR_SOUTH,
    EP_INDEX,
    MOVE_CASTLING,
    MOVE_EN_PASSANT,
    MOVE_NONE,
    MOVE_PROMOTION,
    PHASE_BOUND,
    PHASE_ENDGAME,
    PHASE_MIDDLEGAME,
    PIECE_B_BISHOP,
    PIECE_B_KING,
    PIECE_B_KNIGHT,
    PIECE_B_PAWN,
    PIECE_B_QUEEN,
    PIECE_B_ROOK,
    PIECE_BLANK,
    PIECE_W_BISHOP,
    PIECE_W_KING,
    PIECE_W_KNIGHT,
    PIECE_W_PAWN,
    PIECE_W_QUEEN,
    PIECE_W_ROOK,
    PT_BISHOP,
    PT_KING,
    PT_KNIGHT,
    PT_PAWN,
    PT_QUEEN,
    PT_ROOK,
    SIDE_INDEX,
    SIDE_KINGSIDE,
    SIDE_QUEENSIDE,
    SQUARE_NONE,
    ZOBRIST_KEYS,
    extract_from,
    extract_promo,
    extract_to,
    extract_type,
    get_piece_color,
    get_piece_type,
    get_zobrist_piece_key,
)

if TYPE_CHECKING:
    from engine.nnue import NNUEState


class Position:
    """Board position state maintaining bitboards and mailbox arrays."""

    def __init__(self) -> None:
        self.board = np.zeros(64, dtype=np.uint8)
        self.colors_bb = np.zeros(2, dtype=np.uint64)
        self.pieces_bb = np.zeros(7, dtype=np.uint64)
        self.material_count = np.zeros(10, dtype=np.int32)
        self.castling_squares = np.full((2, 2), SQUARE_NONE, dtype=np.uint8)
        self.ep_square: int = SQUARE_NONE
        self.color: int = COLOR_WHITE
        self.halfmoves: int = 0
        self.zobrist_key: np.uint64 = np.uint64(0)
        self.pawn_key: np.uint64 = np.uint64(0)
        self.non_pawn_key = np.zeros(2, dtype=np.uint64)

    def clone(self) -> Position:
        p = Position()
        p.board = self.board.copy()
        p.colors_bb = self.colors_bb.copy()
        p.pieces_bb = self.pieces_bb.copy()
        p.material_count = self.material_count.copy()
        p.castling_squares = self.castling_squares.copy()
        p.ep_square = self.ep_square
        p.color = self.color
        p.halfmoves = self.halfmoves
        p.zobrist_key = self.zobrist_key
        p.pawn_key = self.pawn_key
        p.non_pawn_key = self.non_pawn_key.copy()
        return p


def get_king_pos(pos: Position, color: int) -> int:
    bb = pos.colors_bb[color] & pos.pieces_bb[PT_KING]
    return get_lsb(bb)


def attacks_square(pos: Position, sq: int, color: int, occ: np.uint64 | None = None) -> np.uint64:
    """Checks which pieces of `color` attack square `sq`."""
    if sq < 0 or sq >= 64:
        return np.uint64(0)

    if occ is None:
        occ = pos.colors_bb[0] | pos.colors_bb[1]

    bishops = pos.pieces_bb[PT_BISHOP] | pos.pieces_bb[PT_QUEEN]
    rooks = pos.pieces_bb[PT_ROOK] | pos.pieces_bb[PT_QUEEN]
    opp_color = color ^ 1

    pawn_att = PAWN_ATTACKS[opp_color, sq] & pos.pieces_bb[PT_PAWN]
    knight_att = KNIGHT_ATTACKS[sq] & pos.pieces_bb[PT_KNIGHT]
    bishop_att = get_bishop_attacks(sq, occ) & bishops
    rook_att = get_rook_attacks(sq, occ) & rooks
    king_att = KING_ATTACKS[sq] & pos.pieces_bb[PT_KING]

    all_att = pawn_att | knight_att | bishop_att | rook_att | king_att
    res: np.uint64 = np.uint64(all_att & pos.colors_bb[color])
    return res


def is_in_check(pos: Position, color: int) -> bool:
    king_sq = get_king_pos(pos, color)
    if king_sq >= 64:
        return False
    return bool(attacks_square(pos, king_sq, color ^ 1))


def total_mat(pos: Position) -> int:
    """Total non-king material value on board."""
    return int(
        (pos.material_count[0] + pos.material_count[1]) * 100
        + (pos.material_count[2] + pos.material_count[3]) * 300
        + (pos.material_count[4] + pos.material_count[5]) * 300
        + (pos.material_count[6] + pos.material_count[7]) * 500
        + (pos.material_count[8] + pos.material_count[9]) * 900
    )


def material_eval(pos: Position) -> int:
    """Material balance from side to move's perspective."""
    diff = int(
        (pos.material_count[0] - pos.material_count[1]) * 100
        + (pos.material_count[2] - pos.material_count[3]) * 300
        + (pos.material_count[4] - pos.material_count[5]) * 300
        + (pos.material_count[6] - pos.material_count[7]) * 500
        + (pos.material_count[8] - pos.material_count[9]) * 900
    )
    return -diff if pos.color == COLOR_BLACK else diff


def is_material_draw(pos: Position) -> bool:
    """Detects insufficient material draws."""
    if (
            pos.material_count[0]
            or pos.material_count[1]
            or pos.material_count[6]
            or pos.material_count[7]
            or pos.material_count[8]
            or pos.material_count[9]
    ):
        return False

    if (
            pos.material_count[4] > 1
            or pos.material_count[2] > 2
            or (pos.material_count[2] and pos.material_count[4])
    ):
        return False

    return not (
            pos.material_count[5] > 1
            or pos.material_count[3] > 2
            or (pos.material_count[3] and pos.material_count[5])
    )


def set_fen(pos: Position, fen: str) -> None:
    """Initializes Position from a FEN string."""
    tokens = fen.strip().split()
    if not tokens:
        return

    pos.board.fill(0)
    pos.colors_bb.fill(0)
    pos.pieces_bb.fill(0)
    pos.material_count.fill(0)
    pos.castling_squares.fill(SQUARE_NONE)
    pos.ep_square = SQUARE_NONE
    pos.halfmoves = 0

    fen_board = tokens[0]
    sq = 56
    for char in fen_board:
        if char == "/":
            sq -= 16
        elif char.isdigit():
            sq += int(char)
        else:
            piece_map = {
                "P": PIECE_W_PAWN, "N": PIECE_W_KNIGHT, "B": PIECE_W_BISHOP,
                "R": PIECE_W_ROOK, "Q": PIECE_W_QUEEN, "K": PIECE_W_KING,
                "p": PIECE_B_PAWN, "n": PIECE_B_KNIGHT, "b": PIECE_B_BISHOP,
                "r": PIECE_B_ROOK, "q": PIECE_B_QUEEN, "k": PIECE_B_KING,
            }
            if char in piece_map:
                piece = piece_map[char]
                color = get_piece_color(piece)
                pt = get_piece_type(piece)

                pos.board[sq] = piece
                pos.colors_bb[color] |= np.uint64(1) << np.uint64(sq)
                pos.pieces_bb[pt] |= np.uint64(1) << np.uint64(sq)

                if pt != PT_KING:
                    pos.material_count[piece - 2] += 1
            sq += 1

    pos.color = COLOR_BLACK if (len(tokens) > 1 and tokens[1] == "b") else COLOR_WHITE

    if len(tokens) > 2:
        castling = tokens[2]
        if castling != "-":
            for c in castling:
                color = COLOR_BLACK if c.islower() else COLOR_WHITE
                c_low = c.lower()
                base = 56 if color == COLOR_BLACK else 0
                if c_low == "k":
                    pos.castling_squares[color, SIDE_KINGSIDE] = base + 7
                elif c_low == "q":
                    pos.castling_squares[color, SIDE_QUEENSIDE] = base

    if len(tokens) > 3 and tokens[3] != "-":
        ep_str = tokens[3]
        file = ord(ep_str[0]) - ord("a")
        rank = ord(ep_str[1]) - ord("1")
        pos.ep_square = rank * 8 + file

    if len(tokens) > 4:
        pos.halfmoves = int(tokens[4])

    recalculate_zobrist(pos)


def recalculate_zobrist(pos: Position) -> None:
    """Computes Zobrist key, pawn key, and non-pawn key from scratch."""
    z_key = np.uint64(0)
    p_key = np.uint64(0)
    np_key_w = np.uint64(0)
    np_key_b = np.uint64(0)

    for sq in range(64):
        piece = int(pos.board[sq])
        if piece != PIECE_BLANK:
            idx = get_zobrist_piece_key(piece, sq)
            key = ZOBRIST_KEYS[idx]
            z_key ^= key
            pt = get_piece_type(piece)
            color = get_piece_color(piece)
            if pt == PT_PAWN:
                p_key ^= key
            else:
                if color == COLOR_WHITE:
                    np_key_w ^= key
                else:
                    np_key_b ^= key

    if pos.color == COLOR_WHITE:
        z_key ^= ZOBRIST_KEYS[SIDE_INDEX]

    if pos.ep_square != SQUARE_NONE:
        z_key ^= ZOBRIST_KEYS[EP_INDEX]

    for color in (COLOR_WHITE, COLOR_BLACK):
        for side in (SIDE_QUEENSIDE, SIDE_KINGSIDE):
            if pos.castling_squares[color, side] != SQUARE_NONE:
                z_key ^= ZOBRIST_KEYS[CASTLING_INDEX + color * 2 + side]

    pos.zobrist_key = z_key
    pos.pawn_key = p_key
    pos.non_pawn_key[COLOR_WHITE] = np_key_w
    pos.non_pawn_key[COLOR_BLACK] = np_key_b


def make_move(
        pos: Position,
        move: int,
        nnue_state: NNUEState | None = None,
        phase: int = PHASE_MIDDLEGAME,
) -> int:
    """Executes move on Position, updating bitboards, Zobrist keys, and NNUE accumulators.

    Returns the resulting phase.
    """
    pos.halfmoves += 1

    if move == MOVE_NONE:
        pos.color ^= 1
        if pos.ep_square != SQUARE_NONE:
            pos.zobrist_key ^= ZOBRIST_KEYS[EP_INDEX]
            pos.ep_square = SQUARE_NONE
        pos.zobrist_key ^= ZOBRIST_KEYS[SIDE_INDEX]
        return phase

    from_sq = extract_from(move)
    to_sq = extract_to(move)
    move_type = extract_type(move)
    color = pos.color
    opp_color = color ^ 1

    from_piece = int(pos.board[from_sq])
    from_type = get_piece_type(from_piece)
    to_piece = from_piece

    captured_piece = int(pos.board[to_sq])
    captured_sq = to_sq
    new_ep = SQUARE_NONE
    base_rank = 56 if color == COLOR_BLACK else 0

    if move_type == MOVE_CASTLING:
        king_pos = get_king_pos(pos, color)
        side = SIDE_KINGSIDE if to_sq > king_pos else SIDE_QUEENSIDE
        to_sq = base_rank + 6 if side == SIDE_KINGSIDE else base_rank + 2
        rook_from = int(pos.castling_squares[color, side])
        rook_to = base_rank + 5 if side == SIDE_KINGSIDE else base_rank + 3
        rook_piece = PIECE_W_ROOK + color

        pos.board[rook_from] = PIECE_BLANK
        pos.board[rook_to] = rook_piece
        rook_mask = (np.uint64(1) << np.uint64(rook_from)) | (np.uint64(1) << np.uint64(rook_to))
        pos.colors_bb[color] ^= rook_mask
        pos.pieces_bb[PT_ROOK] ^= rook_mask

        r_key_from = ZOBRIST_KEYS[get_zobrist_piece_key(rook_piece, rook_from)]
        r_key_to = ZOBRIST_KEYS[get_zobrist_piece_key(rook_piece, rook_to)]
        pos.zobrist_key ^= r_key_from ^ r_key_to
        pos.non_pawn_key[color] ^= r_key_from ^ r_key_to

        if nnue_state is not None:
            nnue_state.update_2add_2sub(
                from_piece, from_sq, to_sq, rook_piece, rook_from, rook_to, phase
            )

    elif captured_piece != PIECE_BLANK:
        pos.halfmoves = 0
        c_type = get_piece_type(captured_piece)
        if c_type != PT_KING:
            pos.material_count[captured_piece - 2] -= 1
        c_mask = np.uint64(1) << np.uint64(captured_sq)
        pos.colors_bb[opp_color] ^= c_mask
        pos.pieces_bb[c_type] ^= c_mask

        c_key = ZOBRIST_KEYS[get_zobrist_piece_key(captured_piece, captured_sq)]
        pos.zobrist_key ^= c_key
        if c_type == PT_PAWN:
            pos.pawn_key ^= c_key
        else:
            pos.non_pawn_key[opp_color] ^= c_key

    elif move_type == MOVE_EN_PASSANT:
        pos.halfmoves = 0
        captured_sq = to_sq + (DIR_SOUTH if color == COLOR_WHITE else DIR_NORTH)
        captured_piece = int(pos.board[captured_sq])
        pos.board[captured_sq] = PIECE_BLANK
        pos.material_count[captured_piece - 2] -= 1

        c_mask = np.uint64(1) << np.uint64(captured_sq)
        pos.colors_bb[opp_color] ^= c_mask
        pos.pieces_bb[PT_PAWN] ^= c_mask

        c_key = ZOBRIST_KEYS[get_zobrist_piece_key(captured_piece, captured_sq)]
        pos.zobrist_key ^= c_key
        pos.pawn_key ^= c_key

    pos.board[from_sq] = PIECE_BLANK
    pos.board[to_sq] = to_piece

    if from_type == PT_PAWN:
        pos.halfmoves = 0
        if move_type == MOVE_PROMOTION:
            promo = extract_promo(move)
            promo_pt = PT_KNIGHT + promo
            to_piece = promo_pt * 2 + color
            pos.board[to_sq] = to_piece

            pos.material_count[from_piece - 2] -= 1
            pos.material_count[to_piece - 2] += 1
            pos.pieces_bb[PT_PAWN] ^= np.uint64(1) << np.uint64(to_sq)
            pos.pieces_bb[promo_pt] |= np.uint64(1) << np.uint64(to_sq)
        elif abs(to_sq - from_sq) == 16:
            new_ep = (to_sq + from_sq) // 2

    elif from_type == PT_KING:
        for side in (SIDE_QUEENSIDE, SIDE_KINGSIDE):
            if pos.castling_squares[color, side] != SQUARE_NONE:
                pos.zobrist_key ^= ZOBRIST_KEYS[CASTLING_INDEX + color * 2 + side]
                pos.castling_squares[color, side] = SQUARE_NONE

    for c in (color, opp_color):
        for side in (SIDE_QUEENSIDE, SIDE_KINGSIDE):
            sq_c = pos.castling_squares[c, side]
            if sq_c != SQUARE_NONE and (from_sq == sq_c or to_sq == sq_c):
                pos.zobrist_key ^= ZOBRIST_KEYS[CASTLING_INDEX + c * 2 + side]
                pos.castling_squares[c, side] = SQUARE_NONE

    from_mask = np.uint64(1) << np.uint64(from_sq)
    to_mask = np.uint64(1) << np.uint64(to_sq)
    pos.colors_bb[color] ^= from_mask | to_mask

    if move_type != MOVE_PROMOTION:
        pos.pieces_bb[from_type] ^= from_mask | to_mask
    else:
        pos.pieces_bb[from_type] ^= from_mask

    f_key = ZOBRIST_KEYS[get_zobrist_piece_key(from_piece, from_sq)]
    t_key = ZOBRIST_KEYS[get_zobrist_piece_key(to_piece, to_sq)]
    pos.zobrist_key ^= f_key ^ t_key

    if from_type == PT_PAWN:
        pos.pawn_key ^= f_key
        if move_type != MOVE_PROMOTION:
            pos.pawn_key ^= t_key
        else:
            pos.non_pawn_key[color] ^= t_key
    else:
        pos.non_pawn_key[color] ^= f_key ^ t_key

    pos.color ^= 1
    pos.zobrist_key ^= ZOBRIST_KEYS[SIDE_INDEX]

    if (pos.ep_square == SQUARE_NONE) ^ (new_ep == SQUARE_NONE):
        pos.zobrist_key ^= ZOBRIST_KEYS[EP_INDEX]
    pos.ep_square = new_ep

    if nnue_state is not None and move_type != MOVE_CASTLING:
        if captured_piece != PIECE_BLANK:
            if phase == PHASE_MIDDLEGAME and total_mat(pos) < PHASE_BOUND:
                phase = PHASE_ENDGAME
                nnue_state.reset_nnue(pos.board, phase)
            else:
                nnue_state.update_1add_2sub(
                    from_piece, from_sq, to_piece, to_sq, captured_piece, captured_sq, phase
                )
        else:
            nnue_state.update_1add_1sub(from_piece, from_sq, to_piece, to_sq, phase)

    return phase


def move_to_uci(move: int) -> str:
    """Converts internal 16-bit move to standard UCI format."""
    f = extract_from(move)
    t = extract_to(move)
    m_type = extract_type(move)

    from_str = chr(ord("a") + (f & 7)) + str((f >> 3) + 1)
    to_str = chr(ord("a") + (t & 7)) + str((t >> 3) + 1)

    if m_type == MOVE_PROMOTION:
        promo_str = "nbrq"[extract_promo(move)]
        return from_str + to_str + promo_str
    return from_str + to_str
