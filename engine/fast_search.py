# ruff: noqa: E501
"""Numba JIT Fast Core Kernel for Search, Move Generation, and Move Picking.

Provides high-speed zero-allocation tree traversal operating directly on preallocated
contiguous NumPy stack arrays in native machine code.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from engine.bitboard import (
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    PAWN_ATTACKS,
    clear_lsb,
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
    ENTRY_EXACT,
    ENTRY_LBOUND,
    ENTRY_NONE,
    ENTRY_UBOUND,
    EP_INDEX,
    MAX_SEARCH_DEPTH,
    MOVE_CASTLING,
    MOVE_EN_PASSANT,
    MOVE_NONE,
    MOVE_NORMAL,
    MOVE_PROMOTION,
    PIECE_BLANK,
    PIECE_W_ROOK,
    PROMO_BISHOP,
    PROMO_KNIGHT,
    PROMO_QUEEN,
    PROMO_ROOK,
    PT_BISHOP,
    PT_KING,
    PT_KNIGHT,
    PT_NONE,
    PT_PAWN,
    PT_QUEEN,
    PT_ROOK,
    SCORE_LOST,
    SCORE_MATE,
    SCORE_NONE,
    SCORE_WIN,
    SIDE_INDEX,
    SIDE_KINGSIDE,
    SIDE_QUEENSIDE,
    SQUARE_NONE,
    ZOBRIST_KEYS,
    extract_from,
    extract_promo,
    extract_to,
    extract_type,
    get_piece_type,
    get_zobrist_piece_key,
    pack_move,
    pack_move_promo,
)
from engine.nnue import (
    feature_indices,
    nnue_update_1add_1sub,
    nnue_update_1add_2sub,
    nnue_update_2add_2sub,
    phase_to_index,
    screlu_flatten,
)
from engine.params import (
    CORR_WEIGHT,
    FP_DEPTH,
    FP_MARGIN,
    HIST_BONUS,
    HIST_MAX,
    IIR_MIN_DEPTH,
    LMP_BASE,
    LMP_DEPTH,
    LMR_MIN_DEPTH,
    LMR_TABLE,
    NMP_BASE,
    NMP_DEPTH_DIV,
    NMP_EVAL_DIV,
    NMP_MIN_DEPTH,
    PROBCUT_MARGIN,
    RFP_MARGIN,
    RFP_MAX_DEPTH,
    SE_DEPTH,
    SE_DOUBLE_EXT_MARGIN,
    SE_TRIPLE_EXT_MARGIN,
    SEE_PRUNING_DEPTH,
    SEE_QUIET_MARGIN,
)


@njit(fastmath=True, nogil=True)
def get_least_valuable_attacker(
        colors_bb: np.ndarray,
        pieces_bb: np.ndarray,
        sq: int,
        side: int,
        occ: np.uint64,
) -> tuple[int, int]:
    """Finds square and piece type of least valuable attacker for SEE."""
    us = colors_bb[side] & occ
    opp = side ^ 1
    # 1. Pawns
    p_att = PAWN_ATTACKS[opp, sq] & pieces_bb[PT_PAWN] & us
    if p_att:
        from_sq = get_lsb(p_att)
        return from_sq, PT_PAWN
    # 2. Knights
    n_att = KNIGHT_ATTACKS[sq] & pieces_bb[PT_KNIGHT] & us
    if n_att:
        from_sq = get_lsb(n_att)
        return from_sq, PT_KNIGHT
    # 3. Bishops
    b_att = get_bishop_attacks(sq, occ) & pieces_bb[PT_BISHOP] & us
    if b_att:
        from_sq = get_lsb(b_att)
        return from_sq, PT_BISHOP
    # 4. Rooks
    r_att = get_rook_attacks(sq, occ) & pieces_bb[PT_ROOK] & us
    if r_att:
        from_sq = get_lsb(r_att)
        return from_sq, PT_ROOK
    # 5. Queens
    q_att = (get_bishop_attacks(sq, occ) | get_rook_attacks(sq, occ)) & pieces_bb[PT_QUEEN] & us
    if q_att:
        from_sq = get_lsb(q_att)
        return from_sq, PT_QUEEN
    # 6. King
    k_att = KING_ATTACKS[sq] & pieces_bb[PT_KING] & us
    if k_att:
        from_sq = get_lsb(k_att)
        return from_sq, PT_KING
    return 64, PT_NONE


@njit(inline="always")
def get_see_value(pt: int) -> int:
    p = int(pt)
    if p == 1:
        return 100
    elif p == 2 or p == 3:
        return 450
    elif p == 4:
        return 650
    elif p == 5:
        return 1250
    elif p == 6:
        return 10000
    return 0


@njit(fastmath=True, nogil=True)
def see_fast(
        board: np.ndarray,
        colors_bb: np.ndarray,
        pieces_bb: np.ndarray,
        move: int,
        threshold: int,
) -> bool:
    """Fast Static Exchange Evaluation (SEE) threshold filter."""
    from_sq = int(extract_from(move))
    to_sq = int(extract_to(move))
    m_type = int(extract_type(move))

    from_piece = int(board[from_sq])
    to_piece = int(board[to_sq])
    from_pt = int(get_piece_type(from_piece))
    to_pt = int(get_piece_type(to_piece)) if to_piece != 0 else 0

    if m_type == MOVE_CASTLING:
        return threshold <= 0

    val = 0
    if to_pt != 0:
        val = get_see_value(to_pt)
    elif m_type == MOVE_EN_PASSANT:
        val = get_see_value(1)

    if m_type == MOVE_PROMOTION:
        promo_pt = int(extract_promo(move)) + 2
        val += get_see_value(promo_pt) - get_see_value(1)

    if val < threshold:
        return False

    val -= get_see_value(from_pt)
    if val >= threshold:
        return True

    gain = np.zeros(32, dtype=np.int32)
    gain[0] = val + get_see_value(from_pt)

    color = from_piece & 1
    side = color ^ 1
    occ = (colors_bb[0] | colors_bb[1]) ^ (np.uint64(1) << np.uint64(from_sq))

    attacker_pt = from_pt
    d = 1

    while d < 32:
        gain[d] = get_see_value(attacker_pt) - gain[d - 1]
        if max(-gain[d - 1], gain[d]) < threshold:
            break

        from_sq, next_attacker_pt = get_least_valuable_attacker(colors_bb, pieces_bb, to_sq, side, occ)
        if next_attacker_pt == 0:
            break

        attacker_pt = int(next_attacker_pt)
        occ ^= np.uint64(1) << np.uint64(from_sq)
        side ^= 1
        d += 1

    while d > 1:
        d -= 1
        gain[d - 1] = -max(-gain[d - 1], gain[d])

    return bool(gain[0] >= threshold)


@njit(fastmath=True, nogil=True)
def get_king_sq(colors_bb: np.ndarray, pieces_bb: np.ndarray, color: int) -> int:
    bb = colors_bb[color] & pieces_bb[PT_KING]
    return get_lsb(bb)


@njit(fastmath=True, nogil=True)
def attacks_square_fast(
        colors_bb: np.ndarray,
        pieces_bb: np.ndarray,
        sq: int,
        color: int,
        occ: np.uint64,
) -> np.uint64:
    if sq < 0 or sq >= 64:
        return np.uint64(0)

    opp_color = color ^ 1
    bishops = pieces_bb[PT_BISHOP] | pieces_bb[PT_QUEEN]
    rooks = pieces_bb[PT_ROOK] | pieces_bb[PT_QUEEN]

    pawn_att = PAWN_ATTACKS[opp_color, sq] & pieces_bb[PT_PAWN]
    knight_att = KNIGHT_ATTACKS[sq] & pieces_bb[PT_KNIGHT]
    bishop_att = get_bishop_attacks(sq, occ) & bishops
    rook_att = get_rook_attacks(sq, occ) & rooks
    king_att = KING_ATTACKS[sq] & pieces_bb[PT_KING]

    all_att = pawn_att | knight_att | bishop_att | rook_att | king_att
    return np.uint64(all_att & colors_bb[color])


@njit(fastmath=True, nogil=True)
def is_in_check_fast(colors_bb: np.ndarray, pieces_bb: np.ndarray, color: int) -> bool:
    k_sq = get_king_sq(colors_bb, pieces_bb, color)
    if k_sq >= 64:
        return False
    occ = colors_bb[0] | colors_bb[1]
    return bool(attacks_square_fast(colors_bb, pieces_bb, k_sq, color ^ 1, occ))


@njit(fastmath=True, nogil=True)
def total_mat_fast(material_stack: np.ndarray, ply: int) -> int:
    return int(
        (material_stack[ply, 0] + material_stack[ply, 1]) * 100
        + (material_stack[ply, 2] + material_stack[ply, 3]) * 300
        + (material_stack[ply, 4] + material_stack[ply, 5]) * 300
        + (material_stack[ply, 6] + material_stack[ply, 7]) * 500
        + (material_stack[ply, 8] + material_stack[ply, 9]) * 900
    )


@njit(fastmath=True, nogil=True)
def eval_position_fast(
        ply: int,
        color_stack: np.ndarray,
        phase_stack: np.ndarray,
        material_stack: np.ndarray,
        pawn_stack: np.ndarray,
        non_pawn_stack: np.ndarray,
        white_acc_stack: np.ndarray,
        black_acc_stack: np.ndarray,
        output_weights: np.ndarray,
        output_biases: np.ndarray,
        pawn_corr_hist: np.ndarray,
        non_pawn_corr_hist: np.ndarray,
) -> int:
    color = int(color_stack[ply])
    phase = int(phase_stack[ply])
    net_idx = phase_to_index(phase)

    if color == COLOR_WHITE:
        us = white_acc_stack[ply]
        them = black_acc_stack[ply]
    else:
        us = black_acc_stack[ply]
        them = white_acc_stack[ply]

    raw_eval = screlu_flatten(us, them, output_weights[net_idx], output_biases[net_idx])
    adjusted = int(raw_eval)

    p_idx = int(pawn_stack[ply] & np.uint64(16383))
    corr = int(pawn_corr_hist[color, p_idx])

    np_w_idx = int(non_pawn_stack[ply, COLOR_WHITE] & np.uint64(16383))
    np_b_idx = int(non_pawn_stack[ply, COLOR_BLACK] & np.uint64(16383))
    corr += int(non_pawn_corr_hist[color, COLOR_WHITE, np_w_idx])
    corr += int(non_pawn_corr_hist[color, COLOR_BLACK, np_b_idx])

    final_eval = adjusted + (CORR_WEIGHT * corr) // 512
    if final_eval > SCORE_WIN - 1:
        return SCORE_WIN - 1
    elif final_eval < SCORE_LOST + 1:
        return SCORE_LOST + 1
    return final_eval


@njit(fastmath=True, nogil=True)
def generate_moves_fast(
        board: np.ndarray,
        colors_bb: np.ndarray,
        pieces_bb: np.ndarray,
        castling_squares: np.ndarray,
        ep_square: int,
        color: int,
        move_list: np.ndarray,
        offset: int,
        gen_type: int = 0,
) -> int:
    count = offset
    opp_color = color ^ 1

    king_sq = get_king_sq(colors_bb, pieces_bb, color)
    if king_sq >= 64:
        return 0

    us = colors_bb[color]
    them = colors_bb[opp_color] & ~pieces_bb[PT_KING]
    occ = us | colors_bb[opp_color]
    empty = ~occ

    checkers = attacks_square_fast(colors_bb, pieces_bb, king_sq, opp_color, occ)
    num_checkers = 0
    c_tmp = checkers
    while c_tmp:
        c_tmp = clear_lsb(c_tmp)
        num_checkers += 1

    target_mask = ~us
    if gen_type == 1:  # GEN_CAPTURES
        target_mask = them
    elif gen_type == 2:  # GEN_QUIETS
        target_mask = empty

    # 1. King moves
    k_moves = KING_ATTACKS[king_sq] & ~us & target_mask
    while k_moves:
        to_sq = get_lsb(k_moves)
        k_moves = clear_lsb(k_moves)
        move_list[count] = pack_move(king_sq, to_sq, MOVE_NORMAL)
        count += 1

    if num_checkers > 1:
        return count - offset

    # Castling
    if num_checkers == 0 and gen_type != 1:
        base_rank = 56 if color == COLOR_BLACK else 0
        k_sq = base_rank + 4

        # Kingside
        if castling_squares[color, SIDE_KINGSIDE] != SQUARE_NONE:
            sq_f = base_rank + 5
            sq_g = base_rank + 6
            if (
                    board[sq_f] == 0
                    and board[sq_g] == 0
                    and not attacks_square_fast(colors_bb, pieces_bb, sq_f, opp_color, occ)
                    and not attacks_square_fast(colors_bb, pieces_bb, sq_g, opp_color, occ)
            ):
                move_list[count] = pack_move(k_sq, sq_g, MOVE_CASTLING)
                count += 1

        # Queenside
        if castling_squares[color, SIDE_QUEENSIDE] != SQUARE_NONE:
            sq_d = base_rank + 3
            sq_c = base_rank + 2
            sq_b = base_rank + 1
            if (
                    board[sq_d] == 0
                    and board[sq_c] == 0
                    and board[sq_b] == 0
                    and not attacks_square_fast(colors_bb, pieces_bb, sq_d, opp_color, occ)
                    and not attacks_square_fast(colors_bb, pieces_bb, sq_c, opp_color, occ)
            ):
                move_list[count] = pack_move(k_sq, sq_c, MOVE_CASTLING)
                count += 1

    check_mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    if num_checkers == 1:
        checker_sq = get_lsb(checkers)
        check_mask = np.uint64(1) << np.uint64(checker_sq)
        checker_pt = int(board[checker_sq]) >> 1
        if checker_pt in (PT_BISHOP, PT_ROOK, PT_QUEEN):
            r1, f1 = king_sq >> 3, king_sq & 7
            r2, f2 = checker_sq >> 3, checker_sq & 7
            dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)
            df = 0 if f1 == f2 else (1 if f2 > f1 else -1)
            cur_r, cur_f = r1 + dr, f1 + df
            while (cur_r, cur_f) != (r2, f2):
                check_mask |= np.uint64(1) << np.uint64(cur_r * 8 + cur_f)
                cur_r += dr
                cur_f += df

    target_mask &= check_mask

    # 2. Pawn moves
    p_bb = pieces_bb[PT_PAWN] & us
    p_step = DIR_NORTH if color == COLOR_WHITE else DIR_SOUTH
    p_promo_rank = 6 if color == COLOR_WHITE else 1
    p_start_rank = 1 if color == COLOR_WHITE else 6

    p_iter = p_bb
    while p_iter:
        from_sq = get_lsb(p_iter)
        p_iter = clear_lsb(p_iter)
        rank = from_sq >> 3
        is_promo = rank == p_promo_rank

        # Single push
        to_1 = from_sq + p_step
        if 0 <= to_1 < 64 and board[to_1] == 0:
            to_mask = np.uint64(1) << np.uint64(to_1)
            if to_mask & target_mask:
                if is_promo:
                    move_list[count] = pack_move_promo(from_sq, to_1, PROMO_QUEEN)
                    move_list[count + 1] = pack_move_promo(from_sq, to_1, PROMO_KNIGHT)
                    move_list[count + 2] = pack_move_promo(from_sq, to_1, PROMO_ROOK)
                    move_list[count + 3] = pack_move_promo(from_sq, to_1, PROMO_BISHOP)
                    count += 4
                elif gen_type != 1:
                    move_list[count] = pack_move(from_sq, to_1, MOVE_NORMAL)
                    count += 1

            # Double push
            if rank == p_start_rank and gen_type != 1:
                to_2 = from_sq + p_step * 2
                if board[to_2] == 0 and ((np.uint64(1) << np.uint64(to_2)) & target_mask):
                    move_list[count] = pack_move(from_sq, to_2, MOVE_NORMAL)
                    count += 1

        # Captures
        p_att = PAWN_ATTACKS[color, from_sq] & them & target_mask
        while p_att:
            to_sq = get_lsb(p_att)
            p_att = clear_lsb(p_att)
            if is_promo:
                move_list[count] = pack_move_promo(from_sq, to_sq, PROMO_QUEEN)
                move_list[count + 1] = pack_move_promo(from_sq, to_sq, PROMO_KNIGHT)
                move_list[count + 2] = pack_move_promo(from_sq, to_sq, PROMO_ROOK)
                move_list[count + 3] = pack_move_promo(from_sq, to_sq, PROMO_BISHOP)
                count += 4
            else:
                move_list[count] = pack_move(from_sq, to_sq, MOVE_NORMAL)
                count += 1

        # En passant
        if (
                ep_square != SQUARE_NONE
                and gen_type != 2
                and ((np.uint64(1) << np.uint64(ep_square)) & PAWN_ATTACKS[color, from_sq])
        ):
            move_list[count] = pack_move(from_sq, ep_square, MOVE_EN_PASSANT)
            count += 1

    # 3. Knights
    n_bb = pieces_bb[PT_KNIGHT] & us
    while n_bb:
        from_sq = get_lsb(n_bb)
        n_bb = clear_lsb(n_bb)
        moves = KNIGHT_ATTACKS[from_sq] & target_mask
        while moves:
            to_sq = get_lsb(moves)
            moves = clear_lsb(moves)
            move_list[count] = pack_move(from_sq, to_sq, MOVE_NORMAL)
            count += 1

    # 4. Bishops & Queens
    b_bb = (pieces_bb[PT_BISHOP] | pieces_bb[PT_QUEEN]) & us
    while b_bb:
        from_sq = get_lsb(b_bb)
        b_bb = clear_lsb(b_bb)
        moves = get_bishop_attacks(from_sq, occ) & target_mask
        while moves:
            to_sq = get_lsb(moves)
            moves = clear_lsb(moves)
            move_list[count] = pack_move(from_sq, to_sq, MOVE_NORMAL)
            count += 1

    # 5. Rooks & Queens
    r_bb = (pieces_bb[PT_ROOK] | pieces_bb[PT_QUEEN]) & us
    while r_bb:
        from_sq = get_lsb(r_bb)
        r_bb = clear_lsb(r_bb)
        moves = get_rook_attacks(from_sq, occ) & target_mask
        while moves:
            to_sq = get_lsb(moves)
            moves = clear_lsb(moves)
            move_list[count] = pack_move(from_sq, to_sq, MOVE_NORMAL)
            count += 1

    return count - offset


@njit(fastmath=True, nogil=True)
def make_move_fast(
        src_ply: int,
        dst_ply: int,
        move: int,
        board_stack: np.ndarray,
        colors_stack: np.ndarray,
        pieces_stack: np.ndarray,
        material_stack: np.ndarray,
        castling_stack: np.ndarray,
        ep_stack: np.ndarray,
        color_stack: np.ndarray,
        halfmoves_stack: np.ndarray,
        zobrist_stack: np.ndarray,
        pawn_stack: np.ndarray,
        non_pawn_stack: np.ndarray,
        phase_stack: np.ndarray,
        white_acc_stack: np.ndarray,
        black_acc_stack: np.ndarray,
        feature_weights: np.ndarray,
) -> None:
    for i in range(64):
        board_stack[dst_ply, i] = board_stack[src_ply, i]
    for i in range(2):
        colors_stack[dst_ply, i] = colors_stack[src_ply, i]
    for i in range(7):
        pieces_stack[dst_ply, i] = pieces_stack[src_ply, i]
    for i in range(10):
        material_stack[dst_ply, i] = material_stack[src_ply, i]
    for c in range(2):
        for s in range(2):
            castling_stack[dst_ply, c, s] = castling_stack[src_ply, c, s]
    ep_stack[dst_ply] = SQUARE_NONE
    color_stack[dst_ply] = color_stack[src_ply]
    halfmoves_stack[dst_ply] = halfmoves_stack[src_ply] + 1
    zobrist_stack[dst_ply] = zobrist_stack[src_ply]
    pawn_stack[dst_ply] = pawn_stack[src_ply]
    non_pawn_stack[dst_ply, 0] = non_pawn_stack[src_ply, 0]
    non_pawn_stack[dst_ply, 1] = non_pawn_stack[src_ply, 1]
    phase = phase_stack[src_ply]
    phase_stack[dst_ply] = phase

    if move == MOVE_NONE:
        color_stack[dst_ply] ^= 1
        if ep_stack[src_ply] != SQUARE_NONE:
            zobrist_stack[dst_ply] ^= ZOBRIST_KEYS[EP_INDEX]
        zobrist_stack[dst_ply] ^= ZOBRIST_KEYS[SIDE_INDEX]
        for i in range(1024):
            white_acc_stack[dst_ply, i] = white_acc_stack[src_ply, i]
            black_acc_stack[dst_ply, i] = black_acc_stack[src_ply, i]
        return

    from_sq = extract_from(move)
    to_sq = extract_to(move)
    move_type = extract_type(move)
    color = int(color_stack[src_ply])
    opp_color = color ^ 1

    from_piece = int(board_stack[src_ply, from_sq])
    from_type = get_piece_type(from_piece)
    to_piece = from_piece

    captured_piece = int(board_stack[src_ply, to_sq])
    captured_sq = to_sq
    new_ep = SQUARE_NONE
    base_rank = 56 if color == COLOR_BLACK else 0

    if move_type == MOVE_CASTLING:
        king_pos = get_king_sq(colors_stack[src_ply], pieces_stack[src_ply], color)
        side = SIDE_KINGSIDE if to_sq > king_pos else SIDE_QUEENSIDE
        to_sq = base_rank + 6 if side == SIDE_KINGSIDE else base_rank + 2
        rook_from = int(castling_stack[src_ply, color, side])
        rook_to = base_rank + 5 if side == SIDE_KINGSIDE else base_rank + 3
        rook_piece = PIECE_W_ROOK + color

        board_stack[dst_ply, rook_from] = PIECE_BLANK
        board_stack[dst_ply, rook_to] = rook_piece
        rook_mask = (np.uint64(1) << np.uint64(rook_from)) | (np.uint64(1) << np.uint64(rook_to))
        colors_stack[dst_ply, color] ^= rook_mask
        pieces_stack[dst_ply, PT_ROOK] ^= rook_mask

        r_key_from = ZOBRIST_KEYS[get_zobrist_piece_key(rook_piece, rook_from)]
        r_key_to = ZOBRIST_KEYS[get_zobrist_piece_key(rook_piece, rook_to)]
        zobrist_stack[dst_ply] ^= r_key_from ^ r_key_to
        non_pawn_stack[dst_ply, color] ^= r_key_from ^ r_key_to

        net_idx = phase_to_index(phase)
        f_w = feature_weights[net_idx]
        wf1, bf1 = feature_indices(from_piece, from_sq)
        wt1, bt1 = feature_indices(from_piece, to_sq)
        wf2, bf2 = feature_indices(rook_piece, rook_from)
        wt2, bt2 = feature_indices(rook_piece, rook_to)

        nnue_update_2add_2sub(white_acc_stack[dst_ply], white_acc_stack[src_ply], f_w, wt1, wt2, wf1, wf2)
        nnue_update_2add_2sub(black_acc_stack[dst_ply], black_acc_stack[src_ply], f_w, bt1, bt2, bf1, bf2)

    elif captured_piece != PIECE_BLANK:
        halfmoves_stack[dst_ply] = 0
        c_type = get_piece_type(captured_piece)
        if c_type != PT_KING:
            material_stack[dst_ply, captured_piece - 2] -= 1
        c_mask = np.uint64(1) << np.uint64(captured_sq)
        colors_stack[dst_ply, opp_color] ^= c_mask
        pieces_stack[dst_ply, c_type] ^= c_mask

        c_key = ZOBRIST_KEYS[get_zobrist_piece_key(captured_piece, captured_sq)]
        zobrist_stack[dst_ply] ^= c_key
        if c_type == PT_PAWN:
            pawn_stack[dst_ply] ^= c_key
        else:
            non_pawn_stack[dst_ply, opp_color] ^= c_key

    elif move_type == MOVE_EN_PASSANT:
        halfmoves_stack[dst_ply] = 0
        captured_sq = to_sq + (DIR_SOUTH if color == COLOR_WHITE else DIR_NORTH)
        captured_piece = int(board_stack[src_ply, captured_sq])
        board_stack[dst_ply, captured_sq] = PIECE_BLANK
        material_stack[dst_ply, captured_piece - 2] -= 1

        c_mask = np.uint64(1) << np.uint64(captured_sq)
        colors_stack[dst_ply, opp_color] ^= c_mask
        pieces_stack[dst_ply, PT_PAWN] ^= c_mask

        c_key = ZOBRIST_KEYS[get_zobrist_piece_key(captured_piece, captured_sq)]
        zobrist_stack[dst_ply] ^= c_key
        pawn_stack[dst_ply] ^= c_key

    board_stack[dst_ply, from_sq] = PIECE_BLANK
    board_stack[dst_ply, to_sq] = to_piece

    if from_type == PT_PAWN:
        halfmoves_stack[dst_ply] = 0
        if move_type == MOVE_PROMOTION:
            promo = extract_promo(move)
            promo_pt = PT_KNIGHT + promo
            to_piece = promo_pt * 2 + color
            board_stack[dst_ply, to_sq] = to_piece

            material_stack[dst_ply, from_piece - 2] -= 1
            material_stack[dst_ply, to_piece - 2] += 1
            pieces_stack[dst_ply, PT_PAWN] ^= np.uint64(1) << np.uint64(to_sq)
            pieces_stack[dst_ply, promo_pt] |= np.uint64(1) << np.uint64(to_sq)
        elif abs(to_sq - from_sq) == 16:
            new_ep = (to_sq + from_sq) // 2

    elif from_type == PT_KING:
        for side in range(2):
            if castling_stack[src_ply, color, side] != SQUARE_NONE:
                zobrist_stack[dst_ply] ^= ZOBRIST_KEYS[CASTLING_INDEX + color * 2 + side]
                castling_stack[dst_ply, color, side] = SQUARE_NONE

    for c in range(2):
        for side in range(2):
            sq_c = castling_stack[src_ply, c, side]
            if sq_c != SQUARE_NONE and (from_sq == sq_c or to_sq == sq_c):
                zobrist_stack[dst_ply] ^= ZOBRIST_KEYS[CASTLING_INDEX + c * 2 + side]
                castling_stack[dst_ply, c, side] = SQUARE_NONE

    from_mask = np.uint64(1) << np.uint64(from_sq)
    to_mask = np.uint64(1) << np.uint64(to_sq)
    colors_stack[dst_ply, color] ^= from_mask | to_mask

    if move_type != MOVE_PROMOTION:
        pieces_stack[dst_ply, from_type] ^= from_mask | to_mask
    else:
        pieces_stack[dst_ply, from_type] ^= from_mask

    f_key = ZOBRIST_KEYS[get_zobrist_piece_key(from_piece, from_sq)]
    t_key = ZOBRIST_KEYS[get_zobrist_piece_key(to_piece, to_sq)]
    zobrist_stack[dst_ply] ^= f_key ^ t_key

    if from_type == PT_PAWN:
        pawn_stack[dst_ply] ^= f_key
        if move_type != MOVE_PROMOTION:
            pawn_stack[dst_ply] ^= t_key
        else:
            non_pawn_stack[dst_ply, color] ^= t_key
    else:
        non_pawn_stack[dst_ply, color] ^= f_key ^ t_key

    color_stack[dst_ply] ^= 1
    zobrist_stack[dst_ply] ^= ZOBRIST_KEYS[SIDE_INDEX]

    if (ep_stack[src_ply] == SQUARE_NONE) ^ (new_ep == SQUARE_NONE):
        zobrist_stack[dst_ply] ^= ZOBRIST_KEYS[EP_INDEX]
    ep_stack[dst_ply] = new_ep

    if move_type != MOVE_CASTLING:
        net_idx = phase_to_index(phase)
        f_w = feature_weights[net_idx]
        w_from, b_from = feature_indices(from_piece, from_sq)
        w_to, b_to = feature_indices(to_piece, to_sq)

        if captured_piece != PIECE_BLANK:
            w_cap, b_cap = feature_indices(captured_piece, captured_sq)
            nnue_update_1add_2sub(white_acc_stack[dst_ply], white_acc_stack[src_ply], f_w, w_to, w_from, w_cap)
            nnue_update_1add_2sub(black_acc_stack[dst_ply], black_acc_stack[src_ply], f_w, b_to, b_from, b_cap)
        else:
            nnue_update_1add_1sub(white_acc_stack[dst_ply], white_acc_stack[src_ply], f_w, w_to, w_from)
            nnue_update_1add_1sub(black_acc_stack[dst_ply], black_acc_stack[src_ply], f_w, b_to, b_from)


@njit(fastmath=True, nogil=True)
def qsearch_fast(
        alpha: int,
        beta: int,
        ply: int,
        board_stack: np.ndarray,
        colors_stack: np.ndarray,
        pieces_stack: np.ndarray,
        material_stack: np.ndarray,
        castling_stack: np.ndarray,
        ep_stack: np.ndarray,
        color_stack: np.ndarray,
        halfmoves_stack: np.ndarray,
        zobrist_stack: np.ndarray,
        pawn_stack: np.ndarray,
        non_pawn_stack: np.ndarray,
        phase_stack: np.ndarray,
        white_acc_stack: np.ndarray,
        black_acc_stack: np.ndarray,
        move_stack: np.ndarray,
        score_stack: np.ndarray,
        feature_weights: np.ndarray,
        output_weights: np.ndarray,
        output_biases: np.ndarray,
        pawn_corr_hist: np.ndarray,
        non_pawn_corr_hist: np.ndarray,
        nodes_count: np.ndarray,
        stop_flag: np.ndarray,
        node_limit: int,
) -> int:
    if ply >= MAX_SEARCH_DEPTH - 1 or stop_flag[0] != 0:
        return eval_position_fast(
            ply, color_stack, phase_stack, material_stack, pawn_stack, non_pawn_stack,
            white_acc_stack, black_acc_stack, output_weights, output_biases,
            pawn_corr_hist, non_pawn_corr_hist,
        )

    nodes_count[0] += 1
    if nodes_count[0] >= node_limit:
        stop_flag[0] = 1
        return eval_position_fast(
            ply, color_stack, phase_stack, material_stack, pawn_stack, non_pawn_stack,
            white_acc_stack, black_acc_stack, output_weights, output_biases,
            pawn_corr_hist, non_pawn_corr_hist,
        )

    color = int(color_stack[ply])
    in_chk = is_in_check_fast(colors_stack[ply], pieces_stack[ply], color)

    stand_pat = SCORE_NONE
    if not in_chk:
        stand_pat = eval_position_fast(
            ply, color_stack, phase_stack, material_stack, pawn_stack, non_pawn_stack,
            white_acc_stack, black_acc_stack, output_weights, output_biases,
            pawn_corr_hist, non_pawn_corr_hist,
        )
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

    # Generate captures (or all evasions if in check)
    gen_mode = 0 if in_chk else 1
    n_moves = generate_moves_fast(
        board_stack[ply], colors_stack[ply], pieces_stack[ply],
        castling_stack[ply], int(ep_stack[ply]), color,
        move_stack[ply], 0, gen_mode,
    )

    # Score captures with MVV-LVA
    for i in range(n_moves):
        m = int(move_stack[ply, i])
        f = extract_from(m)
        t = extract_to(m)
        f_piece = int(board_stack[ply, f])
        t_piece = int(board_stack[ply, t])
        att_val = get_see_value(get_piece_type(f_piece))
        vic_val = get_see_value(get_piece_type(t_piece)) if t_piece != 0 else att_val
        score_stack[ply, i] = vic_val * 100 - att_val

    best_score = stand_pat if not in_chk else SCORE_NONE

    for i in range(n_moves):
        best_i = i
        for j in range(i + 1, n_moves):
            if score_stack[ply, j] > score_stack[ply, best_i]:
                best_i = j
        move_stack[ply, i], move_stack[ply, best_i] = move_stack[ply, best_i], move_stack[ply, i]
        score_stack[ply, i], score_stack[ply, best_i] = score_stack[ply, best_i], score_stack[ply, i]

        m = int(move_stack[ply, i])
        is_promo = extract_type(m) == MOVE_PROMOTION

        # QSearch Delta Pruning (only when not in check):
        # If stand_pat + biggest possible gain (queen value ~950) is still well below alpha, prune
        if not in_chk and not is_promo and stand_pat < alpha - 950:
            continue

        # QSearch SEE Pruning: prune losing captures unless in check
        if not in_chk and not see_fast(board_stack[ply], colors_stack[ply], pieces_stack[ply], m, 0):
            continue

        next_ply = ply + 1
        make_move_fast(
            ply, next_ply, m,
            board_stack, colors_stack, pieces_stack, material_stack, castling_stack,
            ep_stack, color_stack, halfmoves_stack, zobrist_stack, pawn_stack,
            non_pawn_stack, phase_stack, white_acc_stack, black_acc_stack, feature_weights,
        )

        if is_in_check_fast(colors_stack[next_ply], pieces_stack[next_ply], color):
            continue

        score = -qsearch_fast(
            -beta, -alpha, next_ply,
            board_stack, colors_stack, pieces_stack, material_stack, castling_stack,
            ep_stack, color_stack, halfmoves_stack, zobrist_stack, pawn_stack,
            non_pawn_stack, phase_stack, white_acc_stack, black_acc_stack,
            move_stack, score_stack, feature_weights, output_weights, output_biases,
            pawn_corr_hist, non_pawn_corr_hist, nodes_count, stop_flag, node_limit,
        )

        if stop_flag[0] != 0:
            return best_score

        if score > best_score:
            best_score = score
            if score > alpha:
                alpha = score
            if score >= beta:
                break

    return best_score if best_score != SCORE_NONE else (ply - SCORE_MATE if in_chk else 0)


@njit(fastmath=True, nogil=True)
def pvs_search_fast(
        alpha: int,
        beta: int,
        depth: int,
        ply: int,
        is_pv: bool,
        cutnode: bool,
        excluded_move: int,
        # Stack buffers
        board_stack: np.ndarray,
        colors_stack: np.ndarray,
        pieces_stack: np.ndarray,
        material_stack: np.ndarray,
        castling_stack: np.ndarray,
        ep_stack: np.ndarray,
        color_stack: np.ndarray,
        halfmoves_stack: np.ndarray,
        zobrist_stack: np.ndarray,
        pawn_stack: np.ndarray,
        non_pawn_stack: np.ndarray,
        phase_stack: np.ndarray,
        eval_stack: np.ndarray,
        white_acc_stack: np.ndarray,
        black_acc_stack: np.ndarray,
        move_stack: np.ndarray,
        score_stack: np.ndarray,
        pv_table: np.ndarray,
        pv_length: np.ndarray,
        prev_moves: np.ndarray,
        killers: np.ndarray,
        main_history: np.ndarray,
        cap_history: np.ndarray,
        cont_history_1: np.ndarray,
        cont_history_2: np.ndarray,
        counter_moves: np.ndarray,
        pawn_corr_hist: np.ndarray,
        non_pawn_corr_hist: np.ndarray,
        feature_weights: np.ndarray,
        output_weights: np.ndarray,
        output_biases: np.ndarray,
        tt_keys: np.ndarray,
        tt_scores: np.ndarray,
        tt_static_evals: np.ndarray,
        tt_moves: np.ndarray,
        tt_depths: np.ndarray,
        tt_bounds: np.ndarray,
        tt_ages: np.ndarray,
        tt_mask: int,
        current_age: int,
        game_history: np.ndarray,
        game_history_len: int,
        nodes_count: np.ndarray,
        stop_flag: np.ndarray,
        node_limit: int,
) -> int:
    if ply >= MAX_SEARCH_DEPTH - 1 or stop_flag[0] != 0:
        return eval_position_fast(
            ply, color_stack, phase_stack, material_stack, pawn_stack, non_pawn_stack,
            white_acc_stack, black_acc_stack, output_weights, output_biases,
            pawn_corr_hist, non_pawn_corr_hist,
        )

    pv_length[ply] = ply
    if depth <= 0:
        return qsearch_fast(
            alpha, beta, ply,
            board_stack, colors_stack, pieces_stack, material_stack, castling_stack,
            ep_stack, color_stack, halfmoves_stack, zobrist_stack, pawn_stack,
            non_pawn_stack, phase_stack, white_acc_stack, black_acc_stack,
            move_stack, score_stack, feature_weights, output_weights, output_biases,
            pawn_corr_hist, non_pawn_corr_hist, nodes_count, stop_flag, node_limit,
        )

    nodes_count[0] += 1
    if nodes_count[0] >= node_limit:
        stop_flag[0] = 1
        return eval_position_fast(
            ply, color_stack, phase_stack, material_stack, pawn_stack, non_pawn_stack,
            white_acc_stack, black_acc_stack, output_weights, output_biases,
            pawn_corr_hist, non_pawn_corr_hist,
        )

    # 1. Repetition & 50-Move Rule Check
    if halfmoves_stack[ply] >= 100:
        return 0

    if ply > 0:
        cur_key = zobrist_stack[ply]
        # In-tree repetition
        for prev_ply in range(ply - 2, -1, -2):
            if zobrist_stack[prev_ply] == cur_key:
                return 0
            if halfmoves_stack[prev_ply] == 0:
                break
        # Cross-move game history repetition
        if halfmoves_stack[0] > 0 and game_history_len > 0:
            h_limit = min(game_history_len, int(halfmoves_stack[0]))
            for h_idx in range(1, h_limit + 1, 2):
                if game_history[game_history_len - h_idx] == cur_key:
                    return 0

    color = int(color_stack[ply])
    in_chk = is_in_check_fast(colors_stack[ply], pieces_stack[ply], color)

    # Check extension
    if in_chk:
        depth = max(1, depth + 1)

    # 2. TT Probe
    z_key = zobrist_stack[ply]
    tt_idx = int(z_key & np.uint64(tt_mask))
    key_32 = np.uint32(z_key >> np.uint64(32))

    tt_hit = (tt_keys[tt_idx] == key_32) and (excluded_move == MOVE_NONE)
    raw_tt_score = int(tt_scores[tt_idx]) if tt_hit else SCORE_NONE
    tt_static_eval = int(tt_static_evals[tt_idx]) if tt_hit else SCORE_NONE
    tt_move = int(tt_moves[tt_idx]) if tt_hit else MOVE_NONE
    tt_depth = int(tt_depths[tt_idx]) if tt_hit else 0
    tt_bound = int(tt_bounds[tt_idx]) if tt_hit else ENTRY_NONE

    # Mate score ply adjustment on TT hit
    tt_score = raw_tt_score
    if tt_hit and raw_tt_score != SCORE_NONE:
        if raw_tt_score >= SCORE_WIN:
            tt_score = raw_tt_score - ply
        elif raw_tt_score <= SCORE_LOST:
            tt_score = raw_tt_score + ply

    if (
            tt_hit
            and not is_pv
            and tt_depth >= depth
            and (
            tt_bound == ENTRY_EXACT
            or (tt_bound == ENTRY_LBOUND and tt_score >= beta)
            or (tt_bound == ENTRY_UBOUND and tt_score <= alpha)
    )
    ):
        return tt_score

    # 3. Static Evaluation & Improving Detection
    static_eval = (
        tt_static_eval
        if (tt_hit and tt_static_eval != SCORE_NONE)
        else eval_position_fast(
            ply, color_stack, phase_stack, material_stack, pawn_stack, non_pawn_stack,
            white_acc_stack, black_acc_stack, output_weights, output_biases,
            pawn_corr_hist, non_pawn_corr_hist,
        )
    )
    eval_stack[ply] = np.int16(static_eval)
    improving = False
    if not in_chk and ply >= 2:
        improving = static_eval >= int(eval_stack[ply - 2])

    # 4. Reverse Futility Pruning (RFP) with Improving
    rfp_margin = RFP_MARGIN * depth if improving else (RFP_MARGIN - 20) * depth
    if (
            not is_pv
            and not in_chk
            and depth <= RFP_MAX_DEPTH
            and excluded_move == MOVE_NONE
            and static_eval - rfp_margin >= beta
    ):
        return (static_eval + beta) // 2

    # 5. Null Move Pruning (NMP) with Improving
    if (
            not is_pv
            and not in_chk
            and depth >= NMP_MIN_DEPTH
            and static_eval >= beta
            and material_stack[ply, 2 + color] > 0
            and excluded_move == MOVE_NONE
    ):
        r = NMP_BASE + depth // NMP_DEPTH_DIV + min(3, max(0, (static_eval - beta) // NMP_EVAL_DIV))
        if not improving:
            r += 1
        next_ply = ply + 1
        make_move_fast(
            ply, next_ply, MOVE_NONE,
            board_stack, colors_stack, pieces_stack, material_stack, castling_stack,
            ep_stack, color_stack, halfmoves_stack, zobrist_stack, pawn_stack,
            non_pawn_stack, phase_stack, white_acc_stack, black_acc_stack, feature_weights,
        )

        null_score = -pvs_search_fast(
            -beta, -beta + 1, depth - r, next_ply, False, not cutnode, MOVE_NONE,
            board_stack, colors_stack, pieces_stack, material_stack, castling_stack,
            ep_stack, color_stack, halfmoves_stack, zobrist_stack, pawn_stack,
            non_pawn_stack, phase_stack, eval_stack, white_acc_stack, black_acc_stack,
            move_stack, score_stack, pv_table, pv_length, prev_moves, killers,
            main_history, cap_history, cont_history_1, cont_history_2, counter_moves,
            pawn_corr_hist, non_pawn_corr_hist,
            feature_weights, output_weights, output_biases,
            tt_keys, tt_scores, tt_static_evals, tt_moves, tt_depths, tt_bounds, tt_ages,
            tt_mask, current_age, game_history, game_history_len, nodes_count, stop_flag, node_limit,
        )
        if null_score >= beta:
            return beta if null_score >= SCORE_WIN else null_score

    # 6. ProbCut (Probability Cutoff on Captures)
    if (
            depth >= 5
            and not is_pv
            and not in_chk
            and abs(beta) < SCORE_WIN
            and excluded_move == MOVE_NONE
            and static_eval >= beta + PROBCUT_MARGIN - 150
    ):
        probcut_beta = beta + PROBCUT_MARGIN
        n_prob_moves = generate_moves_fast(
            board_stack[ply], colors_stack[ply], pieces_stack[ply],
            castling_stack[ply], int(ep_stack[ply]), color,
            move_stack[ply], 0, 1,  # Captures only
        )
        for p_idx in range(n_prob_moves):
            pm = int(move_stack[ply, p_idx])
            if not see_fast(board_stack[ply], colors_stack[ply], pieces_stack[ply], pm, 0):
                continue
            next_ply = ply + 1
            make_move_fast(
                ply, next_ply, pm,
                board_stack, colors_stack, pieces_stack, material_stack, castling_stack,
                ep_stack, color_stack, halfmoves_stack, zobrist_stack, pawn_stack,
                non_pawn_stack, phase_stack, white_acc_stack, black_acc_stack, feature_weights,
            )
            if is_in_check_fast(colors_stack[next_ply], pieces_stack[next_ply], color):
                continue
            p_score = -pvs_search_fast(
                -probcut_beta, -probcut_beta + 1, depth - 4, next_ply, False, True, MOVE_NONE,
                board_stack, colors_stack, pieces_stack, material_stack, castling_stack,
                ep_stack, color_stack, halfmoves_stack, zobrist_stack, pawn_stack,
                non_pawn_stack, phase_stack, eval_stack, white_acc_stack, black_acc_stack,
                move_stack, score_stack, pv_table, pv_length, prev_moves, killers,
                main_history, cap_history, cont_history_1, cont_history_2, counter_moves,
                pawn_corr_hist, non_pawn_corr_hist,
                feature_weights, output_weights, output_biases,
                tt_keys, tt_scores, tt_static_evals, tt_moves, tt_depths, tt_bounds, tt_ages,
                tt_mask, current_age, game_history, game_history_len, nodes_count, stop_flag, node_limit,
            )
            if p_score >= probcut_beta:
                return beta

    # Internal Iterative Reductions (IIR)
    if depth >= IIR_MIN_DEPTH and tt_move == MOVE_NONE and not in_chk:
        depth -= 1

    # 7. Generate & Score Moves
    n_moves = generate_moves_fast(
        board_stack[ply], colors_stack[ply], pieces_stack[ply],
        castling_stack[ply], int(ep_stack[ply]), color,
        move_stack[ply], 0, 0,
    )

    k1 = int(killers[ply, 0])
    k2 = int(killers[ply, 1])
    cm = int(counter_moves[color, extract_from(tt_move), extract_to(tt_move)]) if tt_move != MOVE_NONE else MOVE_NONE

    prev_m = int(prev_moves[ply - 1]) if ply >= 1 else MOVE_NONE
    prev_piece = int(board_stack[ply - 1, extract_from(prev_m)]) if prev_m != MOVE_NONE else 0
    prev_to = extract_to(prev_m) if prev_m != MOVE_NONE else 0

    prev2_m = int(prev_moves[ply - 2]) if ply >= 2 else MOVE_NONE
    prev2_piece = int(board_stack[ply - 2, extract_from(prev2_m)]) if prev2_m != MOVE_NONE else 0
    prev2_to = extract_to(prev2_m) if prev2_m != MOVE_NONE else 0

    for i in range(n_moves):
        m = int(move_stack[ply, i])
        f = extract_from(m)
        t = extract_to(m)
        f_piece = int(board_stack[ply, f])
        t_piece = int(board_stack[ply, t])
        is_capture = (t_piece != 0) or (extract_type(m) == MOVE_EN_PASSANT)

        if m == tt_move:
            score_stack[ply, i] = 1000000000
        elif extract_type(m) == MOVE_PROMOTION and extract_promo(m) == PROMO_QUEEN:
            score_stack[ply, i] = 900000000
        elif is_capture:
            att_val = get_see_value(get_piece_type(f_piece))
            vic_val = get_see_value(get_piece_type(t_piece)) if t_piece != 0 else att_val
            cap_h = int(cap_history[f_piece, t])
            if see_fast(board_stack[ply], colors_stack[ply], pieces_stack[ply], m, 0):
                score_stack[ply, i] = 800000000 + vic_val * 100 - att_val + cap_h
            else:
                score_stack[ply, i] = -100000000 + vic_val * 100 - att_val + cap_h
        elif m == k1:
            score_stack[ply, i] = 700000000
        elif m == k2:
            score_stack[ply, i] = 690000000
        elif m == cm:
            score_stack[ply, i] = 680000000
        else:
            h = int(main_history[color, f, t])
            if prev_piece != 0:
                h += int(cont_history_1[prev_piece, prev_to, f_piece, t])
            if prev2_piece != 0:
                h += int(cont_history_2[prev2_piece, prev2_to, f_piece, t])
            score_stack[ply, i] = h

    best_score = SCORE_NONE
    best_move = MOVE_NONE
    moves_played = 0
    quiets_played = 0
    searched_quiets = np.zeros(64, dtype=np.uint16)
    num_searched_quiets = 0

    for i in range(n_moves):
        best_i = i
        for j in range(i + 1, n_moves):
            if score_stack[ply, j] > score_stack[ply, best_i]:
                best_i = j
        move_stack[ply, i], move_stack[ply, best_i] = move_stack[ply, best_i], move_stack[ply, i]
        score_stack[ply, i], score_stack[ply, best_i] = score_stack[ply, best_i], score_stack[ply, i]

        m = int(move_stack[ply, i])
        if m == excluded_move:
            continue

        is_quiet = (extract_type(m) == MOVE_NORMAL) and (board_stack[ply, extract_to(m)] == 0)

        # Futility Pruning (FP) with Improving
        fp_margin = FP_MARGIN * depth if improving else (FP_MARGIN - 20) * depth
        if (
                depth <= FP_DEPTH
                and not is_pv
                and not in_chk
                and is_quiet
                and moves_played > 0
                and static_eval + fp_margin <= alpha
        ):
            continue

        # Late Move Pruning (LMP) with Improving
        lmp_limit = (LMP_BASE + depth * depth) if improving else (LMP_BASE + depth * depth) // 2
        if (
                depth <= LMP_DEPTH
                and not is_pv
                and not in_chk
                and is_quiet
                and quiets_played >= lmp_limit
        ):
            continue

        # SEE Pruning in PVS
        if (
                depth <= SEE_PRUNING_DEPTH
                and not in_chk
                and is_quiet
                and not see_fast(board_stack[ply], colors_stack[ply], pieces_stack[ply], m, SEE_QUIET_MARGIN * depth)
        ):
            continue

        next_ply = ply + 1
        make_move_fast(
            ply, next_ply, m,
            board_stack, colors_stack, pieces_stack, material_stack, castling_stack,
            ep_stack, color_stack, halfmoves_stack, zobrist_stack, pawn_stack,
            non_pawn_stack, phase_stack, white_acc_stack, black_acc_stack, feature_weights,
        )

        if is_in_check_fast(colors_stack[next_ply], pieces_stack[next_ply], color):
            continue

        prev_moves[ply] = np.uint16(m)
        moves_played += 1
        if is_quiet:
            quiets_played += 1
            if num_searched_quiets < 64:
                searched_quiets[num_searched_quiets] = np.uint16(m)
                num_searched_quiets += 1

        # 8. Singular Extensions
        extension = 0
        if (
                depth >= SE_DEPTH
                and m == tt_move
                and tt_hit
                and tt_bound in (ENTRY_EXACT, ENTRY_LBOUND)
                and abs(tt_score) < SCORE_WIN
                and excluded_move == MOVE_NONE
        ):
            se_beta = tt_score - depth * 2
            se_score = pvs_search_fast(
                se_beta - 1, se_beta, (depth - 1) // 2, ply, False, cutnode, tt_move,
                board_stack, colors_stack, pieces_stack, material_stack, castling_stack,
                ep_stack, color_stack, halfmoves_stack, zobrist_stack, pawn_stack,
                non_pawn_stack, phase_stack, eval_stack, white_acc_stack, black_acc_stack,
                move_stack, score_stack, pv_table, pv_length, prev_moves, killers,
                main_history, cap_history, cont_history_1, cont_history_2, counter_moves,
                pawn_corr_hist, non_pawn_corr_hist,
                feature_weights, output_weights, output_biases,
                tt_keys, tt_scores, tt_static_evals, tt_moves, tt_depths, tt_bounds, tt_ages,
                tt_mask, current_age, game_history, game_history_len, nodes_count, stop_flag, node_limit,
            )
            if se_score < se_beta:
                if se_score < se_beta - SE_TRIPLE_EXT_MARGIN:
                    extension = 3
                elif se_score < se_beta - SE_DOUBLE_EXT_MARGIN:
                    extension = 2
                else:
                    extension = 1

        # 9. Late Move Reductions (LMR) with History & Improving
        reduction = 0
        if depth >= LMR_MIN_DEPTH and moves_played > 1 and is_quiet:
            reduction = int(LMR_TABLE[min(127, depth), min(255, moves_played)])
            # History-scaled LMR
            f_sq = extract_from(m)
            t_sq = extract_to(m)
            fp = int(board_stack[ply, f_sq])
            h_val = int(main_history[color, f_sq, t_sq])
            if prev_piece != 0:
                h_val += int(cont_history_1[prev_piece, prev_to, fp, t_sq])
            reduction -= h_val // 8192

            if not is_pv:
                reduction += 1
            if not improving:
                reduction += 1
            if cutnode:
                reduction += 1
            reduction = max(0, min(depth - 1, reduction))

        new_depth = depth - 1 + extension

        # PVS search
        if moves_played == 1:
            score = -pvs_search_fast(
                -beta, -alpha, new_depth, next_ply, is_pv, False, MOVE_NONE,
                board_stack, colors_stack, pieces_stack, material_stack, castling_stack,
                ep_stack, color_stack, halfmoves_stack, zobrist_stack, pawn_stack,
                non_pawn_stack, phase_stack, eval_stack, white_acc_stack, black_acc_stack,
                move_stack, score_stack, pv_table, pv_length, prev_moves, killers,
                main_history, cap_history, cont_history_1, cont_history_2, counter_moves,
                pawn_corr_hist, non_pawn_corr_hist,
                feature_weights, output_weights, output_biases,
                tt_keys, tt_scores, tt_static_evals, tt_moves, tt_depths, tt_bounds, tt_ages,
                tt_mask, current_age, game_history, game_history_len, nodes_count, stop_flag, node_limit,
            )
        else:
            score = -pvs_search_fast(
                -alpha - 1, -alpha, new_depth - reduction, next_ply, False, True, MOVE_NONE,
                board_stack, colors_stack, pieces_stack, material_stack, castling_stack,
                ep_stack, color_stack, halfmoves_stack, zobrist_stack, pawn_stack,
                non_pawn_stack, phase_stack, eval_stack, white_acc_stack, black_acc_stack,
                move_stack, score_stack, pv_table, pv_length, prev_moves, killers,
                main_history, cap_history, cont_history_1, cont_history_2, counter_moves,
                pawn_corr_hist, non_pawn_corr_hist,
                feature_weights, output_weights, output_biases,
                tt_keys, tt_scores, tt_static_evals, tt_moves, tt_depths, tt_bounds, tt_ages,
                tt_mask, current_age, game_history, game_history_len, nodes_count, stop_flag, node_limit,
            )
            if score > alpha and reduction > 0:
                score = -pvs_search_fast(
                    -alpha - 1, -alpha, new_depth, next_ply, False, True, MOVE_NONE,
                    board_stack, colors_stack, pieces_stack, material_stack, castling_stack,
                    ep_stack, color_stack, halfmoves_stack, zobrist_stack, pawn_stack,
                    non_pawn_stack, phase_stack, eval_stack, white_acc_stack, black_acc_stack,
                    move_stack, score_stack, pv_table, pv_length, prev_moves, killers,
                    main_history, cap_history, cont_history_1, cont_history_2, counter_moves,
                    pawn_corr_hist, non_pawn_corr_hist,
                    feature_weights, output_weights, output_biases,
                    tt_keys, tt_scores, tt_static_evals, tt_moves, tt_depths, tt_bounds, tt_ages,
                    tt_mask, current_age, game_history, game_history_len, nodes_count, stop_flag, node_limit,
                )
            if alpha < score < beta:
                score = -pvs_search_fast(
                    -beta, -alpha, new_depth, next_ply, True, False, MOVE_NONE,
                    board_stack, colors_stack, pieces_stack, material_stack, castling_stack,
                    ep_stack, color_stack, halfmoves_stack, zobrist_stack, pawn_stack,
                    non_pawn_stack, phase_stack, eval_stack, white_acc_stack, black_acc_stack,
                    move_stack, score_stack, pv_table, pv_length, prev_moves, killers,
                    main_history, cap_history, cont_history_1, cont_history_2, counter_moves,
                    pawn_corr_hist, non_pawn_corr_hist,
                    feature_weights, output_weights, output_biases,
                    tt_keys, tt_scores, tt_static_evals, tt_moves, tt_depths, tt_bounds, tt_ages,
                    tt_mask, current_age, game_history, game_history_len, nodes_count, stop_flag, node_limit,
                )

        if stop_flag[0] != 0:
            return best_score

        if score > best_score:
            best_score = score
            best_move = m

            if score > alpha:
                alpha = score
                pv_table[ply, ply] = np.uint16(m)
                for j in range(ply + 1, pv_length[ply + 1]):
                    pv_table[ply, j] = pv_table[ply + 1, j]
                pv_length[ply] = pv_length[ply + 1]

                if score >= beta:
                    if is_quiet:
                        if killers[ply, 0] != m:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = np.uint16(m)
                        f = extract_from(m)
                        t = extract_to(m)
                        f_piece = int(board_stack[ply, f])
                        bonus = min(HIST_MAX, HIST_BONUS * depth)

                        # Gravity-damped Main History update
                        cur_h = int(main_history[color, f, t])
                        main_history[color, f, t] = np.int16(cur_h + bonus - (cur_h * abs(bonus)) // 16384)

                        # 1-Ply Continuation History update
                        if prev_piece != 0:
                            cur_c1 = int(cont_history_1[prev_piece, prev_to, f_piece, t])
                            cont_history_1[prev_piece, prev_to, f_piece, t] = np.int16(
                                cur_c1 + bonus - (cur_c1 * abs(bonus)) // 16384
                            )

                        # 2-Ply Continuation History update
                        if prev2_piece != 0:
                            cur_c2 = int(cont_history_2[prev2_piece, prev2_to, f_piece, t])
                            cont_history_2[prev2_piece, prev2_to, f_piece, t] = np.int16(
                                cur_c2 + bonus - (cur_c2 * abs(bonus)) // 16384
                            )

                        counter_moves[color, f, t] = np.uint16(m)

                        # History Malus on previously searched quiets that failed low
                        for q_idx in range(num_searched_quiets - 1):
                            prev_q = int(searched_quiets[q_idx])
                            pf = extract_from(prev_q)
                            pt = extract_to(prev_q)
                            pp = int(board_stack[ply, pf])

                            qh = int(main_history[color, pf, pt])
                            main_history[color, pf, pt] = np.int16(qh - bonus - (qh * abs(bonus)) // 16384)

                            if prev_piece != 0:
                                qc1 = int(cont_history_1[prev_piece, prev_to, pp, pt])
                                cont_history_1[prev_piece, prev_to, pp, pt] = np.int16(
                                    qc1 - bonus - (qc1 * abs(bonus)) // 16384
                                )

                            if prev2_piece != 0:
                                qc2 = int(cont_history_2[prev2_piece, prev2_to, pp, pt])
                                cont_history_2[prev2_piece, prev2_to, pp, pt] = np.int16(
                                    qc2 - bonus - (qc2 * abs(bonus)) // 16384
                                )
                    else:
                        # Capture history update on beta cutoff
                        f = extract_from(m)
                        t = extract_to(m)
                        f_piece = int(board_stack[ply, f])
                        bonus = min(HIST_MAX, HIST_BONUS * depth)
                        cur_cap = int(cap_history[f_piece, t])
                        cap_history[f_piece, t] = np.int16(cur_cap + bonus - (cur_cap * abs(bonus)) // 16384)
                    break

    if moves_played == 0:
        return ply - SCORE_MATE if in_chk else 0

    bound = ENTRY_EXACT
    if best_score >= beta:
        bound = ENTRY_LBOUND
    elif best_score <= alpha:
        bound = ENTRY_UBOUND

    # Mate score ply adjustment for TT storage
    stored_score = best_score
    if stored_score >= SCORE_WIN:
        stored_score += ply
    elif stored_score <= SCORE_LOST:
        stored_score -= ply

    # Age-aware TT replacement: replace if empty, different key, older generation, or deeper
    is_stale = (tt_ages[tt_idx] != current_age)
    if tt_keys[tt_idx] != key_32 or is_stale or depth >= int(tt_depths[tt_idx]):
        tt_keys[tt_idx] = key_32
        tt_scores[tt_idx] = np.int16(max(-32000, min(32000, stored_score)))
        tt_static_evals[tt_idx] = np.int16(max(-32000, min(32000, static_eval)))
        if best_move != MOVE_NONE:
            tt_moves[tt_idx] = np.uint16(best_move)
        tt_depths[tt_idx] = np.uint8(min(255, depth))
        tt_bounds[tt_idx] = np.uint8(bound)
        tt_ages[tt_idx] = np.uint8(current_age)

    return best_score
