"""The submission entrypoint. The platform imports this file and calls get_move."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Lock single thread execution per tournament rules
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from engine.book import PolyglotBook
from engine.nnue import load_weights
from engine.ponder import PonderManager
from engine.position import Position, set_fen
from engine.search import EngineSearch

ROOT_DIR = Path(__file__).parent
WEIGHTS_PATH = ROOT_DIR / "weights" / "weights_3nets.npz"
BOOK_PATH = ROOT_DIR / "assets" / "book.bin"

# Global engine state persisting across moves within the same game
nnue_weights = load_weights(WEIGHTS_PATH)
book = PolyglotBook(BOOK_PATH) if BOOK_PATH.exists() else None
searcher = EngineSearch(nnue_weights)
ponderer = PonderManager(searcher)

# Game history tracker for repetition detection across moves
game_history_keys = np.zeros(2048, dtype=np.uint64)
game_history_count = 0

# Fast JIT pre-compilation during import in < 0.5s
searcher.warmup()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion
    """
    global game_history_count

    try:
        # 1. Stop background ponder thread if active
        ponderer.stop()

        # 2. Record position in game history for repetition detection
        pos = Position()
        set_fen(pos, fen)
        if game_history_count < 2048:
            game_history_keys[game_history_count] = pos.zobrist_key
            game_history_count += 1

        # 3. Check opening book for instant theory moves
        if book is not None:
            book_move = book.probe(fen)
            if book_move is not None:
                ponderer.start_ponder(fen, book_move)
                return book_move

        # 4. Perform alpha-beta PVS search with dynamic time management & repetition check
        best_move_uci, predicted_reply = searcher.search_root(
            fen, time_left_ms, game_history_keys, game_history_count
        )

        # 5. Start background pondering on predicted opponent reply
        if predicted_reply is not None:
            ponderer.start_ponder(fen, best_move_uci, predicted_reply)

        return best_move_uci
    except Exception:
        import chess
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        return legal[0].uci() if legal else "0000"
