"""The submission entrypoint. The platform imports this file and calls get_move."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from engine.book import PolyglotBook
from engine.nnue import load_weights
from engine.ponder import PonderManager
from engine.search import EngineSearch

# Lock single thread execution per tournament rules
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 60-Second Init Budget: Load weights and pre-compile Numba JIT
ROOT_DIR = Path(__file__).parent
WEIGHTS_PATH = ROOT_DIR / "weights" / "weights_3nets.npz"
BOOK_PATH = ROOT_DIR / "assets" / "book.bin"

# Global engine state persisting across moves within the same game
nnue_weights = load_weights(WEIGHTS_PATH)
book = PolyglotBook(BOOK_PATH) if BOOK_PATH.exists() else None
searcher = EngineSearch(nnue_weights)
ponderer = PonderManager(searcher)

# Run warm-up search on 5 positions to JIT-compile all functions before clock starts
searcher.warmup()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion
    """
    try:
        # 1. Stop background ponder thread if active
        ponderer.stop()

        # 2. Check opening book for instant theory moves
        if book is not None:
            book_move = book.probe(fen)
            if book_move is not None:
                ponderer.start_ponder(fen, book_move)
                return book_move

        # 3. Perform alpha-beta PVS search with dynamic time management
        best_move_uci, predicted_reply = searcher.search_root(fen, time_left_ms)

        # 4. Start background pondering on predicted opponent reply
        if predicted_reply is not None:
            ponderer.start_ponder(fen, best_move_uci, predicted_reply)

        return best_move_uci
    except Exception:
        import chess
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        return legal[0].uci() if legal else "0000"
