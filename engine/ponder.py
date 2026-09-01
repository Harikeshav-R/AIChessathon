"""Background Pondering Manager.

Leverages dedicated container core to search predicted opponent replies
during the opponent's turn.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any

import chess


class PonderManager:
    """Manages background search thread on opponent's time."""

    def __init__(self, searcher: Any) -> None:
        self.searcher = searcher
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.last_ponder_fen: str | None = None

    def start_ponder(self, fen: str, played_move: str, predicted_reply: str | None = None) -> None:
        """Starts background search on expected position after opponent moves."""
        self.stop()
        self.stop_event.clear()

        try:
            board = chess.Board(fen)
            move_obj = chess.Move.from_uci(played_move)
            if move_obj not in board.legal_moves:
                return
            board.push(move_obj)

            if predicted_reply is not None:
                reply_obj = chess.Move.from_uci(predicted_reply)
                if reply_obj in board.legal_moves:
                    board.push(reply_obj)

            if board.is_game_over():
                return

            ponder_fen = board.fen()
            self.last_ponder_fen = ponder_fen
        except Exception:
            return

        def _worker() -> None:
            with contextlib.suppress(Exception):
                self.searcher.search_ponder(ponder_fen, self.stop_event)

        self.thread = threading.Thread(target=_worker, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stops active background ponder thread."""
        self.stop_event.set()
        if hasattr(self.searcher, "stop_flag"):
            self.searcher.stop_flag[0] = 1
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        self.thread = None
