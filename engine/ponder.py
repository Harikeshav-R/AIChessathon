"""Background Pondering Manager.

Leverages dedicated container core to search predicted opponent replies
during the opponent's turn, pre-populating TT and history tables.
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

        with contextlib.suppress(Exception):
            board = chess.Board(fen)
            m1 = chess.Move.from_uci(played_move)
            if m1 in board.legal_moves:
                board.push(m1)
                if predicted_reply is not None:
                    m2 = chess.Move.from_uci(predicted_reply)
                    if m2 in board.legal_moves:
                        board.push(m2)

                ponder_fen = board.fen()
                self.last_ponder_fen = ponder_fen

                def _worker() -> None:
                    with contextlib.suppress(Exception):
                        self.searcher.search_ponder(ponder_fen, self.stop_event)

                self.thread = threading.Thread(target=_worker, daemon=True)
                self.thread.start()

    def stop(self) -> None:
        """Stops active background ponder thread."""
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.05)
        self.thread = None
