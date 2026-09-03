"""Polyglot Opening Book Reader.

Probes binary Polyglot opening books on-demand via binary search with 0 init overhead.
"""

from __future__ import annotations

import contextlib
import random
from pathlib import Path

import chess
import chess.polyglot


class PolyglotBook:
    """Reads and probes Polyglot .bin opening books on demand."""

    def __init__(self, book_path: Path) -> None:
        self.book_path = book_path
        self.exists = book_path.exists()

    def probe(self, fen: str) -> str | None:
        """Probes opening book for position FEN using weighted choice."""
        if not self.exists:
            return None

        with contextlib.suppress(Exception):
            board = chess.Board(fen)
            with chess.polyglot.open_reader(str(self.book_path)) as reader:
                entries = list(reader.find_all(board))
                if not entries:
                    return None

                moves = [e.move for e in entries]
                weights = [max(1, e.weight) for e in entries]
                selected_move = random.choices(moves, weights=weights, k=1)[0]
                return selected_move.uci()

        return None
